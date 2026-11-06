#include "llama-context.h"
#include "llama-model.h"
#include "llama-kv-cache.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"
#include "llama-pipeline-reserve.h"
#include "pipeline-shapes.hpp"
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <map>
#include <set>
#include <limits>
#include <cinttypes>

bool llama_context::set_pipeline_peer(const llama_context * peer) {
    if (decode_pipelined || !pipe_scheds.empty()) { return false; }
    pipeline_has_peer = peer != nullptr;
    pipeline_peer_lifetime = peer ? peer->pipeline_lifetime : std::weak_ptr<llama_pipeline_context_state>();
    return true;
}

void llama_context::set_decode_pipeline_lanes(uint32_t lanes) {
    GGML_ASSERT(!decode_pipelined && pipe_scheds.empty() && lanes > 0);
    n_pipe_scheds = lanes + 1;
}

void llama_context::set_decode_pipelined(bool enable) {
    if (decode_pipelined == enable) {
        return;
    }

    std::shared_ptr<llama_pipeline_context_state> peer_state;
    std::unique_lock<std::mutex> peer_guard;
    const llama_context * peer = nullptr;
    std::set<const llama_model *> configuration_models;
    std::vector<std::unique_lock<std::recursive_mutex>> configuration_guards;
    if (enable && !pipeline_configuration_frozen) {
        if (pipeline_has_peer) {
            peer_state = pipeline_peer_lifetime.lock();
            if (!peer_state) { throw std::runtime_error("ParaFlow draft context was released before startup"); }
            peer_guard = std::unique_lock<std::mutex>(peer_state->mutex);
            peer = peer_state->context;
            if (!peer) { throw std::runtime_error("ParaFlow draft context was released before startup"); }
        }
        configuration_models.insert(&model);
        if (peer) { configuration_models.insert(&peer->model); }
        for (const auto * configured : configuration_models) {
            configuration_guards.emplace_back(configured->pipeline_configuration->mutex);
        }
    }

    // quiesce before switching modes: in-flight decodes must complete with the
    // output routing they were issued with
    synchronize();
    const bool lazy_primary = !cparams.training && !cparams.embeddings &&
        cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP && !model.hparams.no_alloc;
    if (enable && lazy_primary) {
        for (auto * backend : backend_ptrs) {
            fprintf(stderr, "pipeline primary workspace released: backend=%s bytes=%zu\n", ggml_backend_name(backend),
                    ggml_backend_sched_get_buffer_size(sched.get(), backend));
        }
        for (auto & res : gf_res_prev) { res.reset(); }
        gf_res_prev_active = nullptr;
        sched_new(graph_max_nodes(std::min(cparams.n_ctx, cparams.n_ubatch)), cparams.pipeline_parallel);
        std::fill(backend_buf_exp_size.begin(), backend_buf_exp_size.end(), 0);
        sched_need_reserve = true;
    }


    if (enable) {
        if (pipe_scheds.empty()) {
            if (n_pipe_scheds == 0) { n_pipe_scheds = pipeline_stages() + 3; }
            pipe_scheds.resize(n_pipe_scheds);
            pipe_gf_res.resize(n_pipe_scheds);
            n_decode_tickets = std::max(8, 2 * std::max(n_pipe_scheds, pipeline_stages() + 3));
            tickets.resize(n_decode_tickets);
        }
        // private (sched, graph result) ring for generation-sized decodes; sized for
        // one token per sequence (larger batches stay on the main sched)
        llama_memory_context_ptr mctx_full;
        if (memory) { mctx_full = memory->init_full(); }
        auto * reserve_kv = dynamic_cast<llama_kv_cache_context *>(mctx_full.get());
        if (auto * hybrid = dynamic_cast<llama_memory_hybrid_context *>(mctx_full.get())) {
            reserve_kv = const_cast<llama_kv_cache_context *>(hybrid->get_attn());
        }
        const bool tight_reserve = reserve_kv || dynamic_cast<llama_memory_recurrent_context *>(mctx_full.get());
        const uint32_t old_tokens_max = std::max<uint32_t>(cparams.n_seq_max, cparams.n_outputs_max);
        const uint32_t n_tokens_max = tight_reserve ? std::min(old_tokens_max, pipeline_verify_ubatch()) : old_tokens_max;
        const uint32_t n_seqs_max = std::min(std::max<uint32_t>(cparams.n_seq_max, 1), n_tokens_max);
        const auto reserve_shapes = llama_pipeline_reserve_shapes(n_tokens_max, n_seqs_max, tight_reserve);
        const size_t max_nodes = graph_max_nodes(n_tokens_max);

        for (int i = 0; i < n_pipe_scheds; ++i) {
            if (!pipe_scheds[i]) {
                pipe_scheds[i].reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(),
                        backend_ptrs.size(), max_nodes, cparams.pipeline_parallel, cparams.op_offload));
                pipe_gf_res[i].reset(new llm_graph_result(max_nodes));

                // Reserve the normal lane's generation shapes up front. The last
                // scheduler remains sized for an untagged global batch. A later,
                // unusually unbalanced lane can still use the scheduler's existing
                // synchronized allocation-growth path; lane buffers never alias.
                sched_cur  = pipe_scheds[i].get();
                gf_res_cur = pipe_gf_res[i].get();

                const bool lane_scheduler = tight_reserve && !cparams.embeddings &&
                        i + 1 < n_pipe_scheds && n_pipe_scheds > 1;
                const uint32_t lanes = std::max(n_pipe_scheds - 1, 1);
                const uint32_t lane_seqs = std::max<uint64_t>(1,
                        (uint64_t(cparams.n_seq_max) + lanes - 1) / lanes);
                const uint64_t rows_per_seq = 1 + uint64_t(cparams.n_rs_seq);
                const uint32_t reserve_tokens = lane_scheduler ?
                        std::min<uint64_t>(n_tokens_max, uint64_t(lane_seqs) * rows_per_seq) : n_tokens_max;
                const uint32_t reserve_seqs = lane_scheduler ? std::min(n_seqs_max, lane_seqs) : n_seqs_max;
                auto lane_shapes = lane_scheduler ?
                        llama_pipeline_reserve_shapes(reserve_tokens, reserve_seqs, tight_reserve) : reserve_shapes;
                if (lane_scheduler) {
                    // Ordinary decode never has multiple rows per sequence; MTP
                    // verification has at most one plus the configured draft cap.
                    // Avoid reserving unreachable prompt-like recurrent graphs.
                    lane_shapes.erase(std::remove_if(lane_shapes.begin(), lane_shapes.end(),
                            [&](const auto & shape) {
                                return uint64_t(shape.first) > uint64_t(shape.second) * rows_per_seq;
                            }), lane_shapes.end());
                }
                const bool lazy_global = tight_reserve && !cparams.embeddings && cparams.pipeline_parallel &&
                        i + 1 == n_pipe_scheds && n_pipe_scheds > 1;
                if (lazy_global) {
                    // The untagged allocator remains distinct. Its first real graph uses
                    // the existing synchronized allocation/growth path. Reserving it up
                    // front would cost a full lane workspace on every device.
                    lane_shapes.clear();
                } else {
                    GGML_ASSERT(!lane_shapes.empty());
                }
                for (const auto & shape : lane_shapes) {
                    if (reserve_kv) reserve_kv->set_reserve_streams(shape.second);
                    if (!graph_reserve(shape.first, shape.second, shape.first, mctx_full.get())) {
                        LLAMA_LOG_WARN("%s: failed to reserve compute buffers for pipe slot %d (%u tokens, %u sequences)\n",
                                __func__, i, shape.first, shape.second);
                    }
                }
            }
        }

        if (reserve_kv) reserve_kv->set_reserve_streams(0);

        // private backend instances (own low-priority CUDA streams) where supported;
        // shared instances otherwise. with private instances, chunk kernels overlap
        // with decode-lane kernels instead of queueing in front of them, and the
        // lanes' blocking input syncs no longer wait out the queued chunks.
        // fills `ptrs` with one backend per device (private where possible, shared
        // otherwise); `owned` keeps the new instances.

        auto build_pf_backends = [&](std::vector<ggml_backend_ptr> & owned, std::vector<ggml_backend_t> & ptrs) {
            owned.clear();
            ptrs.clear();
            for (auto * b : backend_ptrs) {
                ggml_backend_t nb = nullptr;

                ggml_backend_dev_t dev = ggml_backend_get_device(b);
                if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);

                    auto fn_init_background = reg ?
                        (ggml_backend_t (*)(ggml_backend_dev_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_init_background") : nullptr;
                    if (fn_init_background) {
                        nb = fn_init_background(dev);
                    }
                }

                if (nb) {
                    owned.emplace_back(nb);
                    ptrs.push_back(nb);
                } else {
                    ptrs.push_back(b);
                }
            }
        };

        if (!prefill_sched) {
            const uint32_t n_tokens_pp  = std::min(cparams.n_ctx, cparams.n_ubatch);
            const size_t   max_nodes_pp = graph_max_nodes(n_tokens_pp);

            build_pf_backends(prefill_backends, prefill_backend_ptrs);
            if (!prefill_backends.empty()) {
                LLAMA_LOG_INFO("%s: prefill sched uses %zu private background backends\n", __func__, prefill_backends.size());
            }

            prefill_sched.reset(ggml_backend_sched_new(prefill_backend_ptrs.data(), backend_buft.data(),
                    prefill_backend_ptrs.size(), max_nodes_pp, /*parallel =*/ true, cparams.op_offload));
            prefill_gf_res.reset(new llm_graph_result(max_nodes_pp));

            sched_cur       = prefill_sched.get();
            gf_res_cur      = prefill_gf_res.get();
            pf_cur          = true;
            pf_cur_backends = &prefill_backend_ptrs;

            if (!graph_reserve(n_tokens_pp,
                            llama_prefill_initial_reserve_sequences(model, cparams, cparams.n_seq_max),
                            llama_prefill_initial_reserve_sequences(model, cparams, cparams.n_seq_max) != cparams.n_seq_max
                                ? std::min(n_tokens_pp, cparams.n_outputs_max) : n_tokens_pp,
                            mctx_full.get())) {
                LLAMA_LOG_WARN("%s: failed to reserve compute buffers for the prefill sched\n", __func__);
                prefill_sched.reset();
                prefill_gf_res.reset();
                prefill_backends.clear();
                prefill_backend_ptrs.clear();
            }
        }
    }

    sched_cur       = nullptr;
    gf_res_cur      = nullptr;
    pf_cur          = false;
    pf_cur_backends = nullptr;

    if (enable && !pipeline_configuration_frozen) {
        std::vector<std::shared_ptr<llama_pipeline_model_state>> frozen;
        for (const auto * configured : configuration_models) { frozen.push_back(configured->pipeline_configuration); }
        for (const auto & configured : frozen) {
            if (configured->users == std::numeric_limits<size_t>::max()) {
                throw std::runtime_error("ParaFlow configuration reference overflow");
            }
        }
        for (const auto & configured : frozen) { ++configured->users; }
        pipeline_frozen_models = std::move(frozen);
        pipeline_configuration_frozen = true;
    }

    decode_pipelined = enable;
    if (!enable && lazy_primary) { sched_need_reserve = true; }

    LLAMA_LOG_INFO("%s: pipelined decode %s\n", __func__, enable ? "enabled" : "disabled");
}

void llama_context::set_decode_pipelane(int32_t lane) {
    pipe_lane_hint = lane;
}

uint64_t llama_context::decode_ticket_last() const {
    return ticket_seq;
}

int32_t llama_context::decode_ticket_wait(uint64_t ticket) {
    if (ticket == 0 || ticket > ticket_seq) {
        return -1;
    }

    auto & tk = tickets[ticket % n_decode_tickets];
    if (tk.id != ticket) {
        // recycled: the caller let too many decodes pass before waiting
        return -1;
    }

    if (tk.fallback || !tk.event_recorded) {
        synchronize();
        return 0;
    }

    for (const auto & c : tk.completions) {
        if (c.used) { ggml_backend_event_synchronize(c.event); }
    }

    return 0;
}

bool llama_context::decode_ticket_select(uint64_t ticket) {
    if (ticket == 0) {
        selected_ticket = 0;
        return true;
    }
    if (decode_ticket_wait(ticket) != 0) {
        return false;
    }
    auto & tk = tickets[ticket % n_decode_tickets];
    if (tk.fallback) {
        return false;
    }
    selected_ticket = ticket;
    return true;
}

bool llama_context::decode_ticket_retain(uint64_t ticket) {
    if (!ticket || !n_decode_tickets || ticket > ticket_seq) { return false; }
    auto & tk = tickets[ticket % n_decode_tickets];
    if (tk.id != ticket || tk.fallback) { return false; }
    tk.retained = true;
    return true;
}

bool llama_context::decode_ticket_release(uint64_t ticket) {
    if (!ticket || !n_decode_tickets || ticket > ticket_seq) { return false; }
    auto & tk = tickets[ticket % n_decode_tickets];
    if (tk.id != ticket || !tk.retained) { return false; }
    tk.retained = false;
    return true;
}

int32_t llama_context::decode_ticket_retained_count() const {
    return std::count_if(tickets.begin(), tickets.end(), [](const decode_ticket & tk) { return tk.retained; });
}

bool llama_context::decode_ticket_ready(uint64_t ticket) const {
    if (ticket == 0 || ticket > ticket_seq) {
        return true; // wait would fail fast
    }

    const auto & tk = tickets[ticket % n_decode_tickets];
    if (tk.id != ticket) {
        return true; // recycled: wait would fail fast
    }

    if (tk.fallback || !tk.event_recorded) {
        return true; // wait degrades to a full sync either way
    }

    for (const auto & c : tk.completions) {
        if (c.used && !ggml_backend_event_query(c.event)) { return false; }
    }
    return true;
}

float * llama_context::decode_ticket_logits_ith(uint64_t ticket, int32_t i) {
    if (ticket == 0 || ticket > ticket_seq) {
        return nullptr;
    }

    auto & tk = tickets[ticket % n_decode_tickets];
    if (tk.id != ticket) {
        return nullptr;
    }

    if (tk.fallback) {
        // outputs went to the main buffer (already synchronized by ticket_wait)
        return get_logits_ith(i);
    }

    const int32_t row = i < 0 ? tk.n_outputs + i :
            (size_t) i < tk.output_ids.size() ? tk.output_ids[i] : -1;
    if (row < 0 || row >= tk.n_outputs) {
        return nullptr;
    }

    return tk.data + (size_t) row * model.vocab.n_tokens();
}

