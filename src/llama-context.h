#pragma once

#include "llama.h"
#include "llama-ext.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-impl.h"
#include "llama-memory.h"
#include "llama-moe-cache.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <array>
#include <map>
#include <vector>

struct llama_model;
class llama_batch_allocr;
class llama_moe_cache;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// stores copy of the memory in device buffer. used for fast state save/load
struct llama_memory_buffer {
    int n_tensors = 0;
    size_t total_size = 0;

    ggml_backend_buffer_ptr buf;

    ggml_context_ptr ctx;

    std::vector<ggml_tensor *> org;
    std::vector<ggml_tensor *> cpy;
};

using llama_memory_buffers = std::map<ggml_backend_buffer_type_t, llama_memory_buffer>;

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    void synchronize();

    llama_pipeline_subbatch_stats get_pipeline_subbatch_stats() const;

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    float * get_embeddings_nextn();
    float * get_embeddings_nextn_ith(int32_t i);

    float * get_embeddings_layer_inp(uint32_t lid);

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    bool get_causal_attn() const;

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_embeddings_nextn(bool value, bool masked);
    void set_embeddings_layer_inp(uint32_t lid, bool enable);
    void set_nextn_layer_offset(int32_t offset);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    int encode(const llama_batch_ext & batch_inp);
    int decode(const llama_batch_ext & batch_inp);

    // compat version
    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    //
    // pipelined decode (decode tickets)
    //

    bool     decode_pipelined_supported() const;
    int32_t  pipeline_stages() const;
    void     set_input_snapshots(bool enable);
    ggml_backend_input_snapshot_stats get_input_snapshot_stats() const;
    ggml_backend_input_snapshot_stats get_generation_input_snapshot_stats() const;
    ggml_backend_input_snapshot_stats get_prefill_input_snapshot_stats() const;
    void     set_decode_pipeline_lanes(uint32_t lanes);
    void     set_decode_pipelined(bool enable);
    void     set_decode_pipelane(int32_t lane);
    uint64_t decode_ticket_last() const;
    int32_t  decode_ticket_wait(uint64_t ticket);
    bool     decode_ticket_ready(uint64_t ticket) const;
    float *  decode_ticket_logits_ith(uint64_t ticket, int32_t i);
    bool     decode_ticket_select(uint64_t ticket);
    bool     decode_ticket_retain(uint64_t ticket);
    bool     decode_ticket_release(uint64_t ticket);
    int32_t  decode_ticket_retained_count() const;
    bool     decode_ticket_selected() const { return selected_ticket != 0; }
    int32_t  prefill_inflight();
    int32_t  pf_ring_active() const { return pf_ring_n; }

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);

    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    // V41: wait only for unfinished pipelined work that may touch seq_id (else full sync); 1 = full sync used
    int32_t state_seq_prepare_capture(llama_seq_id seq_id);
    int64_t state_seq_get_data_async(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags);
    int32_t state_seq_async_wait(int64_t handle, bool block);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    llama_memory_breakdown memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    // async-copy enabled layer-input tensors (per cparams.output_layer_inp)
    // from backend into host-side embd_layer_inp buffers
    bool extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens);

    //
    // graph
    //

public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

private:
    llm_graph_result * get_gf_res_prev();

    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    llm_graph_cb graph_get_cb() const;

    // create a new backend scheduler for a graph of the specified size and register the callbacks below
    void sched_new(size_t max_nodes, bool parallel);

    // ggml_backend_sched copy callback, copies only the experts used by MUL_MAT_ID and updates the MoE cache
    static bool sched_copy_experts(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph, void * user_data);

    // ggml_backend_sched realloc callback, counts and traces the graph reallocations of this context
    static void sched_graph_realloc(ggml_backend_sched_t sched, ggml_cgraph * gf, bool backend_ids_changed, bool graph_size_changed, void * user_data);

    // disable auto fused ops (Flash Attention, Gated Delta Net) whose op lands on a device
    // that differs from the layer it belongs to (usually due to missing backend support)
    void resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs);

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    llama_cparams cparams;

    llama_adapter_cvec_ptr  cvec;
    llama_adapter_loras_ptr loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    llama_memory_ptr memory;
    llama_moe_cache_ptr moe_cache;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    // hidden state required by the nextn layers (2-dimensional array: [n_outputs][n_embd])
    // populated only when cparams.embeddings_nextn is enabled and the model graph
    // sets llm_graph_result::t_h_nextn
    buffer_view<float> embd_nextn = {nullptr, 0};

    // host buffers for output layer input embeddings, per layer
    // populated when cparams.output_layer_inp[il] is true
    std::vector<buffer_view<float>> embd_layer_inp;
    std::vector<int32_t> embd_batch_idxs; // extracted index -> original batch index

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_input_tensors = 0; // number of tensors marked as input during the last graph reserve
    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    bool sched_need_reserve = true;

    // whether the reserve graph has a logits tensor, set in sched_reserve()
    // true until then, because the first output_reserve() runs before any graph is reserved
    bool graph_has_logits = true;

    // state of sched_copy_experts, reset before each graph compute
    struct copy_experts_info {
        const ggml_tensor *  ids = nullptr;
        std::vector<int32_t> ids_data;
        std::vector<bool>    used;

        void reset() {
            ids = nullptr;
            ids_data.clear();
            used.clear();
        }
    };

    copy_experts_info copy_experts;

    // graph reallocations during inference where the backend assignments did not change, split by whether the
    // graph size is the same as the previously scheduled one (unexpected) or different
    int32_t n_reallocs_same = 0;
    int32_t n_reallocs_diff = 0;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    // Separate arenas give batches with and without outputs distinct CUDA graph cache keys.
    std::array<llm_graph_result_ptr, 2> gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    llm_graph_result * gf_res_prev_active = nullptr;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    //
    // pipelined decode (decode tickets)
    //

    // per-decode output snapshot: private pinned host slab for logits + completion event,
    // so the caller can read the result of one decode while later decodes are in flight
    struct decode_ticket {
        uint64_t id = 0;                  // 0 = unused
        bool retained = false;            // caller owns the outputs until explicit release
        bool fallback = false;            // outputs went to the main buffer; wait = full sync
        bool event_recorded = false;
        int32_t n_outputs = 0;
        struct completion {
            ggml_backend_t backend = nullptr;
            ggml_backend_event_t event = nullptr;
            bool used = false;
        };
        std::vector<completion> completions;

        void use_backend(ggml_backend_t backend) {
            for (auto & c : completions) {
                if (c.backend == backend) {
                    if (!c.event) { c.event = ggml_backend_event_new(ggml_backend_get_device(backend)); }
                    c.used = true;
                    return;
                }
            }
            completions.push_back({backend, ggml_backend_event_new(ggml_backend_get_device(backend)), true});
        }
        std::vector<int32_t> output_ids;  // batch index -> slab row (-1 = none)
        ggml_backend_buffer_ptr buf;      // pinned host slab
        float * data = nullptr;
        size_t  size = 0;                 // capacity in floats
        float * hidden = nullptr;        // dense, original batch token order
        size_t hidden_rows = 0;
    };

    // Retained lane outputs cannot be recycled while faster lanes submit again.
    // Unretained prompt/legacy tickets keep the existing rotating-slab policy.
    int n_decode_tickets = 0;
    std::vector<decode_ticket> tickets;
    uint64_t      selected_ticket  = 0;
    uint64_t      ticket_seq       = 0;     // id of the most recently issued ticket
    bool          decode_pipelined = false;

    // pipelined decode: ring of private (sched, graph result) pairs, one per ticket
    // slot. generation-sized decodes rotate through the ring, and each slot reuses
    // its own graph across decodes: no per-decode rebuild/re-split, stable tensor
    // objects (so backend-level graph capture such as CUDA graphs can warm up), and
    // no input hazard - a slot's previous decode has always completed by the time
    // the slot comes around again (the server waits a lane's ticket before reissue).
    // decoupled from n_decode_tickets: tickets are cheap (a pinned slab), scheds
    // hold per-device compute buffers. slots 0..n-2 are keyed by lane, slot n-1 is
    // shared by out-of-lane generation-sized decodes. sized for one lane more than
    // there are stages: an extra lane hides the CPU sample+issue time in the rotation.
    int n_pipe_scheds = 0;
    std::vector<ggml_backend_sched_ptr> pipe_scheds;
    std::vector<llm_graph_result_ptr> pipe_gf_res;

    // dedicated always-parallel sched for prompt-sized pipelined decodes (interleaved
    // chunks): the main sched can run without pipeline parallelism (n_copies == 1, no
    // events), where every user-input split copy fully synchronizes its backend -
    // consecutive chunks then serialize and all but one pipeline stage idles during
    // interleaved prefill. graphs on this sched are always rebuilt (no reuse): the
    // input-copy rotation makes reuse + direct set_inputs unsafe.
    // private backend instances for the prefill sched where the backend supports it
    // (CUDA: own streams at default priority, while the regular backends run at the
    // device's greatest priority). chunk kernels then no longer queue in front of
    // decode-lane kernels on a shared stream. entries missing here fall back to the
    // shared instances in backend_ptrs. declared before the sched: it must outlive it
    // (ggml_backend_sched_free synchronizes its backends).
    std::vector<ggml_backend_ptr> prefill_backends;
    std::vector<ggml_backend_t>   prefill_backend_ptrs;

    ggml_backend_sched_ptr prefill_sched;
    llm_graph_result_ptr   prefill_gf_res;

    // fence ring over the prefill backends: one slot per submitted chunk, an event
    // per GPU backend. lets the server bound the number of chunks in flight -
    // unbounded queueing throttles cudaLaunchKernel and stalls the submission thread
    struct pf_fence {
        std::vector<std::pair<ggml_backend_event_t, ggml_backend_t>> events;
        const void * key = nullptr;  // actual backend set the events were built for
        bool pending = false;
    };
    static constexpr int n_pf_fences = 8;

    pf_fence pf_fences[n_pf_fences];
    int      pf_fence_next = 0;

    // optional ring of independent prompt-sized scheds for prefill chunks
    // (LLAMA_PIPE_PF_RING=N, default 0 = single prefill_sched). each ring slot owns
    // its own private background backends (own CUDA streams); consecutive chunks
    // routed round-robin land on different streams and overlap across the pipeline
    // stages instead of serializing through one sched's input-copy rotation. when
    // enabled it replaces prefill_sched for routing (prefill_sched stays allocated
    // for fallback / structural detection). prompt chunks always rebuild their graph.
    static constexpr int n_pf_ring_max = 4;
    std::vector<ggml_backend_ptr> pf_ring_backends[n_pf_ring_max];
    std::vector<ggml_backend_t>   pf_ring_backend_ptrs[n_pf_ring_max];
    ggml_backend_sched_ptr        pf_ring_scheds[n_pf_ring_max];
    llm_graph_result_ptr          pf_ring_gf_res[n_pf_ring_max];
    int pf_ring_n    = 0;   // active ring size (0 = disabled, use prefill_sched)
    int pf_ring_next = 0;   // round-robin cursor

    // overrides for the current decode (null = use the main sched / gf_res_prev)
    ggml_backend_sched_t sched_cur  = nullptr;
    llm_graph_result *   gf_res_cur = nullptr;

    // current decode runs on a prefill sched (single prefill_sched or a ring slot):
    // graphs rebuild (parallel input-copy rotation), norm pinning + the prefill fence
    // use the actual backend set (private where available, shared otherwise). set
    // during routing, cleared per decode.
    bool                          pf_cur          = false;
    std::vector<ggml_backend_t> * pf_cur_backends = nullptr;  // for norm pinning (private + shared)

    // ring slot selector: lanes map to slots 0..n-2 (stable sequence sets, so their
    // graphs are reusable); -1 selects the last slot, shared by out-of-lane decodes
    int32_t pipe_lane_hint = -1;

    ggml_backend_sched_t cur_sched() const {
        return sched_cur ? sched_cur : sched.get();
    }

    // keep copies of the per-sequence memory on the device
    std::map<llama_seq_id, llama_memory_buffers> mem_storage;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // true after the prism.hadamard coverage check passes on a reserved graph
    bool hadamard_verified = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused

    // An ordinal belongs to exactly one physical ubatch within a lane's ticket.
    // Separate arenas avoid overwriting any earlier ubatch still on the devices.
    struct pipeline_subbatch_slot {
        ggml_backend_sched_ptr scheduler;
        llm_graph_result_ptr result;
        bool attempted = false;
        uint32_t reserved_tokens = 0;
        uint32_t reserved_seqs = 0;
    };
    std::vector<std::vector<pipeline_subbatch_slot>> pipe_subbatches;
    llama_pipeline_subbatch_stats pipe_subbatch_stats;
    int pipe_subbatch_graph_kind = 0; // 0: unrelated, 1: primary, 2: auxiliary
    pipeline_subbatch_slot * ensure_pipeline_subbatch(int lane, uint32_t ordinal, uint32_t tokens, uint32_t seqs);
};
