#include "llama-memory-recurrent.h"
#include "llama-pipeline-reserve.h"

#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-batch.h"
#include "llama-model.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <map>
#include <stdexcept>

//
// llama_memory_recurrent
//

llama_memory_recurrent::llama_memory_recurrent(
        const llama_model & model,
                ggml_type   type_r,
                ggml_type   type_s,
                     bool   offload,
                 uint32_t   mem_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
    const layer_filter_cb & filter) : hparams(model.hparams), n_seq_max(n_seq_max) {
    const int32_t n_layer = hparams.n_layer();

    head = 0;
    size = mem_size;
    used = 0;

    this->n_rs_seq = n_rs_seq;
    rs_idx.assign(n_seq_max, 0);
    const char * flag = std::getenv("PARAFLOW_RECURRENT_REPLAY");
    if (!flag) { flag = std::getenv("PARAFLOW_DIAG_RS_DEFERRED"); } // Older validation scripts.
    if (flag && strcmp(flag, "0") != 0 && strcmp(flag, "1") != 0 && strcmp(flag, "auto") != 0) {
        throw std::runtime_error("PARAFLOW_RECURRENT_REPLAY must be auto, 0, or 1");
    }
    const int64_t dim = hparams.ssm_d_state;
    const int64_t hk = hparams.ssm_n_group, hv = hparams.ssm_dt_rank;
    const bool shape_supported = (dim == 16 || dim == 32 || dim == 64 || dim == 128) &&
            hk > 0 && hk <= 1024 && hv > 0 && hv <= 1024 && hv % hk == 0 &&
            dim * hv == hparams.ssm_d_inner && dim * dim * hv == hparams.n_embd_s();
    compact_s = (!flag || strcmp(flag, "0") != 0) && n_rs_seq > 1 && n_rs_seq < 256 &&
            mem_size > 0 && mem_size <= 65535 && model.arch == LLM_ARCH_QWEN35 &&
            offload && type_s == GGML_TYPE_F32 && shape_supported;
    std::vector<bool> layer_enabled(n_layer);
    for (int i = 0; i < n_layer; ++i) {
        layer_enabled[i] = !filter || filter(i);
        if (compact_s && layer_enabled[i]) {
            auto * dev = model.dev_layer(i);
            compact_s = dev && strncmp(ggml_backend_dev_name(dev), "CUDA", 4) == 0;
        }
    }

    cells.clear();
    cells.resize(mem_size);

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    // create a context for each buffer type
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                // r and s per layer, plus the separate PLE conv row where the model has one
                /*.mem_size   =*/ size_t((compact_s ? 4u : (hparams.ple_conv_state() > 0 ? 3u : 2u))*n_layer*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    r_l.resize(n_layer);
    s_l.resize(n_layer);
    j_l.resize(n_layer);
    p_l.resize(n_layer);

    for (int i = 0; i < n_layer; i++) {
        if (!layer_enabled[i]) {
            LLAMA_LOG_DEBUG("%s: layer %3d: skipped\n", __func__, i);
            continue;
        }

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(i);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s, layer %3d: dev = %s\n", __func__, i, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for rs cache");
        }

        const uint32_t n_rows = mem_size * (1 + n_rs_seq);
        ggml_tensor * r = ggml_new_tensor_2d(ctx, type_r, hparams.n_embd_r(), n_rows);
        ggml_tensor * s = ggml_new_tensor_2d(ctx, type_s, hparams.n_embd_s(), compact_s ? 2 * mem_size : n_rows);
        if (compact_s) {
            const auto * name = ggml_backend_dev_name(ggml_backend_buft_get_device(buft));
            if (strncmp(name, "CUDA", 4) != 0) {
                GGML_ABORT("recurrent replay requires native CUDA layers");
            }
            const int64_t features = hparams.ssm_d_state * hparams.ssm_n_group +
                    hparams.ssm_d_inner + 2 * hparams.ssm_dt_rank;
            j_l[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, features * (1 + n_rs_seq), mem_size);
            ggml_format_name(j_l[i], "cache_j_l%d", i);
        }
        ggml_format_name(r, "cache_r_l%d", i);
        ggml_format_name(s, "cache_s_l%d", i);
        r_l[i] = r;
        s_l[i] = s;

        // the PLE history needs its own row: Meta must mirror it while the delta-net conv state next door stays split
        if (hparams.ple_conv_state() > 0 && hparams.is_ple(i)) {
            ggml_tensor * p = ggml_new_tensor_2d(ctx, type_r, hparams.ple_conv_state(), n_rows);
            ggml_format_name(p, "cache_ple_r_l%d", i);
            p_l[i] = p;
        }
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for rs cache");
        }
        ggml_backend_buffer_clear(buf, 0);
        LLAMA_LOG_INFO("%s: %10s RS buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    if (is_empty()) {
        compact_s = false;
        if (n_rs_seq > 0) {
            n_rs_seq = 0;
            LLAMA_LOG_INFO("%s: disabling rollback snapshots because the memory module is empty\n", __func__);
        }
    }

    {
        const size_t memory_size_r = size_r_bytes();
        const size_t memory_size_s = size_s_bytes();
        const size_t memory_size_p = size_p_bytes();
        size_t memory_size_j = 0;
        for (const auto * journal : j_l) { if (journal) { memory_size_j += ggml_nbytes(journal); } }
        if (compact_s) {
            LLAMA_LOG_INFO("DEFERRED_RS_STORAGE matrix_planes=2 conv_planes=%u journal_MiB=%.4f\n",
                    n_rs_seq + 1, memory_size_j / (1024.0 * 1024.0));
        }

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u seqs %2u rs_seq), R (%s): %7.2f MiB, S (%s): %7.2f MiB, P (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_r + memory_size_s + memory_size_p + memory_size_j) / (1024.0f * 1024.0f), mem_size, n_layer, n_seq_max, n_rs_seq,
                ggml_type_name(type_r), (float)memory_size_r / (1024.0f * 1024.0f),
                ggml_type_name(type_s), (float)memory_size_s / (1024.0f * 1024.0f),
                ggml_type_name(type_r), (float)memory_size_p / (1024.0f * 1024.0f));
    }
}

void llama_memory_recurrent::clear(bool data) {
    for (int32_t i = 0; i < (int32_t) size; ++i) {
        cells[i].pos = -1;
        cells[i].seq_id.clear();
        cells[i].src = -1;
        cells[i].tail = -1;
        cells[i].journal_length = cells[i].journal_input_length = 0;
    }

    head = 0;
    used = 0;

    if (data) {
        for (auto & [_, buf] : ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }

    std::fill(rs_idx.begin(), rs_idx.end(), 0);
}

bool llama_memory_recurrent::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    uint32_t new_head = size;

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if ((uint32_t) seq_id >= this->n_seq_max) {
        LLAMA_LOG_ERROR("%s: invalid seq_id (%d) - larger than n_seq_max (%d)\n", __func__, seq_id, this->n_seq_max);
        return false;
    }

    const bool rm_all = p0 == 0 && p1 == std::numeric_limits<llama_pos>::max();
    if (rm_all) {
        set_rs_idx(seq_id, 0);
    }

    // models like Mamba or RWKV can't have a state partially erased at the end
    // of the sequence because their state isn't preserved for previous tokens
    if (seq_id >= (int64_t) size) {
        // could be fatal
        return false;
    }
    if (0 <= seq_id) {
        int32_t & tail_id = cells[seq_id].tail;
        if (tail_id >= 0) {
            auto & cell = cells[tail_id];

            // partial rollback via per-token snapshot index (bounded by n_rs_seq)
            if (0 < p0 && p0 <= cell.pos && p1 > cell.pos) {
                // a cell shared by several sequences cannot move back for only one of them
                if (cell.seq_id.size() > 1) {
                    return false;
                }
                // the filter kept no layer (e.g. an MTP draft context), so only the position moves back
                if (is_empty()) {
                    cell.pos = p0 - 1;
                    return true;
                }
                const llama_pos rollback = cell.pos - (p0 - 1);
                // pending rollback is single-use
                const bool pending = rs_idx[seq_id] != 0;
                if (!pending && rollback >= 1 && rollback <= (llama_pos) n_rs_seq) {
                    if (compact_s && (uint32_t) rollback > cell.journal_length) {
                        LLAMA_LOG_ERROR("DEFERRED_RS_UNSUPPORTED_ROLLBACK seq=%d rollback=%d span=%u\n",
                                seq_id, (int) rollback, cell.journal_length);
                        return false;
                    }
                    set_rs_idx(seq_id, (uint32_t) rollback);
                    cell.pos = p0 - 1;
                    return true;
                }
                return false;
            }
            // invalidate tails which will be cleared
            if (p0 <= cell.pos && cell.pos < p1) {
                tail_id = -1;
            }
        }
    } else {
        // seq_id is negative, then the range should include everything or nothing
        if (p0 != p1 && (p0 != 0 || p1 != std::numeric_limits<llama_pos>::max())) {
            //printf("[DEBUG] inside `llama_memory_recurrent::seq_rm`: `seq_id` is negative, so returning false\n");
            return false;
        }
    }

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].pos >= p0 && cells[i].pos < p1) {
            if (seq_id < 0) {
                cells[i].seq_id.clear();
            } else if (cells[i].has_seq_id(seq_id)) {
                cells[i].seq_id.erase(seq_id);
            } else {
                continue;
            }
            if (cells[i].is_empty()) {
                // keep count of the number of used cells
                if (cells[i].pos >= 0) {
                    used--;
                }
                cells[i].pos = -1;
                cells[i].src = -1;
                if (new_head == size) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != size && new_head < head) {
        head = new_head;
    }

    return true;
}

void llama_memory_recurrent::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    if (seq_id_src == seq_id_dst) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if ((uint32_t) seq_id_dst < size && (uint32_t) seq_id_src < size) {
        auto & tail_src = cells[seq_id_src];
        auto & tail_dst = cells[seq_id_dst];
        if (tail_dst.tail >= 0) {
            // clear destination seq_id if it wasn't empty
            auto & cell_dst = cells[tail_dst.tail];

            cell_dst.seq_id.erase(seq_id_dst);
            tail_dst.tail = -1;
            if (cell_dst.seq_id.empty()) {
                cell_dst.pos = -1;
                cell_dst.src = -1;
                used -= 1;
            }
        }
        if (tail_src.tail >= 0) {
            auto & cell_src = cells[tail_src.tail];

            cell_src.seq_id.insert(seq_id_dst);
            tail_dst.tail = tail_src.tail;
            if (compact_s) { rs_idx[seq_id_dst] = rs_idx[seq_id_src]; }
        }
    }
}

void llama_memory_recurrent::seq_keep(llama_seq_id seq_id) {
    uint32_t new_head = size;

    for (uint32_t i = 0; i < size; ++i) {
        if ((llama_seq_id) i != seq_id) {
            cells[i].tail = -1;
        }

        if (!cells[i].has_seq_id(seq_id)) {
            if (cells[i].pos >= 0) {
                used--;
            }

            cells[i].pos = -1;
            cells[i].src = -1;
            cells[i].seq_id.clear();

            if (new_head == size){
                new_head = i;
            }
        } else {
            cells[i].seq_id.clear();
            cells[i].seq_id.insert(seq_id);
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != size && new_head < head) {
        head = new_head;
    }
}

void llama_memory_recurrent::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    if (shift == 0) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the
    if (p0 == p1) {
        return;
    }

    // for Mamba-like or RWKV models, only the pos needs to be shifted
    if (0 <= seq_id && seq_id < (int64_t) size) {
        const int32_t tail_id = cells[seq_id].tail;
        if (tail_id >= 0) {
            auto & cell = cells[tail_id];
            if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                cell.pos += shift;
            }
        }
    }
}

void llama_memory_recurrent::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (d == 1) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    // for Mamba-like or RWKV models, only the pos needs to be changed
    if (0 <= seq_id && seq_id < (int64_t) size) {
        const int32_t tail_id = cells[seq_id].tail;
        if (tail_id >= 0) {
            auto & cell = cells[tail_id];
            if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                cell.pos /= d;
            }
        }
    }
}

llama_pos llama_memory_recurrent::seq_pos_min(llama_seq_id seq_id) const {
    llama_pos result = std::numeric_limits<llama_pos>::max();

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id)) {
            result = std::min(result, cells[i].pos);
        }
    }

    if (result == std::numeric_limits<llama_pos>::max()) {
        result = -1;
    }

    return result;
}

llama_pos llama_memory_recurrent::seq_pos_max(llama_seq_id seq_id) const {
    llama_pos result = -1;

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id)) {
            result = std::max(result, cells[i].pos);
        }
    }

    return result;
}

void llama_memory_recurrent::set_rs_idx(llama_seq_id seq_id, uint32_t idx) {
    if (seq_id < 0) {
        std::fill(rs_idx.begin(), rs_idx.end(), 0);
        return;
    }

    assert(n_seq_max == rs_idx.size());

    GGML_ASSERT((uint32_t) seq_id < n_seq_max);
    GGML_ASSERT(idx <= n_rs_seq);

    rs_idx[seq_id] = idx;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_recurrent::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [_, buf] : ctxs_bufs) {
        ret[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get());
    }
    return ret;
}

llama_memory_context_ptr llama_memory_recurrent::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // TODO: non-sequential equal split can be done if using unified KV cache
                //       for simplicity, we always use sequential equal split for now
                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                ubatch = balloc.split_equal(n_ubatch, true, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        if (!prepare(ubatches)) {
            break;
        }

        return std::make_unique<llama_memory_recurrent_context>(this, std::move(ubatches));
    } while (false);

    return std::make_unique<llama_memory_recurrent_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_recurrent::init_full() {
    return std::make_unique<llama_memory_recurrent_context>(this);
}

llama_memory_context_ptr llama_memory_recurrent::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(lctx);
    GGML_UNUSED(optimize);

    return std::make_unique<llama_memory_recurrent_context>(LLAMA_MEMORY_STATUS_NO_UPDATE);
}

bool llama_memory_recurrent::prepare(const std::vector<llama_ubatch> & ubatches) {
    // simply remember the full state because it is very small for this type of cache
    // TODO: optimize
    auto org_cells = cells;
    auto org_used = used;
    auto org_head = head;

    bool success = true;

    for (const auto & ubatch : ubatches) {
        if (!find_slot(ubatch)) {
            success = false;
            break;
        }
    }

    // restore the original state
    cells = std::move(org_cells);
    used = org_used;
    head = org_head;

    return success;
}

// process-wide reorder counters (llama_get_sparse_kv_stats)
std::atomic<uint64_t> g_rs_find_slot{0}, g_rs_moves{0}, g_rs_extra{0};

bool llama_memory_recurrent::find_slot(const llama_ubatch & ubatch) {
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;
    const uint32_t n_seqs       = ubatch.n_seqs;

    // if we have enough unused cells before the current head ->
    //   better to start searching from the beginning of the cache, hoping to fill it
    if (head > used + 2*n_seqs) {
        head = 0;
    }

    // For recurrent state architectures (like Mamba or RWKV),
    // each cache cell can store the state for a whole sequence.
    // A slot should be always be contiguous.

    // can only process batches with an equal number of new tokens in each sequence
    GGML_ASSERT(ubatch.equal_seqs());

    int32_t min = size - 1;
    int32_t max = 0;

    // everything should fit if all seq_ids are smaller than the max
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens; // first token of sequence set s
        const uint32_t n_seq_id = ubatch.n_seq_id[i];

        for (uint32_t j = 0; j < n_seq_id; ++j) {
            const llama_seq_id seq_id = ubatch.seq_id[i][j];

            if (seq_id < 0 || (uint32_t) seq_id >= size) {
                // too big seq_id
                // TODO: would it be possible to resize the cache instead?
                LLAMA_LOG_ERROR("%s: seq_id=%d >= n_seq_max=%u Try using a bigger --parallel value\n", __func__, seq_id, n_seq_max);
                return false;
            }
            if (j > 0) {
                auto & seq = cells[seq_id];
                if (seq.tail >= 0) {
                    auto & cell = cells[seq.tail];
                    // clear cells from seq_ids that become shared
                    // (should not normally happen, but let's handle it anyway)
                    cell.seq_id.erase(seq_id);
                    seq.tail = -1;
                    if (cell.seq_id.empty()) {
                        cell.pos = -1;
                        cell.src = -1;
                        used -= 1;
                    }
                }
            }
        }
    }

#ifndef NDEBUG
    {
        std::vector<int32_t> tails_verif;
        tails_verif.assign(size, -1);
        for (uint32_t i = 0; i < size; ++i) {
            auto & cell = cells[i];
            for (llama_seq_id seq_id : cell.seq_id) {
                if (tails_verif[seq_id] != -1) {
                    LLAMA_LOG_ERROR("%s: duplicate tail for seq_id %d in cell %d and %d\n", __func__, seq_id, i, tails_verif[seq_id]);
                }
                tails_verif[seq_id] = i;
            }
        }
        for (uint32_t i = 0; i < size; ++i) {
            if (tails_verif[i] != cells[i].tail) {
                LLAMA_LOG_ERROR("%s: wrong tail for seq_id %d, (%d instead of %d)\n", __func__, i, cells[i].tail, tails_verif[i]);
            }
        }
    }
#endif

    // find next empty cell
    uint32_t next_empty_cell = head;

    for (uint32_t i = 0; i < size; ++i) {
        if (next_empty_cell >= size) { next_empty_cell -= size; }
        auto & cell = cells[next_empty_cell];
        if (cell.is_empty()) { break; }
        next_empty_cell += 1;
    }

    // find usable cell range
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens;
        const llama_seq_id seq_id = ubatch.seq_id[i][0];
        auto & seq_meta = cells[seq_id];
        bool has_cell = false;
        if (seq_meta.tail >= 0) {
            auto & cell = cells[seq_meta.tail];
            GGML_ASSERT(cell.has_seq_id(seq_id));
            // does this seq_id "own" the cell?
            if (cell.seq_id.size() == 1) { has_cell = true; }
        }
        if (!has_cell && cell_hint && (size_t) seq_id < cell_hint->size() && (*cell_hint)[seq_id] < size &&
                (*cell_hint)[seq_id] != (uint32_t) next_empty_cell && cells[(*cell_hint)[seq_id]].is_empty()) {
            // take the cell of this sequence's stream; next_empty_cell stays free for the others
            auto & hinted = cells[(*cell_hint)[seq_id]];
            if (seq_meta.tail >= 0) {
                auto & orig_cell = cells[seq_meta.tail];
                hinted.pos = orig_cell.pos;
                hinted.src = orig_cell.src;
                hinted.journal_length = orig_cell.journal_length;
                orig_cell.seq_id.erase(seq_id);
                hinted.seq_id.insert(seq_id); // will be overwritten
                GGML_ASSERT(!orig_cell.is_empty()); // has at least one remaining seq_id
            }
            seq_meta.tail = (*cell_hint)[seq_id];
            has_cell = true;
        }
        if (!has_cell) {
            auto & empty_cell = cells[next_empty_cell];
            GGML_ASSERT(empty_cell.is_empty());
            // copy old tail into the empty cell
            if (seq_meta.tail >= 0) {
                auto & orig_cell = cells[seq_meta.tail];
                empty_cell.pos = orig_cell.pos;
                empty_cell.src = orig_cell.src;
                empty_cell.journal_length = orig_cell.journal_length;
                orig_cell.seq_id.erase(seq_id);
                empty_cell.seq_id.insert(seq_id); // will be overwritten
                GGML_ASSERT(!orig_cell.is_empty()); // has at least one remaining seq_id
            }
            seq_meta.tail = next_empty_cell;
            // find next empty cell
            if (s + 1 < n_seqs) {
                for (uint32_t j = 0; j < size; ++j) {
                    next_empty_cell += 1;
                    if (next_empty_cell >= size) { next_empty_cell -= size; }
                    auto & cell = cells[next_empty_cell];
                    if (cell.is_empty()) { break; }
                }
            }
        }
        if (min > seq_meta.tail) { min = seq_meta.tail; }
        if (max < seq_meta.tail) { max = seq_meta.tail; }
    }

    // gather and re-order
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens;
        const int32_t dst_id = s + min;
        const int32_t src_id = cells[ubatch.seq_id[i][0]].tail;
        if (dst_id != src_id) {
            g_rs_moves++;
            auto & dst_cell = cells[dst_id];
            auto & src_cell = cells[src_id];

            std::swap(dst_cell.pos, src_cell.pos);
            std::swap(dst_cell.src, src_cell.src);
            std::swap(dst_cell.journal_length, src_cell.journal_length);
            std::swap(dst_cell.seq_id, src_cell.seq_id);

            // swap tails
            for (uint32_t j = 0; j < size; ++j) {
                int32_t & tail = cells[j].tail;
                if (tail == src_id) {
                    tail = dst_id;
                } else if (tail == dst_id) {
                    tail = src_id;
                }
            }
        }
    }

    // Snapshot the previous verification span before recording the next batch.
    if (compact_s) {
        for (int i = min; i <= max; ++i) {
            cells[i].journal_input_length = cells[i].src < 0 ? 0 : cells[i].journal_length;
        }
    }

    // update the pos of the used seqs
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens;
        const llama_pos last_pos = ubatch.pos[i + n_seq_tokens - 1];
        const int32_t cell_id = s + min;
        auto & cell = cells[cell_id];

        if (cell.pos >= 0 && last_pos != cell.pos + (llama_pos) n_seq_tokens) {
            // What should happen when the pos backtracks or skips a value?
            // Clearing the state mid-batch would require special-casing which isn't done.
            LLAMA_LOG_WARN("%s: non-consecutive token position %d after %d for sequence %d with %u new tokens\n",
                __func__, last_pos, cell.pos, ubatch.seq_id[i][0], n_seq_tokens);
        }
        cell.pos = last_pos;
        if (compact_s) { cell.journal_length = std::min(n_seq_tokens, n_rs_seq + 1); }
        cell.seq_id.clear();
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            const llama_seq_id seq_id = ubatch.seq_id[i][j];
            cell.seq_id.insert(seq_id);
            cells[seq_id].tail = cell_id;
        }
    }

    // Find first cell without src refs, to use as the zero-ed state
    {
        // TODO: bake-in src refcounts in the cell metadata
        std::vector<int32_t> refcounts(size, 0);
        for (size_t i = 0; i < size; ++i) {
            const int32_t src = cells[i].src;
            if (src >= 0) {
                refcounts[src] += 1;
            }
        }

        rs_z = -1;
        for (int i = min; i <= max; ++i) {
            if (refcounts[i] == 0) {
                rs_z = i;
                break;
            }
        }

        for (int i = min; i <= max; ++i) {
            if (cells[i].src < 0) {
                GGML_ASSERT(rs_z >= 0);
                cells[i].src0 = rs_z;
            } else {
                // Stage the source ids for all used cells to allow correct seq_* behavior
                // and still make these values available when setting the inputs
                cells[i].src0 = cells[i].src;
            }
            cells[i].src = i; // avoid moving or clearing twice
        }
    }

    // allow getting the range of used cells, from head to head + n
    head = min;
    n    = max - min + 1;
    g_rs_find_slot++;
    g_rs_extra += n - n_seqs;
    used = std::count_if(cells.begin(), cells.end(),
        [](const mem_cell & cell){ return !cell.is_empty(); });

    // sanity check
    return n >= n_seqs;
}

bool llama_memory_recurrent::seq_move_cell(llama_seq_id seq_id, uint32_t dst, llama_seq_id may_displace) {
    if (seq_id < 0 || (uint32_t) seq_id >= size || dst >= size) {
        return false;
    }
    const int32_t src = cells[seq_id].tail;
    if (src == (int32_t) dst) {
        return true;
    }
    if (src < 0) {
        // nothing stored yet; find_slot() takes the hinted cell later
        return true;
    }
    auto & a = cells[src];
    auto & b = cells[dst];
    if (a.seq_id.size() != 1 || b.seq_id.size() > 1) {
        return false;
    }
    const llama_seq_id other = b.seq_id.empty() ? -1 : *b.seq_id.begin();
    if (other >= 0 && (other != may_displace || cells[other].tail != (int32_t) dst)) {
        return false;
    }
    // a pending move would place the physical rows elsewhere than the cell index
    if ((a.src >= 0 && a.src != src) || (b.src >= 0 && b.src != (int32_t) dst)) {
        return false;
    }

    // the live rows: plane rs_idx of each sequence (rollback snapshot pending) or plane 0
    const uint32_t pa = n_rs_seq ? rs_idx[seq_id] : 0;
    const uint32_t pb = n_rs_seq && other >= 0 ? rs_idx[other] : 0;

    auto row_view = [](ggml_tensor * t, uint32_t row) {
        ggml_tensor v = *t;
        v.ne[1] = 1;
        v.data = (char *) t->data + size_t{row} * t->nb[1];
        v.view_offs = t->view_offs + size_t{row} * t->nb[1];
        return v;
    };
    if (compact_s) {
        // A settled stream exchange must keep recurrent rows physically aligned.
        // The compact basis and journal are live data, never scratch space.
        const bool have_a = a.src >= 0;
        const bool have_b = other >= 0 && b.src >= 0;
        std::vector<ggml_tensor *> tensors;
        std::map<ggml_backend_buffer_type_t, size_t> scratch_sizes;
        for (uint32_t il = 0; il < r_l.size(); ++il) {
            for (ggml_tensor * tensor : { r_l[il], s_l[il],
                    il < p_l.size() ? p_l[il] : nullptr,
                    il < j_l.size() ? j_l[il] : nullptr }) {
                if (!tensor) { continue; }
                if (!ggml_is_contiguous(tensor) || tensor->ne[2] != 1 || tensor->ne[3] != 1 ||
                        tensor->ne[1] < size || tensor->ne[1] % size != 0) { return false; }
                tensors.push_back(tensor);
                if (have_a && have_b) {
                    const auto buft = ggml_backend_buffer_get_type(tensor->buffer);
                    scratch_sizes[buft] = std::max(scratch_sizes[buft], tensor->nb[1]);
                }
            }
        }
        // Allocate every scratch row before the first write, so refusal is atomic.
        std::map<ggml_backend_buffer_type_t, ggml_backend_buffer_ptr> scratch;
        for (const auto & entry : scratch_sizes) {
            ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(entry.first, entry.second));
            if (!buffer) { return false; }
            scratch.emplace(entry.first, std::move(buffer));
        }
        for (auto * tensor : tensors) {
            const uint32_t planes = tensor->ne[1] / size;
            for (uint32_t plane = 0; plane < planes; ++plane) {
                auto from_a = row_view(tensor, plane * size + src);
                auto from_b = row_view(tensor, plane * size + dst);
                if (have_a && have_b) {
                    const auto buft = ggml_backend_buffer_get_type(tensor->buffer);
                    ggml_tensor temporary = from_a;
                    temporary.buffer = scratch.at(buft).get();
                    temporary.data = ggml_backend_buffer_get_base(temporary.buffer);
                    temporary.view_src = nullptr;
                    temporary.view_offs = 0;
                    ggml_backend_tensor_copy(&from_a, &temporary);
                    ggml_backend_tensor_copy(&from_b, &from_a);
                    ggml_backend_tensor_copy(&temporary, &from_b);
                } else if (have_a) {
                    ggml_backend_tensor_copy(&from_a, &from_b);
                } else if (have_b) {
                    ggml_backend_tensor_copy(&from_b, &from_a);
                }
            }
        }
    } else if (n_rs_seq > 0) {
        // Different live planes cannot overlap. Equal planes use one stale snapshot row as scratch.
        const uint32_t scratch_plane = pa == 0 ? 1 : 0;
        for (uint32_t il = 0; il < r_l.size(); ++il) {
            for (ggml_tensor * t : { r_l[il], s_l[il], il < p_l.size() ? p_l[il] : nullptr }) {
                if (t == nullptr) { continue; }
                ggml_tensor from_a = row_view(t, pa*size + src), to_a = row_view(t, pa*size + dst);
                ggml_tensor from_b = row_view(t, pb*size + dst), to_b = row_view(t, pb*size + src);
                if (pa != pb) {
                    if (a.src >= 0) { ggml_backend_tensor_copy(&from_a, &to_a); }
                    if (other >= 0 && b.src >= 0) { ggml_backend_tensor_copy(&from_b, &to_b); }
                } else {
                    ggml_tensor scratch = row_view(t, scratch_plane*size + src);
                    if (a.src >= 0) { ggml_backend_tensor_copy(&from_a, &scratch); }
                    if (other >= 0 && b.src >= 0) { ggml_backend_tensor_copy(&from_b, &to_b); }
                    if (a.src >= 0) { ggml_backend_tensor_copy(&scratch, &to_a); }
                }
            }
        }
    } else {
        // Without snapshot planes, preserve both rows through host staging.
        struct staged_t { ggml_tensor * t; size_t offs; std::vector<uint8_t> data; };
        std::vector<staged_t> staged;
        auto stage = [&](ggml_tensor * t, uint32_t row_from, uint32_t row_to) {
            if (t == nullptr) {
                return;
            }
            staged.push_back({ t, size_t{row_to} * t->nb[1], std::vector<uint8_t>(t->nb[1]) });
            ggml_backend_tensor_get(t, staged.back().data.data(), size_t{row_from} * t->nb[1], t->nb[1]);
        };
        for (uint32_t il = 0; il < r_l.size(); ++il) {
            for (ggml_tensor * t : { r_l[il], s_l[il], il < p_l.size() ? p_l[il] : nullptr }) {
                if (a.src >= 0) {
                    stage(t, pa*size + src, pa*size + dst);
                }
                if (other >= 0 && b.src >= 0) {
                    stage(t, pb*size + dst, pb*size + src);
                }
            }
        }
        for (const auto & st : staged) {
            ggml_backend_tensor_set(st.t, st.data.data(), st.offs, st.data.size());
        }
    }

    const bool a_state = a.src >= 0, b_state = b.src >= 0 && (!compact_s || other >= 0);
    if (compact_s) {
        std::swap(a.journal_length, b.journal_length);
        std::swap(a.journal_input_length, b.journal_input_length);
    }
    std::swap(a.pos, b.pos);
    std::swap(a.seq_id, b.seq_id);
    b.src = a_state ? (int32_t) dst : -1;
    a.src = b_state ? src : -1;
    if (compact_s) { a.src0 = a.src; b.src0 = b.src; }
    cells[seq_id].tail = dst;
    if (other >= 0) {
        cells[other].tail = src;
    }
    return true;
}

bool llama_memory_recurrent::get_can_shift() const {
    // shifting the pos is trivial for recurrent models
    return true;
}

bool llama_memory_recurrent::is_empty() const {
    const bool res = ctxs_bufs.empty();
    assert(!res || total_size() == 0);
    return res;
}

size_t llama_memory_recurrent::total_size() const {
    size_t size = 0;
    for (const auto & [_, buf] : ctxs_bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

size_t llama_memory_recurrent::size_r_bytes() const {
    size_t size_r_bytes = 0;

    for (const auto & r : r_l) {
        if (r != nullptr) {
            size_r_bytes += ggml_nbytes(r);
        }
    }

    return size_r_bytes;
}

size_t llama_memory_recurrent::size_s_bytes() const {
    size_t size_s_bytes = 0;

    for (const auto & s : s_l) {
        if (s != nullptr) {
            size_s_bytes += ggml_nbytes(s);
        }
    }

    return size_s_bytes;
}

size_t llama_memory_recurrent::size_p_bytes() const {
    size_t size_p_bytes = 0;

    for (const auto & p : p_l) {
        if (p != nullptr) {
            size_p_bytes += ggml_nbytes(p);
        }
    }

    return size_p_bytes;
}

void llama_memory_recurrent::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    GGML_UNUSED(flags);

    std::vector<std::pair<uint32_t, uint32_t>> cell_ranges; // ranges, from inclusive, to exclusive
    std::vector<std::pair<uint32_t, uint32_t>> cell_ranges_data; // logical source row ranges
    uint32_t cell_count = 0;
    std::vector<int32_t> journal_lengths;

    // Count the number of cells with the specified seq_id
    // Find all the ranges of cells with this seq id (or all, when -1)
    uint32_t cell_range_begin = size;
    for (uint32_t i = 0; i < size; ++i) {
        const auto & cell = cells[i];
        // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_LLAMA_SEQ_ID_NEG]
        if ((seq_id == -1 && !cell.is_empty()) || cell.has_seq_id(seq_id)) {
            ++cell_count;
            uint32_t rs_idx_cur = 0;

            if (n_rs_seq != 0) {
                if (seq_id != -1) {
                    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < rs_idx.size());
                    rs_idx_cur = rs_idx[seq_id];
                } else {
                    bool has_rs_idx = false;
                    for (const llama_seq_id cell_seq_id : cell.seq_id) {
                        GGML_ASSERT(cell_seq_id >= 0 && (size_t) cell_seq_id < rs_idx.size());

                        const uint32_t seq_rs_idx = rs_idx[cell_seq_id];
                        if (!has_rs_idx) {
                            rs_idx_cur = seq_rs_idx;
                            has_rs_idx = true;
                        } else if (rs_idx_cur != seq_rs_idx) {
                            GGML_ABORT("cannot write shared recurrent state with different rollback indices");
                        }
                    }
                }
            }

            journal_lengths.push_back(cell.journal_length);
            const uint32_t cell_id = rs_idx_cur * size + (cell.src >= 0 ? cell.src : (int32_t) i);
            if (cell_ranges_data.empty() || cell_ranges_data.back().second != cell_id) {
                cell_ranges_data.emplace_back(cell_id, cell_id + 1);
            } else {
                cell_ranges_data.back().second++;
            }

            if (cell_range_begin == size) {
                cell_range_begin = i;
            }
        } else {
            if (cell_range_begin != size) {
                cell_ranges.emplace_back(cell_range_begin, i);
                cell_range_begin = size;
            }
        }
    }
    if (cell_range_begin != size) {
        cell_ranges.emplace_back(cell_range_begin, size);
    }

    if ((flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) && cell_ranges.size() > 1) {
        GGML_ABORT("cannot save/load multiple ranges of cells to/from device memory\n");
    }

    // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
    uint32_t cell_count_check = 0;
    for (const auto & range : cell_ranges) {
        cell_count_check += range.second - range.first;
    }
    GGML_ASSERT(cell_count == cell_count_check);

    cell_count_check = 0;
    for (const auto & range : cell_ranges_data) {
        cell_count_check += range.second - range.first;
    }
    GGML_ASSERT(cell_count == cell_count_check);

    io.write(&cell_count, sizeof(cell_count));

    state_write_meta(io, cell_ranges, seq_id);
    state_write_data(io, cell_ranges_data, journal_lengths);
}

void llama_memory_recurrent::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(flags);

    uint32_t cell_count;
    io.read(&cell_count, sizeof(cell_count));

    bool res = true;

    // save the head of the restored cells - could be needed to clear the state
    // the head is valid only when state_read_meta() succeeded
    const bool meta_read = state_read_meta(io, cell_count, seq_id);
    const uint32_t cell_head = head;

    res = res && meta_read;

    try {
        res = res && state_read_data(io, cell_count);
    } catch (...) {
        res = false;
    }

    if (!res) {
        state_clear(seq_id, cell_head, meta_read ? cell_count : 0);
        throw std::runtime_error("failed to restore kv cache");
    }

    if (n_rs_seq != 0) {
        set_rs_idx(seq_id, 0);
    }
}

void llama_memory_recurrent::state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id) const {
    for (const auto & range : cell_ranges) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            const auto & cell = cells[i];
            const llama_pos pos      = cell.pos;
            const uint32_t  n_seq_id = seq_id == -1 ? cell.seq_id.size() : 0;

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id) {
                for (auto seq_id : cell.seq_id) {
                    io.write(&seq_id, sizeof(seq_id));
                }
            }
        }
    }
}

namespace {
struct deferred_rs_export {
    ggml_backend_ptr backend;
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * output = nullptr;
};

static std::shared_ptr<deferred_rs_export> deferred_rs_export_make(
        ggml_tensor * states, ggml_tensor * journal, const llama_hparams & hp,
        uint32_t cells, uint32_t capacity, const std::vector<int32_t> & indices,
        const std::vector<int32_t> & lengths) {
    GGML_ASSERT(indices.size() == lengths.size() && !indices.empty());
    auto job = std::make_shared<deferred_rs_export>();
    auto dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(states->buffer));
    job->backend.reset(ggml_backend_dev_init(dev, nullptr));
    if (!job->backend) { throw std::runtime_error("deferred export backend allocation failed"); }
    ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(16, false), nullptr, true };
    job->ctx.reset(ggml_init(ip));
    if (!job->ctx) { throw std::runtime_error("deferred export context allocation failed"); }
    auto ctx = job->ctx.get();
    auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, indices.size());
    auto * lens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, lengths.size());
    auto * output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, states->ne[0], indices.size());
    output->op = GGML_OP_GATED_DELTA_NET;
    output->src[0] = states;
    output->src[1] = journal;
    output->src[2] = ids;
    output->src[3] = lens;
    int32_t params[7] = {1, 0x52534a31, (int32_t) hp.ssm_d_state, (int32_t) hp.ssm_n_group,
            (int32_t) hp.ssm_dt_rank, (int32_t) capacity, (int32_t) cells};
    memcpy(output->op_params, params, sizeof(params));
    ggml_set_input(ids); ggml_set_input(lens); ggml_set_output(output);
    if (!ggml_backend_supports_op(job->backend.get(), output)) {
        throw std::runtime_error("deferred export operator unsupported");
    }
    auto graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, output);
    job->buffer.reset(ggml_backend_alloc_ctx_tensors(ctx, job->backend.get()));
    if (!job->buffer) { throw std::runtime_error("deferred export buffer allocation failed"); }
    ggml_backend_tensor_set(ids, indices.data(), 0, indices.size() * sizeof(int32_t));
    ggml_backend_tensor_set(lens, lengths.data(), 0, lengths.size() * sizeof(int32_t));
    if (ggml_backend_graph_compute(job->backend.get(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("deferred export replay failed");
    }
    ggml_backend_synchronize(job->backend.get());
    job->output = output;
    return job;
}
} // namespace

void llama_memory_recurrent::state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, const std::vector<int32_t> & journal_lengths) const {
    const uint32_t s_trans = 0;
    const uint32_t n_layer = hparams.n_layer();

    io.write(&s_trans, sizeof(s_trans));
    io.write(&n_layer, sizeof(n_layer));

    // Iterate and write all the R tensors first, each row is a cell
    // Get whole range at a time
    for (uint32_t il = 0; il < n_layer; ++il) {
        // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
        if (r_l[il] == nullptr) continue;

        // Write R tensor type
        const int32_t r_type_i = (int32_t)r_l[il]->type;
        io.write(&r_type_i, sizeof(r_type_i));

        // Write row size of R tensor
        const uint64_t r_size_row = ggml_row_size(r_l[il]->type, hparams.n_embd_r());
        io.write(&r_size_row, sizeof(r_size_row));

        // Write each logical cell row range. With pending recurrent rollback,
        // the logical current state may live in a rollback snapshot plane.
        for (const auto & range : cell_ranges) {
            const size_t range_size = range.second - range.first;
            const size_t buf_size = range_size * r_size_row;
            io.write_tensor(r_l[il], range.first * r_size_row, buf_size);
        }

        // the PLE conv history is a second recurrent row, so it has to travel with the first
        if (p_l[il] != nullptr) {
            const uint64_t p_size_row = ggml_row_size(p_l[il]->type, hparams.ple_conv_state());
            io.write(&p_size_row, sizeof(p_size_row));

            for (const auto & range : cell_ranges) {
                const size_t range_size = range.second - range.first;
                io.write_tensor(p_l[il], range.first * p_size_row, range_size * p_size_row);
            }
        }
    }

    if (!s_trans) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
            if (s_l[il] == nullptr) continue;

            // Write S tensor type
            const int32_t s_type_i = (int32_t)s_l[il]->type;
            io.write(&s_type_i, sizeof(s_type_i));

            // Write row size of S tensor
            const uint64_t s_size_row = ggml_row_size(s_l[il]->type, hparams.n_embd_s());
            io.write(&s_size_row, sizeof(s_size_row));

            ggml_tensor * export_tensor = nullptr;
            if (compact_s && io.requires_tensor_data()) {
                std::vector<int32_t> indices;
                bool pending = false;
                for (const auto & range : cell_ranges) {
                    for (uint32_t row = range.first; row < range.second; ++row) {
                        indices.push_back(row);
                        pending = pending || row >= size;
                    }
                }
                if (pending) {
                    auto job = deferred_rs_export_make(s_l[il], j_l[il], hparams, size,
                            n_rs_seq + 1, indices, journal_lengths);
                    export_tensor = job->output;
                    io.retain_tensor_storage(job);
                }
            }
            size_t export_offset = 0;
            // Write each logical cell row range. With pending recurrent rollback,
            // the logical current state may live in a rollback snapshot plane.
            for (const auto & range : cell_ranges) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * s_size_row;
                if (export_tensor) {
                    io.write_tensor(export_tensor, export_offset, buf_size);
                    export_offset += buf_size;
                } else {
                    // Size-only writers do not dereference the logical rollback row.
                    io.write_tensor(s_l[il], range.first * s_size_row, buf_size);
                }
            }
        }
    } else {
        // When S tensor is transposed, we also need the element size and get the element ranges from each row
        const uint32_t mem_size = size;
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
            if (s_l[il] == nullptr) continue;

            const uint32_t n_embd_s = hparams.n_embd_s();

            // Write S tensor type
            const int32_t s_type_i = (int32_t)s_l[il]->type;
            io.write(&s_type_i, sizeof(s_type_i));

            // Write element size
            const uint32_t s_size_el = ggml_type_size(s_l[il]->type);
            io.write(&s_size_el, sizeof(s_size_el));

            // Write GQA embedding size
            io.write(&n_embd_s, sizeof(n_embd_s));

            // For each row, we get the element values of each logical cell
            for (uint32_t j = 0; j < n_embd_s; ++j) {
                for (const auto & range : cell_ranges) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + j * mem_size) * s_size_el;
                    const size_t buf_size = range_size * s_size_el;
                    io.write_tensor(s_l[il], src_offset, buf_size);
                }
            }
        }
    }
}

bool llama_memory_recurrent::state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id) {
    if (dest_seq_id != -1) {
        // single sequence
        if (cell_count > size) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        seq_rm(dest_seq_id, -1, -1);

        if (cell_count == 0) {
            return true;
        }

        llama_batch_allocr balloc(hparams.n_pos_per_embd());

        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 0) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            ubatch.pos[i] = pos;
        }
        ubatch.n_seq_id[0] = 1;
        ubatch.seq_id[0] = &dest_seq_id;

        if (!find_slot(ubatch)) {
            LLAMA_LOG_ERROR("%s: failed to find available cells in kv cache\n", __func__);
            return false;
        }

        // DEBUG CHECK: kv.head should be our first cell, kv.head + cell_count - 1 should be our last cell (verify seq_id and pos values)
        // Assume that this is one contiguous block of cells
        GGML_ASSERT(head + cell_count <= size);
        GGML_ASSERT(cells[head].pos == ubatch.pos[0]);
        GGML_ASSERT(cells[head + cell_count - 1].pos == ubatch.pos[cell_count - 1]);
        GGML_ASSERT(cells[head].has_seq_id(dest_seq_id));
        GGML_ASSERT(cells[head + cell_count - 1].has_seq_id(dest_seq_id));
    } else {
        // whole KV cache restore

        if (cell_count > size) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        clear(true);

        for (uint32_t i = 0; i < cell_count; ++i) {
            auto & cell = cells[i];

            llama_pos pos;
            uint32_t  n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            cell.pos = pos;

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= this->n_seq_max) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, this->n_seq_max);
                    return false;
                }

                cell.seq_id.insert(seq_id);

                int32_t & tail = cells[seq_id].tail;
                if (tail != -1) {
                    LLAMA_LOG_ERROR("%s: duplicate tail for seq_id %d in cell %d and %d\n", __func__, seq_id, i, tail);
                    return false;
                }
                tail = i;
            }
        }

        head = 0;
        used = cell_count;
    }

    for (uint32_t i = 0; i < cell_count; ++i) {
        uint32_t cell_id = head + i;
        // make sure the recurrent states will keep their restored state
        cells[cell_id].src = cell_id;
        cells[cell_id].journal_length = cells[cell_id].journal_input_length = 0;
    }

    return true;
}

bool llama_memory_recurrent::state_read_data(llama_io_read_i & io, uint32_t cell_count) {
    uint32_t s_trans;
    uint32_t n_layer;
    io.read(&s_trans, sizeof(s_trans));
    io.read(&n_layer, sizeof(n_layer));

    if (n_layer != hparams.n_layer()) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, hparams.n_layer());
        return false;
    }
    if (cell_count > size) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, size);
        return false;
    }
    if (false != (bool) s_trans) {
        LLAMA_LOG_ERROR("%s: incompatible s transposition\n", __func__);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (uint32_t il = 0; il < n_layer; ++il) {
        // skip null layers
        if (r_l[il] == nullptr) continue;

        // Read type of key
        int32_t r_type_i_ref;
        io.read(&r_type_i_ref, sizeof(r_type_i_ref));
        const int32_t r_type_i = (int32_t) r_l[il]->type;
        if (r_type_i != r_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched r type (%d != %d, layer %d)\n", __func__, r_type_i, r_type_i_ref, il);
            return false;
        }

        // Read row size of key
        uint64_t r_size_row_ref;
        io.read(&r_size_row_ref, sizeof(r_size_row_ref));
        const size_t r_size_row = ggml_row_size(r_l[il]->type, hparams.n_embd_r());
        if (r_size_row != r_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched r row size (%zu != %zu, layer %d)\n", __func__, r_size_row, (size_t) r_size_row_ref, il);
            return false;
        }

        if (cell_count) {
            // Read and set the keys for the whole cell range
            io.read_tensor(r_l[il], head * r_size_row, cell_count * r_size_row);
        }

        if (p_l[il] != nullptr) {
            uint64_t p_size_row_ref;
            io.read(&p_size_row_ref, sizeof(p_size_row_ref));
            const size_t p_size_row = ggml_row_size(p_l[il]->type, hparams.ple_conv_state());
            if (p_size_row != p_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched ple row size (%zu != %zu, layer %d)\n", __func__, p_size_row, (size_t) p_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                io.read_tensor(p_l[il], head * p_size_row, cell_count * p_size_row);
            }
        }
    }

    if (!s_trans) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers
            if (s_l[il] == nullptr) continue;

            // Read type of value
            int32_t s_type_i_ref;
            io.read(&s_type_i_ref, sizeof(s_type_i_ref));
            const int32_t s_type_i = (int32_t)s_l[il]->type;

            if (s_type_i != s_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s type (%d != %d, layer %d)\n", __func__, s_type_i, s_type_i_ref, il);
                return false;
            }

            // Read row size of value
            uint64_t s_size_row_ref;
            io.read(&s_size_row_ref, sizeof(s_size_row_ref));
            const size_t s_size_row = ggml_row_size(s_l[il]->type, hparams.n_embd_s());
            if (s_size_row != s_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s row size (%zu != %zu, layer %d)\n", __func__, s_size_row, (size_t) s_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                // Read and set the values for the whole cell range
                io.read_tensor(s_l[il], head * s_size_row, cell_count * s_size_row);
            }
        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers
            if (s_l[il] == nullptr) continue;

            const uint32_t n_embd_s = hparams.n_embd_s();

            // Read type of value
            int32_t s_type_i_ref;
            io.read(&s_type_i_ref, sizeof(s_type_i_ref));
            const int32_t s_type_i = (int32_t)s_l[il]->type;
            if (s_type_i != s_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s type (%d != %d, layer %d)\n", __func__, s_type_i, s_type_i_ref, il);
                return false;
            }

            // Read element size of value
            uint32_t s_size_el_ref;
            io.read(&s_size_el_ref, sizeof(s_size_el_ref));
            const size_t s_size_el = ggml_type_size(s_l[il]->type);
            if (s_size_el != s_size_el_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s element size (%zu != %zu, layer %d)\n", __func__, s_size_el, (size_t) s_size_el_ref, il);
                return false;
            }

            // Read state embedding size
            uint32_t n_embd_s_ref;
            io.read(&n_embd_s_ref, sizeof(n_embd_s_ref));
            if (n_embd_s != n_embd_s_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s embedding size (%u != %u, layer %d)\n", __func__, n_embd_s, n_embd_s_ref, il);
                return false;
            }

            if (cell_count) {
                // For each row in the transposed matrix, read the values for the whole cell range
                for (uint32_t j = 0; j < n_embd_s; ++j) {
                    const size_t dst_offset = (head + j * size) * s_size_el;
                    io.read_tensor(s_l[il], dst_offset, cell_count * s_size_el);
                }
            }
        }
    }

    return true;
}

// the cleared ranges mirror the write pattern of state_read_data() - keep both in sync
// the transposed s layout is not handled - state_read_data() rejects it before any write
void llama_memory_recurrent::state_clear(llama_seq_id seq_id, uint32_t cell_head, uint32_t cell_count) {
    // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_LLAMA_SEQ_ID_NEG]
    if (seq_id == -1) {
        clear(true);
        return;
    }

    seq_rm(seq_id, -1, -1);

    if (cell_count == 0) {
        return;
    }

    const uint32_t n_layer = hparams.n_layer();

    for (uint32_t il = 0; il < n_layer; ++il) {
        if (r_l[il] != nullptr) {
            const size_t r_size_row = ggml_row_size(r_l[il]->type, hparams.n_embd_r());
            llama_clear_tensor_data(r_l[il], cell_head * r_size_row, cell_count * r_size_row);
        }

        if (s_l[il] != nullptr) {
            const size_t s_size_row = ggml_row_size(s_l[il]->type, hparams.n_embd_s());
            llama_clear_tensor_data(s_l[il], cell_head * s_size_row, cell_count * s_size_row);
        }

        if (p_l[il] != nullptr) {
            const size_t p_size_row = ggml_row_size(p_l[il]->type, hparams.ple_conv_state());
            llama_clear_tensor_data(p_l[il], cell_head * p_size_row, cell_count * p_size_row);
        }
    }
}

//
// llama_memory_recurrent_context
//

llama_memory_recurrent_context::llama_memory_recurrent_context(llama_memory_status status) : status(status) {}

llama_memory_recurrent_context::llama_memory_recurrent_context(
        llama_memory_recurrent * mem) : status(LLAMA_MEMORY_STATUS_SUCCESS), mem(mem), is_full(true) {
}

llama_memory_recurrent_context::llama_memory_recurrent_context(
        llama_memory_recurrent * mem,
        std::vector<llama_ubatch> ubatches) : status(LLAMA_MEMORY_STATUS_SUCCESS), mem(mem), ubatches(std::move(ubatches)) {}

llama_memory_recurrent_context::~llama_memory_recurrent_context() = default;

bool llama_memory_recurrent_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_recurrent_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    // no ubatches -> this is an update
    if (ubatches.empty()) {
        // recurrent cache never performs updates
        assert(status == LLAMA_MEMORY_STATUS_NO_UPDATE);

        return true;
    }

    mem->find_slot(ubatches[i_next]);

    return true;
}

llama_memory_status llama_memory_recurrent_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_recurrent_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

uint32_t llama_memory_recurrent_context::get_n_rs() const {
    return is_full ? (llama_recurrent_reserve_sequences ? std::min(mem->size, llama_recurrent_reserve_sequences) : mem->size) : mem->n;
}

uint32_t llama_memory_recurrent_context::get_head() const {
    return is_full ? 0 : mem->head;
}

int32_t llama_memory_recurrent_context::get_rs_z() const {
    return is_full ? 0 : mem->rs_z;
}

uint32_t llama_memory_recurrent_context::get_size() const {
    return mem->size;
}

ggml_tensor * llama_memory_recurrent_context::get_r_l(int32_t il) const {
    return mem->r_l[il];
}

ggml_tensor * llama_memory_recurrent_context::get_s_l(int32_t il) const {
    return mem->s_l[il];
}

ggml_tensor * llama_memory_recurrent_context::get_p_l(int32_t il) const {
    return mem->p_l[il];
}

int32_t llama_memory_recurrent_context::s_copy(int i) const {
    const uint32_t cell_idx = i + mem->head;
    const int32_t  src0     = mem->cells[cell_idx].src0;

    if (mem->n_rs_seq == 0) {
        return src0;
    }

    if (mem->compact_s) {
        auto & cell = mem->cells[cell_idx];
        uint32_t idx = 0;
        bool first = true;
        for (const auto seq : cell.seq_id) {
            if (first) { idx = mem->rs_idx[seq]; first = false; }
            else { GGML_ASSERT(idx == mem->rs_idx[seq]); }
        }
        GGML_ASSERT(idx <= cell.journal_input_length);
        for (const auto seq : cell.seq_id) { mem->rs_idx[seq] = 0; }
        // Compaction consumes rollback; no new decode means no new rollback history.
        const uint32_t active = is_full ? mem->size : ubatches[i_next].n_seqs;
        if ((uint32_t) i >= active && idx) { cell.journal_length = 0; }
        return (int32_t) (idx * mem->size) + src0;
    }

    uint32_t idx = 0;
    if (!mem->cells[cell_idx].seq_id.empty()) {
        const llama_seq_id seq = *mem->cells[cell_idx].seq_id.begin();
        if (seq >= 0 && (size_t) seq < mem->rs_idx.size()) {
            idx = mem->rs_idx[seq];
            // reset rollback idx
            mem->rs_idx[seq] = 0;
        }
    }
    return (int32_t)(idx * mem->size) + src0;
}


ggml_tensor * llama_memory_recurrent_context::get_j_l(int32_t il) const {
    return mem->compact_s ? mem->j_l[il] : nullptr;
}

ggml_tensor * llama_memory_recurrent_context::journal_for(const ggml_tensor * s) const {
    if (!mem->compact_s) { return nullptr; }
    for (size_t i = 0; i < mem->s_l.size(); ++i) {
        if (mem->s_l[i] == s) { return mem->j_l[i]; }
    }
    return nullptr;
}

bool llama_memory_recurrent_context::compact_s_enabled() const { return mem->compact_s; }

uint32_t llama_memory_recurrent_context::journal_input_length(int i) const {
    return mem->cells[i + mem->head].journal_input_length;
}
