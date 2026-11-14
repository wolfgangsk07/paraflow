#include "server-context.h"
#include "server-loop-guard.h"
#include "server-checkpoint-tail.h"
#include "server-request-policy.h"
#include "pipeline-lane-plan.h"
#include "src/llama-ext.h"

// libllama exports (C++ linkage, defined in src/llama-context.cpp)
LLAMA_API int32_t llama_state_seq_prepare_capture(llama_context * ctx, llama_seq_id seq_id);
LLAMA_API size_t  llama_state_seq_get_data_ext_prepared(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags);
LLAMA_API void    llama_state_seq_capture_counts(const llama_context * ctx, uint64_t * fast, uint64_t * full, uint64_t * verified, uint64_t * mismatches);
LLAMA_API void    llama_state_seq_capture_record_verify(const llama_context * ctx, bool match);
LLAMA_API int64_t llama_state_seq_get_data_async(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags);
LLAMA_API int32_t llama_state_seq_async_wait(llama_context * ctx, int64_t handle, bool block);
LLAMA_API void    llama_state_seq_async_counts(const llama_context * ctx, uint64_t * enqueued, uint64_t * sync, uint64_t * unmapped, uint64_t * outstanding);
#include <functional>
#include "server-chat.h"
#include "server-common.h"
#include "server-decision.h"
#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"
#include "server-schema.h"
#include "server-stream.h"

#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <cinttypes>
#include <ctime>
#include <unistd.h>
#include <exception>
#include <future>
#include <thread>
#include <condition_variable>
#include <deque>
#include <atomic>
#include <tuple>
#include <mutex>
#include <chrono>
#include <memory>
#include <filesystem>
#include <random>
#include <utility>
#include <fstream>
#include <sstream>
#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

// Wall counters remain active in release builds (scoped_timer is DEBUG_TIMINGS-only).
struct server_wall_timer {
    int64_t & total, & calls;
    int64_t start;
    server_wall_timer(int64_t & t, int64_t & c) : total(t), calls(c), start(ggml_time_us()) { ++calls; }
    ~server_wall_timer() { total += ggml_time_us() - start; }
};

constexpr int HTTP_POLLING_SECONDS = 1;

static bool server_has_system_message(const json & body) {
    return body.contains("messages") && body.at("messages").is_array() && !body.at("messages").empty() &&
            body.at("messages").front().is_object() &&
            json_value(body.at("messages").front(), "role", std::string()) == "system";
}

static common_speculative_output_limits server_output_limits(const common_params & params) {
    if (params.embedding ||
            (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED && params.pooling_type != LLAMA_POOLING_TYPE_NONE)) {
        return { params.n_batch, 1 };
    }

    auto result = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));

    result.total   = std::max<int32_t>(1, result.total);
    result.per_seq = std::max<int32_t>(1, result.per_seq);
    return result;
}

// a checkpoint restore dropped tokens the target had accepted - re-accept them rather than verify again
static std::vector<llama_token> server_accept_replay(
        common_sampler * smpl,
        llama_context * ctx,
        const std::vector<int32_t> & idxs,
        const llama_tokens & draft) {
    GGML_ASSERT(idxs.size() == draft.size() + 1);

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    for (size_t i = 0; i < draft.size(); ++i) {
        // the token is discarded - the call is what advances the sampler over this position
        common_sampler_sample(smpl, ctx, idxs[i]);
        common_sampler_accept(smpl, draft[i], true);
        result.push_back(draft[i]);
    }

    const llama_token id = common_sampler_sample(smpl, ctx, idxs[draft.size()]);
    common_sampler_accept(smpl, id, true);
    result.push_back(id);

    return result;
}

// synthetic draft verification for benchmarking - accept draft tokens at random instead of by match with the target
// on replay the draft was already accepted before a context checkpoint restore, so repeat the same decisions
static std::vector<llama_token> server_sample_and_accept_synth(
        common_sampler * smpl,
        llama_context * ctx,
        const std::vector<int32_t> & idxs,
        const llama_tokens & draft,
        const std::vector<double> & synth_probs,
        std::mt19937 & rng,
        bool is_replay) {
    GGML_ASSERT(idxs.size() == draft.size() + 1);
    GGML_ASSERT(synth_probs.size() >= draft.size());

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < draft.size(); ++i) {
        const llama_token id = common_sampler_sample(smpl, ctx, idxs[i]);
        const bool accept = is_replay || dist(rng) < synth_probs[i];
        // do not accept a drafted EOG token - it would end the generation early
        // on replay the last token is from the target and can be EOG, so skip this check
        if (accept && (is_replay || !llama_vocab_is_eog(vocab, draft[i]))) {
            // synthetic draft tokens do not advance grammar or reasoning state
            // the last replay token is from the target and must advance both
            const bool is_replay_target = is_replay && i + 1 == draft.size();
            common_sampler_accept(smpl, draft[i], is_replay_target);
            result.push_back(draft[i]);
            continue;
        }

        common_sampler_accept(smpl, id, true);
        result.push_back(id);
        return result;
    }

    const llama_token id = common_sampler_sample(smpl, ctx, idxs[draft.size()]);
    common_sampler_accept(smpl, id, true);
    result.push_back(id);

    return result;
}

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER, // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,    // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

struct server_slot; // forward declaration

struct server_batch {
    common_batch view; // the rendered sub-batch [off, off + n_tokens), see render()

    struct token {
        int32_t id_slot;
        llama_token token;
        std::array<llama_pos, GGML_MROPE_SECTIONS> pos; // only pos[0] is used for text tokens
        bool output;
        bool is_prompt; // for stats tracking
        int32_t decision_order = 0;
        int32_t i_embd = -1; // row in embd, -1 if this is a token
    };
    std::vector<token> tokens;
    int32_t n_tokens_alloc = 0;
    int32_t n_embd = 0;

    // track if given slot can be batched with slots already in the batch
    server_slot * slot_batched = nullptr;

    // embedding entries, they can be mixed with tokens if the context supports it
    std::vector<float> embd; // n_embd per row

    float  alora_scale       = -1.0f;
    size_t alora_disabled_id = 0;

    void init(llama_context * ctx, int32_t n_tokens_alloc, int32_t n_embd) {
        this->n_tokens_alloc = n_tokens_alloc;
        this->n_embd = n_embd;
        view = common_batch(ctx);
        tokens.reserve(n_tokens_alloc);
    }

    bool add(int32_t id_slot, llama_token token, llama_pos pos, bool output, bool is_prompt) {
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, token, { pos, 0, 0, 0 }, output, is_prompt });
        return true;
    }

    // embd_in has n_embd values, pos has GGML_MROPE_SECTIONS values
    bool add_embd(int32_t id_slot, const float * embd_in, const llama_pos * pos, bool output, bool is_prompt) {
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, LLAMA_TOKEN_NULL, { pos[0], pos[1], pos[2], pos[3] }, output, is_prompt });
        tokens.back().i_embd = (int32_t) (embd.size() / n_embd);
        embd.insert(embd.end(), embd_in, embd_in + n_embd);
        return true;
    }

    bool has_embd() const {
        return !embd.empty();
    }

    void clear() {
        tokens.clear();
        embd.clear();
        view.clear();
        slot_batched      = nullptr;
        alora_scale       = -1.0f;
        alora_disabled_id = 0;
    }

    int32_t size() const {
        return (int32_t)tokens.size();
    }

    // remove the entries after the first n
    void truncate(int32_t n) {
        GGML_ASSERT(n >= 0 && n <= size());
        for (int32_t i = n; i < size(); i++) {
            if (tokens[i].i_embd >= 0) {
                embd.resize((size_t) tokens[i].i_embd * n_embd);
                break;
            }
        }
        tokens.resize(n);
    }

    void set_output(int32_t idx, bool output) {
        GGML_ASSERT(idx >= 0 && idx < (int32_t)tokens.size());
        tokens[idx].output = output;
    }

    void set_decision_order(int32_t idx, int32_t order) {
        GGML_ASSERT(idx >= 0 && idx < (int32_t)tokens.size());
        tokens[idx].decision_order = order;
    }

    // render the sub-batch [off, off + n_tokens) into view, index i in view is index off + i here
    void render(int32_t off, int32_t n_tokens) {
        GGML_ASSERT(off >= 0 && off < size());
        GGML_ASSERT(n_tokens > 0 && off + n_tokens <= size());

        view.clear();
        for (int32_t i = off; i < off + n_tokens; i++) {
            const auto & t = tokens[i];
            if (t.i_embd >= 0) {
                view.add_embd({ embd.data() + (size_t) t.i_embd * n_embd, 1, (size_t) n_embd }, t.pos.data(), t.id_slot, t.output);
            } else {
                view.add(t.token, t.pos[0], t.id_slot, t.output);
            }
            view.tokens.back().decision_order = t.decision_order;
        }
    }
};

// Test-only hooks (LLAMA_DIAGNOSTICS builds), e.g. the N-th call that fails.
// Release builds compile every hook out.
static int64_t server_test_hook(const char * name) {
#ifdef GGML_DIAGNOSTICS
    const char * value = getenv(name);
    return value ? (int64_t) atoll(value) : 0;
#else
    (void) name;
    return 0;
#endif
}

struct server_slot {
    int id;

    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    common_memory mem;

    // multimodal
    mtmd_context * mctx = nullptr;
    mtmd::batch_ptr mbatch = nullptr;

    // speculative decoding
    common_speculative * spec = nullptr;

    llama_tokens spec_draft;

    // draft candidates per token in spec_draft; only draft-simple and draft-mtp fill it
    std::vector<std::vector<llama_token_data>> spec_draft_q;
    llama_tokens spec_prompt;
    std::vector<int32_t> spec_i_batch;
    common_prompt_checkpoint spec_ckpt;
    bool spec_is_replay = false;
    std::mt19937 spec_synth_rng;

    // TODO: move members that belong to the task (such as `generated`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx   = 0;  // context size per slot
    int32_t n_keep  = 0;
    int32_t i_batch = -1;

    // effective generation limit for the current task, -1 means unlimited
    int32_t n_predict_max = -1;

    size_t last_nl_pos = 0;

    common_chat_input generated;
    std::string  debug_generated_text;
    llama_tokens generated_tokens;
    std::deque<llama_token> loop_guard_tokens;
    size_t loop_guard_generated = 0;
    size_t n_sent_text = 0; // number of sent text character (i.e. handle partial UTF-8 on streaming)

    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;

    stop_type stop;

    std::string stopping_word;

    // state
    slot_state state = SLOT_STATE_IDLE;
    bool resident_paused = false;
    uint64_t borrowed_generation = 0;
    int pipeline_owner = -1;
    int pipeline_lane = -1;

    int system_cache_prefix_tokens = 0;
    int system_cache_restored_tokens = 0;
    int64_t system_cache_restore_us = 0;

    server_prompt prompt;
    std::function<void(int)> ckpt_barrier; // completes this slot's pending async checkpoint captures
    std::function<void(std::vector<uint8_t> &)> ckpt_recycle; // returns a faulted buffer to the pool

    bool prompt_save(server_prompt_cache & prompt_cache) const {
        if (ckpt_barrier) { ckpt_barrier(id); }
        if (prompt.tokens.size() == 0) {
            return false;
        }

        const size_t cur_size_tgt =           llama_state_seq_get_size_ext(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t cur_size_dft = ctx_dft && prompt.mtp_ready ? llama_state_seq_get_size_ext(ctx_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

        const size_t cur_size = cur_size_tgt + cur_size_dft;

        SRV_TRC(" - saving prompt with length %d, total state size = %.3f MiB (draft: %.3f MiB)\n",
                (int) prompt.tokens.size(), cur_size / (1024.0 * 1024.0), cur_size_dft / (1024.0 * 1024.0));

        auto * cur = prompt_cache.alloc(prompt, cur_size_tgt, cur_size_dft);
        if (cur == nullptr) {
            return false;
        }

        llama_state_seq_get_data_ext(ctx_tgt, cur->data.main.data(), cur_size_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (cur_size_dft) {
            llama_state_seq_get_data_ext(ctx_dft, cur->data.drft.data(), cur_size_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }

        return true;
    }

    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens, bool require_mtp = false) {
        if (ckpt_barrier) { ckpt_barrier(id); }
        bool res = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, id, require_mtp);
        if (res && prompt.mtp_ready && !prompt.speculative_state.empty()) {
            common_speculative_set_state(spec, id, prompt.speculative_state);
        }
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
        }

        return res;
    }

    void prompt_clear() {
        if (ckpt_barrier) { ckpt_barrier(id); }
        if (ckpt_recycle) { for (auto & c : prompt.checkpoints) { ckpt_recycle(c.data_tgt); } }
        SLT_TRC(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

        mem.seq_rm(id, -1, -1);

        prompt.clear();
    }

    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;

    llama_token sampled; // in speculative mode, this is the last accepted token

    // for TTS models, this is the embd generated from prev step, decode this to generate next hidden state
    // corresponding to one token position (size = n_embd)
    std::vector<float> inp_embd;

    server_slot_stats stats;

    // accepted tokens per draft position
    // not in server_slot_stats to avoid copying to every task result
    std::vector<uint64_t> n_accepted_per_pos;

    std::function<void(int /* id_slot */)>   callback_on_release;
    std::function<void(const server_slot &)> callback_on_reset; // called before reset()

    // this is for printing timings with slot progress, not part of metrics
    int64_t t_print_last = 0;
    int32_t n_gen_last = 0;

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        mtp_degraded = false;
        spec_is_replay = false;
        resident_paused = false;

        last_nl_pos    = 0;
        generated      = {};
        has_new_line   = false;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stopping_word  = "";
        n_sent_text    = 0;

        if (can_speculate()) {
            spec_draft.clear();
            spec_i_batch.clear();
            spec_ckpt.clear();
        }
        generated_tokens.clear();
        loop_guard_tokens.clear();
        loop_guard_generated = 0;
        generated_token_probs.clear();
        json_schema = json();

        task_prev = std::move(task);
        task.reset();

        // note: callback_on_reset() must have run before this, see release()
        stats = {};
        n_accepted_per_pos.clear();

        n_predict_max = -1;

        llama_set_sampler(ctx_tgt, id, nullptr);

        // clear alora start
        alora_invocation_start = -1;

        // clear multimodal state
        mbatch.reset();
    }

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;

        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];

            if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(smpl.get(), id, false);
                n_text++;
            }
        }

        SLT_TRC(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    bool need_embd() const {
        GGML_ASSERT(task);
        return task->need_embd();
    }

    bool can_split() const {
        GGML_ASSERT(task);
        // a decision over a non-causal context reads the whole prompt in one batch
        if (task->type == SERVER_TASK_TYPE_DECISION && !llama_get_causal_attn(ctx_tgt)) {
            return false;
        }
        // MTP supports splitting - uses task->need_embd() not need_embd()
        if (!task->need_embd()) {
            return true;
        }
        // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
        if (!llama_get_memory(ctx_tgt)) {
            return false;
        }
        // context can be chunked/split if the pooling type is LAST
        const auto pooling = llama_pooling_type(ctx_tgt);
        if (pooling == LLAMA_POOLING_TYPE_LAST) {
            return true;
        }
        // causal rerankers read the last token and have a KV cache, so they can also be chunked/split.
        if (pooling == LLAMA_POOLING_TYPE_RANK && llama_get_causal_attn(ctx_tgt)) {
            return true;
        }
        // a decision task reads its outputs from the last batch
        if (task->type == SERVER_TASK_TYPE_DECISION) {
            return true;
        }
        return false;
    }

    bool can_batch_with(server_slot & other_slot) const {
        GGML_ASSERT(task);

        // a joint decision head reads the whole batch
        if (!task->decision.order.empty() || !other_slot.task->decision.order.empty()) {
            return false;
        }

        return task->type == other_slot.task->type
            && inp_embd.size() == other_slot.inp_embd.size()
            && are_lora_equal(lora, other_slot.lora);
    }

    // returns -1 if the generation is limitless
    int32_t n_remaining() const {
        return n_predict_max == -1 ? -1 : n_predict_max - (int32_t) stats.n_gen;
    }

    bool has_budget() const {
        return n_predict_max == -1 || n_remaining() > 0;
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    bool mtp_degraded = false; // draft disabled after a draft-side memory failure

    bool can_speculate() const {
        if (!spec || !task || mtp_degraded) {
            return false;
        }
        const auto & cfg = task->params.speculative;
        const bool mtp = std::find(cfg.types.begin(), cfg.types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != cfg.types.end();
        return (!mtp || task->params.mtp) && cfg.draft.n_max > 0;
    }

    // at temp 0 both p and q are point masses, so rejection is the same as sample-and-match
    bool use_spec_rejection() const {
        return false;
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    int get_n_draft_max() const {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        // determine the max draft that fits the current slot state
        // note: slot.prompt is not yet expanded with the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context shifts
        int n_draft_max = n_ctx - prompt.n_tokens() - 2;

        if (n_remaining() > 0) {
            n_draft_max = std::min(n_draft_max, n_remaining() - 1);
        }

        SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

        return std::min(n_draft_max, task->params.speculative.draft.n_max);
    }

    // add sampled token of this slot to the batch, optionally add the speculative draft tokens if any
    void handle_last_sampled_token(server_batch & batch) {
        bool add_ok = true;
        if (spec_draft.empty()) {
            // no speculative decoding
            i_batch = batch.size();

            if (!inp_embd.empty()) {
                // text embeddings broadcast the same position across the M-RoPE sections
                const llama_pos p = prompt.tokens.pos_next();
                const llama_pos pos[GGML_MROPE_SECTIONS] = { p, p, p, 0 };
                GGML_ASSERT((int32_t) inp_embd.size() == batch.n_embd);
                add_ok &= batch.add_embd(id, inp_embd.data(), pos, true, false);
            } else {
                add_ok &= batch.add(id, sampled, prompt.tokens.pos_next(), true, false);
            }

            SLT_DBG(*this, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                    sampled, n_ctx, prompt.n_tokens(), truncated);
        } else {
            SLT_DBG(*this, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n",
                    sampled, prompt.tokens.size(), spec_draft.size(), prompt.tokens.pos_next());

            GGML_ASSERT(spec_i_batch.empty());

            spec_i_batch.push_back(batch.size());
            for (size_t i = 0; i < spec_draft.size(); i++) {
                spec_i_batch.push_back(batch.size() + i + 1);
            }

            auto pos0 = prompt.tokens.pos_next();

            add_ok &= batch.add(id, sampled, pos0++, true, false);
            for (auto token : spec_draft) {
                add_ok &= batch.add(this->id, token, pos0++, true, false);
            }
        }

        GGML_ASSERT(add_ok && "batch must be large enough to hold the sampled and draft tokens");

        prompt.tokens.push_back(sampled);
        prompt.tokens.insert(spec_draft);
    }

    void release() {
        if (is_processing()) {
            GGML_ASSERT(task);

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);

            t_last_used = ggml_time_us();

            state = SLOT_STATE_IDLE;

            // do not keep context of the child slots - the parent's context is enough
            // nor of the stateless slots - no task ever reuses it
            if (task->is_child() || task->is_stateless()) {
                prompt_clear();
            }

            callback_on_reset(*this);

            reset();

            callback_on_release(id);
        }
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings_tg() {
        if (stats.n_gen < 100) {
            return;
        }

        const int64_t t_now = ggml_time_us();

        if (t_now - t_print_last < 3*1000*1000) {
            return;
        }

        const double n_gen_second     = stats.n_gen_tps();
        const double n_gen_second_win = 1e6 / (t_now - t_print_last) * (stats.n_gen - n_gen_last);

        t_print_last = t_now;
        n_gen_last = stats.n_gen;

        SLT_INF(*this, "n_gen = %6d, tg = %6.2f t/s, tg_3s = %6.2f t/s\n", (int) stats.n_gen, n_gen_second, n_gen_second_win);
    }

    void print_timings_pp() const {
        const double t_prompt_total = stats.t_prompt_ms();

        if (t_prompt_total < 3000.0) {
            return;
        }

        const double n_prompt_second = stats.n_prompt_tps();
        const double f_progress = task->n_tokens() > 0 ? (double) prompt.n_tokens() / task->n_tokens() : 0.0;

        SLT_INF(*this, "prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second\n",
                (int) stats.n_prompt_processed, f_progress, t_prompt_total / 1e3, n_prompt_second);
    }

    void print_timings() const {
        const double t_prompt_total = stats.t_prompt_ms();
        const double t_gen_total    = stats.t_gen_ms();

        const double t_prompt        = stats.t_prompt_per_token_ms();
        const double n_prompt_second = stats.n_prompt_tps();

        const double t_gen        = stats.t_gen_per_token_ms();
        const double n_gen_second = stats.n_gen_tps();

        SLT_INF(*this,
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_prompt_total, (int) stats.n_prompt_processed, t_prompt, n_prompt_second);

        SLT_INF(*this,
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_gen_total, (int) stats.n_gen, t_gen, n_gen_second);

        SLT_INF(*this,
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_total + t_gen_total, (int) (stats.n_prompt_processed + stats.n_gen));

        SLT_INF(*this,
                "   graphs reused = %10d\n",
                llama_perf_context(ctx_tgt).n_reused);

        SLT_INF(*this,
                "  graph reallocs = %10d    / %5d        (same size) / (changed size)\n",
                llama_perf_context(ctx_tgt).n_reallocs_same, llama_perf_context(ctx_tgt).n_reallocs_diff);

        const int32_t n_draft_total       = stats.n_draft_tokens;
        const int32_t n_draft_accepted    = stats.n_draft_accepted;
        const int32_t n_draft_verif_steps = stats.n_draft_verif_steps;

        if (n_draft_total > 0) {
            const float  draft_ratio  = (float) n_draft_accepted / n_draft_total;
            const double mean_acc_len = n_draft_verif_steps > 0 ? 1.0 + (double) n_draft_accepted / (double) n_draft_verif_steps : 1.0;

            std::string acceptance_rates_per_pos;
            if (n_draft_verif_steps > 0) {
                for (size_t i = 0; i < n_accepted_per_pos.size(); ++i) {
                    if (i > 0) {
                        acceptance_rates_per_pos += ", ";
                    }
                    acceptance_rates_per_pos += string_format("%.3f", (double) n_accepted_per_pos[i] / (double) n_draft_verif_steps);
                }
            }

            SLT_INF(*this,
                    "draft acceptance = %0.5f (%5d accepted / %5d generated), mean len = %5.2f\n",
                    draft_ratio, n_draft_accepted, n_draft_total, mean_acc_len);
            SLT_TRC(*this,
                    "     acc per pos = (%s)\n", acceptance_rates_per_pos.c_str());
        }

        common_speculative_print_stats(spec);
    }

    json to_json(bool only_metrics = false) const {
        json res;

        res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"speculative",   can_speculate()},
            {"is_processing", is_processing()},
            {"resident_paused", resident_paused},
            {"pipeline_lane", pipeline_lane},
            {"kv_stream", ctx_tgt ? llama_memory_seq_stream(llama_get_memory(ctx_tgt), id) : -1},
            {"kv_stream_draft", ctx_dft ? llama_memory_seq_stream(llama_get_memory(ctx_dft), id) : -1},
            {"state", int(state)},
        };

        uint64_t checkpoint_bytes = 0;
        for (const auto & checkpoint : prompt.checkpoints) { checkpoint_bytes += checkpoint.size(); }
        res["context_checkpoints"] = {{"count", prompt.checkpoints.size()}, {"bytes", checkpoint_bytes}};

        const auto & ptask = task ? task : task_prev;

        if (ptask) {
            res["system_cache_policy"] = {
                {"prefix_tokens", system_cache_prefix_tokens},
                {"restored_tokens", system_cache_restored_tokens},
                {"restore_us", system_cache_restore_us},
            };
            res["id_task"] = ptask->id;
            res["n_prompt_tokens"]           = (int32_t) prompt.tokens.size();
            res["n_prompt_tokens_processed"] = stats.n_prompt_processed;
            res["n_prompt_tokens_cache"]     = stats.n_prompt_cached;
            res["params"] = ptask->params.to_json(only_metrics);
            res["next_token"] = json::array({
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining()},
                    {"n_decoded",      stats.n_gen},
                }
            });

            if (!only_metrics) {
                res["prompt"] = ptask->tokens.detokenize(ctx_tgt, true);
                res["generated"] = generated.empty() ? debug_generated_text : generated.text;
            }
        }

        return res;
    }

    // the other slot continues from the tokens processed so far
    void copy_prompt_to(server_slot & other) const {
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), other.id, -1, -1);
        llama_memory_seq_cp(llama_get_memory(ctx_tgt), id, other.id, -1, -1);
        if (ctx_dft && other.ctx_dft) {
            llama_memory_seq_rm(llama_get_memory(other.ctx_dft), other.id, -1, -1);
            if (ctx_dft == other.ctx_dft) {
                llama_memory_seq_cp(llama_get_memory(ctx_dft), id, other.id, -1, -1);
            } else {
                std::vector<uint8_t> data(llama_state_seq_get_size_ext(ctx_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE));
                if (llama_state_seq_get_data_ext(ctx_dft, data.data(), data.size(), id, LLAMA_STATE_SEQ_FLAGS_NONE) != data.size() ||
                        llama_state_seq_set_data_ext(other.ctx_dft, data.data(), data.size(), other.id, LLAMA_STATE_SEQ_FLAGS_NONE) != data.size()) {
                    throw std::runtime_error("cannot copy draft state between pipeline lanes");
                }
            }
            std::vector<uint8_t> spec_data;
            if (common_speculative_get_state(spec, id, spec_data)) {
                common_speculative_set_state(other.spec, other.id, spec_data);
            }
        }

        other.prompt = prompt.clone();
    }

    void copy_state_to(server_slot & other) const {
        GGML_ASSERT(state == SLOT_STATE_DONE_PROMPT);

        copy_prompt_to(other);

        other.i_batch = i_batch;

        other.stats = stats;
        other.loop_guard_tokens = loop_guard_tokens;
        other.loop_guard_generated = loop_guard_generated;

        other.init_sampler();
    }
};

// encode the mtmd chunk at idx, batched with as many of the next chunks as possible
// returns 0 on success
static int encode_mtmd_chunk(const server_slot & slot, mtmd::batch_ptr & mbatch, size_t idx) {
    const auto & mctx = slot.mctx;
    const auto & input_tokens = slot.task->tokens;
    const auto & chunk = input_tokens.find_chunk(idx);

    mbatch.reset(mtmd_batch_init(mctx));
    int32_t res = mtmd_batch_add_chunk(mbatch.get(), chunk.get());
    GGML_ASSERT(res == 0); // we should never have an empty batch

    // try batching as much as possible
    int n_added = 1;
    size_t idx_cur = idx;
    while (res == 0) {
        auto [next_chunk, next_idx] = input_tokens.find_next_media_chunk(idx_cur);
        if (next_chunk == nullptr) {
            break;
        }
        res = mtmd_batch_add_chunk(mbatch.get(), next_chunk->get());
        n_added += (res == 0 ? 1 : 0);
        idx_cur = next_idx;
        SLT_DBG(slot, "try adding media chunk idx = %zu to batch, res = %d\n", next_idx, res);
        // if res != 0, batch is full or chunk is not compatible -> this loop breaks
    }

    // TODO @ngxson : move this log line to debug when it become more stable
    SLT_TRC(slot, "encoding mtmd batch from idx = %zu, n_chunks = %d\n", idx, n_added);

    res = mtmd_batch_encode(mbatch.get());
    if (res != 0) {
        SLT_ERR(slot, "failed to encode mtmd batch for chunk idx = %zu, res = %d\n", idx, res);
        return -1;
    }

    return 0;
}

// returns 0 on success
// caller need to update prompt.tokens after a successful call to keep track of the processing progress
// note: this is not a member of server_slot because we want to run it inside yield_to_queue
//       slot is passed as const to avoid accidental modification of the slot state
//       some pointers are allowed to be used, they are not used by to_json()
static int process_mtmd_chunk(const server_slot & slot, mtmd::batch_ptr & mbatch, size_t idx, size_t & n_tokens_out) {
    GGML_ASSERT(slot.mctx);
    const auto & mctx = slot.mctx;
    const auto & input_tokens = slot.task->tokens;
    const auto & chunk = input_tokens.find_chunk(idx);
    int32_t res = 0;

    auto try_decode = [&]() -> int32_t {
        if (mbatch) {
            float * embd = mtmd_batch_get_output_embd(mbatch.get(), chunk.get());
            if (embd) {
                struct cb_data_t {
                    common_speculative * spec;
                    llama_context * ctx;
                } cb_data = { slot.spec, slot.ctx_tgt };

                static auto cb = [](const mtmd_helper_embd_batch * b, void * user_data) {
                    const auto * data = static_cast<cb_data_t *>(user_data);

                    common_batch batch(data->ctx);
                    for (int32_t i = 0; i < b->n_tokens; ++i) {
                        llama_pos pos[GGML_MROPE_SECTIONS] = { 0, 0, 0, 0 };
                        for (int32_t j = 0; j < b->n_pos; ++j) {
                            pos[j] = b->pos[j * b->n_tokens + i];
                        }
                        batch.add_embd({ b->embd + (size_t) i * b->n_embd, 1, (size_t) b->n_embd }, pos, b->seq_id, false);
                    }

                    return common_speculative_process(data->spec, batch) ? 0 : 1;
                };

                llama_pos new_n_past; // unused for now
                res = mtmd_helper_decode_image_chunk(
                    mctx,
                    slot.ctx_tgt,
                    chunk.get(),
                    embd,
                    slot.prompt.tokens.pos_next(),
                    slot.id,
                    llama_n_batch(slot.ctx_tgt),
                    &new_n_past,
                    cb,
                    &cb_data
                );
                if (res != 0) {
                    SLT_ERR(slot, "failed to decode mtmd chunk, idx = %zu, res = %d\n", idx, res);
                    return -1;
                }
                n_tokens_out = mtmd_input_chunk_get_n_tokens(chunk.get());
                return 0; // success
            }
        }
        return 1; // (non-error) need to create & encode batch
    };

    // if the batch is already exist, try searching & encode
    res = try_decode();
    if (res == 0) {
        return 0;
    }
    if (res < 0) {
        // fatal error
        return res;
    }

    // otherwise, the batch is either uninitialized or is used up
    // we need to create & encode a new batch
    if (encode_mtmd_chunk(slot, mbatch, idx) != 0) {
        return -1;
    }

    return try_decode();
}

// add an encoded mtmd chunk to the batch of the text tokens, instead of decoding it on its own like process_mtmd_chunk()
// embd is the output of the encoder for this chunk
// returns false if the batch is full
static bool add_mtmd_chunk(const server_slot & slot, const mtmd_input_chunk * chunk, const float * embd, server_batch & batch) {
    // positions are the ones of mtmd_helper_decode_image_chunk()
    const auto * image    = mtmd_input_chunk_get_tokens_image(chunk);
    const bool   is_mrope = mtmd_decode_use_mrope(slot.mctx);
    const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
    const size_t n_embd   = batch.n_embd;
    const llama_pos pos_0 = slot.prompt.tokens.pos_next();

    for (size_t i = 0; i < n_tokens; i++) {
        const llama_pos p = pos_0 + (llama_pos) i;
        llama_pos pos[GGML_MROPE_SECTIONS] = { p, p, p, p };
        if (is_mrope && image) {
            const mtmd_decoder_pos rel = mtmd_image_tokens_get_decoder_pos(image, pos_0, i);
            pos[0] = rel.t;
            pos[1] = rel.y;
            pos[2] = rel.x;
            pos[3] = rel.z;
        }
        if (!batch.add_embd(slot.id, embd + i * n_embd, pos, slot.need_embd(), /* is_prompt */ true)) {
            return false;
        }
    }
    return true;
}

//
// server_context_impl (private implementation)
//

// Every verifier result contains at least one token, hence rollback <= draft
// length. Clone only when some legal result can restore the target checkpoint.
// Cloning does not advance the source sampler RNG.
static bool server_speculative_sampler_backup_needed(
        common_context_seq_rm_type removal, uint32_t capacity, size_t draft) {
    return removal == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
            (removal == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft > capacity);
}

const char * paraflow_dash_html(); // server-dash-page.cpp

// Built-in dashboard (--dash). The main loop copies existing counters into this plain struct at most every
// 200 ms while someone polled in the last 10 s, and hands it over with try_lock so it never waits for a reader.
// Readers format JSON on the HTTP thread. Throughput history is kept separately (paraflow_dash_bucket below).
struct paraflow_dash_slot {
    int16_t id, state, lane, stream;
    int32_t n_tokens, n_cells;
    uint8_t vip, mtp, paused, pad;
};
struct paraflow_dash_susp { // a request whose KV was paged out to host RAM
    int16_t  id;            // slot it came from (a VIP may now be using that slot)
    uint8_t  vip, capacity; // capacity = paged out for KV pool space, else to free a slot for a VIP
    int32_t  n_tokens;
    uint64_t bytes;         // host copy of target + draft sequence state
};
struct paraflow_dash_dev {
    char     name[32];
    uint64_t model, context, compute, total;
};
struct paraflow_dash_snap {
    uint64_t seq = 0;
    int64_t  t_us = 0, t_start_us = 0, mem_t_us = 0;
    // cumulative token counters
    uint64_t prompt = 0, predict = 0, cached = 0, decodes = 0;
    uint64_t draft_tokens = 0, draft_accepted = 0, draft_steps = 0;
    uint64_t accepted_pos[8] = {};
    int32_t  n_accepted_pos = 0;
    // pipeline scheduler
    uint64_t settled = 0, cars_prefill = 0, cars_decode = 0, cars_mixed = 0, payload_prompt = 0, payload_decode = 0;
    int64_t  issue_us = 0, wait_us = 0, process_us = 0, drain_us = 0;
    uint64_t admission_us = 0, admission_skipped = 0, vip_pause_skipped = 0, compact_moves = 0, ckpt_holds = 0, deferrals = 0;
    uint64_t oom_isolations = 0, oom_terminated = 0, draft_degrade = 0, draft_failures = 0;
    uint64_t dft_joins = 0, dft_join_waits = 0;
    int64_t  dft_join_wait_us = 0;
    // system prefix cache
    uint64_t sc_hits = 0, sc_hit_tokens = 0, sc_misses = 0, sc_evictions = 0, sc_failures = 0, sc_bytes = 0;
    // live state
    int32_t  n_slots = 0, n_processing = 0, n_prompt = 0, n_gen = 0, n_paused = 0, n_suspended = 0, n_vip = 0, n_deferred = 0;
    uint64_t kv_used = 0, kv_cap = 0, kv_max = 0;
    int32_t  n_rows = 0;
    paraflow_dash_slot rows[256];
    int32_t  n_susp = 0;
    uint64_t susp_bytes = 0;
    paraflow_dash_susp susp[64];
    // cumulative completions per context-length bucket (64 buckets, 1k each: 0-1k, 1-2k, ..., 63-64k, >=64k)
    uint64_t ctx_hist[64] = {};
    int32_t  n_dev = 0;
    paraflow_dash_dev  dev[16];
};

// --dash throughput history: 6 h of 2 s wall-clock buckets kept by the server, so the page shows the same
// curve after a reload or a long background tab. Single writer (main loop, plain relaxed stores); HTTP readers
// copy a bucket seqlock-style and skip one that is being recycled. 10800 x 32 B = 338 KiB, allocated with --dash.
static constexpr int64_t PARAFLOW_DASH_BUCKET_MS = 2000;
static constexpr int64_t PARAFLOW_DASH_BUCKETS   = 6 * 3600 * 1000 / PARAFLOW_DASH_BUCKET_MS;
struct paraflow_dash_bucket {
    std::atomic<uint32_t> id{0};                  // absolute bucket (wall ms / 2000); UINT32_MAX while recycled
    std::atomic<uint32_t> dp{0}, dd{0}, dd_pure{0}; // prefill tokens, decode tokens, decode tokens in pure-decode spans
    std::atomic<uint32_t> ng_ms{0}, np_ms{0};       // integral of generating / prefilling requests (paused excluded) over time
    std::atomic<uint32_t> pure_ms{0}, mix_ms{0};    // time with decode output and no prefill / with both
};
static int64_t paraflow_wall_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

struct server_context_impl {
    friend struct server_context;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model_tgt = nullptr;

    mtmd_context * mctx = nullptr;
    // note: video_params.ffmpeg_bin_dir points into params_base, which outlives this struct
    mtmd_helper_init_opt init_opt = mtmd_helper_init_opt_default();
    const llama_vocab * vocab = nullptr;

    server_decision_context decision;

    server_queue    queue_tasks;
    server_response queue_results;

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    server_state_callback_t callback_state = [](server_state, json) -> void {};

    server_context_impl() {
        mtmd_helper_log_set(common_log_default_callback, nullptr);
    }

    ~server_context_impl() {
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

    server_metrics get_metrics() const {
        return metrics;
    }

    void reset_metrics_bucket() {
        metrics.reset_bucket();
    }

    // HTTP thread: copy the last published dashboard snapshot and mark the dashboard as watched. The loop publishes
    // only while it runs; when it went idle right after a publish (or nothing was published yet) ask for exactly one
    // more pass so the slot table shows the idle state. Once idle is published no further wakeups are requested.
    bool dash_read(paraflow_dash_snap & out, std::string & config) {
        const int64_t now = ggml_time_us();
        dash_last_read_us.store(now, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(dash_mutex);
            out    = dash_pub;
            config = dash_config;
        }
        const bool stale = out.seq == 0 || (out.n_processing > 0 && now - out.t_us > 500 * 1000);
        if (stale && !queue_tasks.is_sleeping()) {
            queue_tasks.request_update();
        }
        return out.seq > 0;
    }

    // HTTP thread: append ,"hist":{...} with complete buckets [from, now). Empty (idle) buckets are written as 0;
    // buckets before this process started are not written, so the page sees them as a gap.
    void dash_hist_json(int64_t from, std::string & o) const {
        if (!dash_hist) { return; }
        const int64_t now_ms = paraflow_wall_ms(), cur = now_ms / PARAFLOW_DASH_BUCKET_MS;
        const int64_t lo  = std::max({ from, dash_hist_start, cur - PARAFLOW_DASH_BUCKETS + 1 });
        char b[192];
        snprintf(b, sizeof(b), ",\"hist\":{\"bucket_ms\":%" PRId64 ",\"now_ms\":%" PRId64 ",\"start\":%" PRId64 ",\"cur\":%" PRId64 ",\"from\":%" PRId64 ",\"rows\":[",
                 PARAFLOW_DASH_BUCKET_MS, now_ms, dash_hist_start, cur, lo);
        o += b;
        for (int64_t k = lo; k < cur; ++k) {
            const auto & e = dash_hist[k % PARAFLOW_DASH_BUCKETS];
            const uint32_t id1 = e.id.load(std::memory_order_acquire);
            const uint32_t v[7] = { e.dp.load(std::memory_order_relaxed), e.dd.load(std::memory_order_relaxed),
                                    e.dd_pure.load(std::memory_order_relaxed), e.ng_ms.load(std::memory_order_relaxed),
                                    e.np_ms.load(std::memory_order_relaxed), e.pure_ms.load(std::memory_order_relaxed),
                                    e.mix_ms.load(std::memory_order_relaxed) };
            std::atomic_thread_fence(std::memory_order_acquire);
            const uint32_t id2 = e.id.load(std::memory_order_relaxed);
            if (k > lo) { o += ','; }
            if (id1 != (uint32_t) k || id2 != (uint32_t) k || !(v[0] | v[1] | v[3] | v[4])) { o += '0'; continue; }
            snprintf(b, sizeof(b), "[%u,%u,%u,%u,%u,%u,%u]", v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
            o += b;
        }
        o += "]}";
    }

private:
    std::mutex             dash_mutex;
    paraflow_dash_snap     dash_pub;                 // guarded by dash_mutex
    std::string            dash_config;              // guarded by dash_mutex, written once
    paraflow_dash_snap     dash_work;                // main thread only
    std::atomic<int64_t>   dash_last_read_us{-1000000000};
    int64_t                dash_last_pub_us = 0;
    uint64_t               dash_ctx_hist[64] = {};   // histogram accumulator (main thread only)
    uint64_t               dash_kv_max = 0;           // KV high-water mark (main thread only)
    std::unique_ptr<paraflow_dash_bucket[]> dash_hist; // throughput history ring (--dash only)
    int64_t                dash_hist_start = 0;       // first bucket of this process (written before serving)
    int64_t                dash_hist_prev_ms = 0;     // main thread only from here on
    bool                   dash_hist_prev_busy = false;
    uint64_t               dash_hist_prompt = 0, dash_hist_predict = 0;

    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    llama_context * ctx_tgt = nullptr;

    server_batch batch;

    // draft executor (see pipeline notes above dft_run)
    struct dft_executor {
        std::mutex mutex;
        std::condition_variable wake, idle;
        std::deque<std::function<void()>> queue;
        std::function<bool()> pump; // one draft head step for the pending drafts; false when none remain
        bool busy = false, stopping = false;
        bool pumping = false; // worker-owned
        std::exception_ptr error;
        uint64_t n_tasks = 0, n_steps = 0, n_joins = 0, n_join_waits = 0;
        int64_t join_wait_us = 0, task_us = 0, step_us = 0;
        std::thread thread;

        void start() { thread = std::thread([this]() { loop(); }); }
        void loop() {
            std::unique_lock<std::mutex> lock(mutex);
            while (true) {
                if (queue.empty() && !pumping) {
                    busy = false;
                    idle.notify_all();
                    wake.wait(lock, [&]() { return stopping || !queue.empty(); });
                    if (queue.empty()) { return; }
                }
                std::function<void()> fn;
                if (!queue.empty()) { fn = std::move(queue.front()); queue.pop_front(); }
                busy = true;
                lock.unlock();
                const bool step = !fn;
                const auto t0 = ggml_time_us();
                std::exception_ptr failure;
                try {
                    if (fn) { fn(); } else { pumping = pump(); }
                } catch (...) {
                    pumping = false;
                    failure = std::current_exception();
                }
                const auto dt = ggml_time_us() - t0;
                lock.lock();
                if (failure && !error) { error = failure; }
                if (step) { ++n_steps; step_us += dt; } else { ++n_tasks; task_us += dt; }
            }
        }
        void push(std::function<void()> fn) {
            { std::lock_guard<std::mutex> lock(mutex); queue.push_back(std::move(fn)); }
            wake.notify_one();
        }
        int64_t join() {
            std::unique_lock<std::mutex> lock(mutex);
            ++n_joins;
            if (queue.empty() && !busy) { return 0; }
            ++n_join_waits;
            const auto t0 = ggml_time_us();
            idle.wait(lock, [&]() { return queue.empty() && !busy; });
            const auto dt = ggml_time_us() - t0;
            join_wait_us += dt;
            return dt;
        }
        std::exception_ptr take_error() {
            std::lock_guard<std::mutex> lock(mutex);
            auto e = error; error = nullptr; return e;
        }
        void stop() {
            { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
            wake.notify_one();
            if (thread.joinable()) { thread.join(); }
        }
        ~dft_executor() { stop(); }
    };

    // scheduler event log (diagnostic)
    struct pipe_event { uint64_t t_ns; uint8_t type, lane; uint16_t ntok, nprompt, nseq; uint32_t ticket; uint64_t slots; };
    static_assert(sizeof(pipe_event) == 32, "pipe_event");
    enum : uint8_t { PE_ISSUE_B = 1, PE_ISSUE_E, PE_STAGE, PE_SETTLE_B, PE_WAIT_E, PE_CATCHUP_B, PE_CATCHUP_E, PE_SETTLE_E,
                     PE_LAUNCH_B, PE_LAUNCH_E, PE_COLLECT_B, PE_COLLECT_E, PE_SLEEP, PE_POST_E };
    FILE * pe_file = nullptr;
    bool pe_checked = false;
    uint64_t pe_t0 = 0, pe_start = 0, pe_end = 0, pe_count = 0;
    static uint64_t pe_now() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t) t.tv_sec * 1000000000ull + t.tv_nsec; }
    // Pipeline event log (LLAMA_DIAGNOSTICS builds): binary scheduler timeline for bubble analysis.
    void pe(uint8_t type, int lane = -1, int ntok = 0, int nprompt = 0, int nseq = 0, uint64_t ticket = 0, uint64_t slot_mask = 0) {
#ifndef GGML_DIAGNOSTICS
        (void) type; (void) lane; (void) ntok; (void) nprompt; (void) nseq; (void) ticket; (void) slot_mask;
#else
        // window opens when the trigger file LLAMA_PIPE_EVENTS_TRIGGER appears (polled every 50 ms), lasts _DUR_S
        if (pe_checked && !pe_file) { return; }
        const uint64_t t = pe_now();
        std::lock_guard<std::mutex> pe_lock(pe_mutex); // the draft executor writes events too
        if (!pe_checked) {
            const char * p = getenv("LLAMA_PIPE_EVENTS"); const char * trig = getenv("LLAMA_PIPE_EVENTS_TRIGGER");
            if (!p || !trig) { pe_checked = true; return; }
            if (t < pe_t0) { return; }
            pe_t0 = t + 50000000ull;
            if (access(trig, F_OK) != 0) { return; }
            pe_checked = true;
            pe_start = t;
            pe_end = t + (uint64_t) (atof(getenv("LLAMA_PIPE_EVENTS_DUR_S") ? getenv("LLAMA_PIPE_EVENTS_DUR_S") : "30") * 1e9);
            pe_file = fopen(p, "wb");
            if (!pe_file) { return; }
            setvbuf(pe_file, nullptr, _IOFBF, 1 << 22);
            SRV_INF("%s", "event log opened\n");
        }
        if (t > pe_end) { fclose(pe_file); pe_file = nullptr; SRV_INF("event log closed: %" PRIu64 " events\n", pe_count); return; }
        pipe_event e = { t, type, (uint8_t) (lane < 0 ? 255 : lane), (uint16_t) ntok, (uint16_t) nprompt, (uint16_t) nseq, (uint32_t) ticket, slot_mask };
        fwrite(&e, sizeof e, 1, pe_file);
        ++pe_count;
#endif
    }
    std::mutex pe_mutex;
    void pe_w(uint8_t type, int ntok = 0, int nseq = 0) {
#ifndef GGML_DIAGNOSTICS
        (void) type; (void) ntok; (void) nseq;
#else
        // draft-executor event (lane 254); written only while the main thread keeps the log open
        std::lock_guard<std::mutex> pe_lock(pe_mutex);
        if (!pe_file) { return; }
        const uint64_t t = pe_now();
        if (t > pe_end) { return; }
        pipe_event e = { t, type, 254, (uint16_t) ntok, 0, (uint16_t) nseq, 0, 0 };
        fwrite(&e, sizeof e, 1, pe_file);
        ++pe_count;
#endif
    }
    uint64_t pe_mask(const server_batch & b) const {
        uint64_t m = 0; for (const auto & token : b.tokens) { if (token.id_slot < 64) { m |= 1ull << token.id_slot; } } return m;
    }

    struct pipeline_lane {
        server_batch batch;
        uint64_t ticket = 0;
        bool in_flight = false;
        std::vector<int> slot_ids;
        int prefill_slot = -1;
        server_task_id prefill_task = -1;
        int prefill_output = -1;
        bool shared_draft_staged = false;
        bool shared_draft_ready = false;
        std::vector<server_slot *> generating;
        std::vector<server_slot *> drafting;
        server_slot * slot_batched = nullptr;
        int64_t issued_us = 0;
    };
    struct suspended_request {
        std::unique_ptr<server_slot> slot;
        std::vector<uint8_t> target;
        std::vector<uint8_t> draft;
        std::function<void()> restore_spec;
        uint64_t generation;
        uint64_t prompt_hash;
        std::string generated;
        size_t sent;
        uint64_t n_gen;
        server_task_id task_id;
        bool capacity_page = false;
    };
    std::vector<suspended_request> suspended_requests;
    uint64_t suspension_generation = 0;

    static uint64_t prompt_hash(const server_slot & slot) {
        return slot.prompt.tokens.continuation_fingerprint();
    }

    int active_vip() const {
        int n = 0;
        for (const auto & slot : slots) {
            n += slot.is_processing() && !slot.resident_paused && slot.task && slot.task->params.priority > 0;
        }
        return n;
    }

    int active_ordinary() const {
        int n = 0;
        for (const auto & slot : slots) {
            n += slot.is_processing() && !slot.resident_paused && slot.task && slot.task->params.priority == 0;
        }
        return n;
    }

    int ordinary_budget(int incoming_vip = 0) const {
        const int vip = active_vip() + incoming_vip;
        return params_base.vip_enable && vip ? std::max(0, pipeline_count - vip) : INT_MAX;
    }

    // would pause_for_vip() pause any ordinary slot? (same budget as below, no mutation)
    bool vip_pause_needed() const {
        const int keep = ordinary_budget(1);
        int ordinary = 0;
        for (const auto & slot : slots) {
            ordinary += slot.is_processing() && !slot.resident_paused && slot.task && slot.task->params.priority == 0;
        }
        return ordinary > keep;
    }
    uint64_t pipeline_vip_pause_skipped = 0;
    void pause_for_vip() {
        if (!vip_pause_needed()) { ++pipeline_vip_pause_skipped; return; }
        pipeline_drain();
        const int keep = ordinary_budget(1);
        std::vector<int> survivors;
        std::vector<bool> covered(pipeline_count, false);
        for (const auto & slot : slots) {
            if (slot.is_processing() && !slot.resident_paused && slot.task->params.priority > 0) {
                covered[pipeline_lane_of(slot.id)] = true;
            }
        }
        for (int pass = 0; pass < 2; ++pass) {
            for (auto & slot : slots) {
                if ((int) survivors.size() >= keep) { break; }
                if (!slot.is_processing() || slot.resident_paused || slot.task->params.priority > 0 ||
                        std::find(survivors.begin(), survivors.end(), slot.id) != survivors.end()) { continue; }
                const int lane = pipeline_lane_of(slot.id);
                if (pass == 0 && covered[lane]) { continue; }
                covered[lane] = true;
                survivors.push_back(slot.id);
            }
        }
        for (auto & slot : slots) {
            if (slot.is_processing() && !slot.resident_paused && slot.task->params.priority == 0 &&
                    std::find(survivors.begin(), survivors.end(), slot.id) == survivors.end()) {
                slot.resident_paused = true;
                SLT_INF(slot, "%s", "VIP resident pause, no KV transfer\n");
            }
        }
    }

    static std::vector<uint8_t> capture_sequence(llama_context * ctx, int id) {
        if (!ctx) { return {}; }
        std::vector<uint8_t> data(llama_state_seq_get_size_ext(ctx, id, LLAMA_STATE_SEQ_FLAGS_NONE));
        if (data.empty() || llama_state_seq_get_data_ext(ctx, data.data(), data.size(), id, LLAMA_STATE_SEQ_FLAGS_NONE) != data.size()) {
            throw std::runtime_error("cannot capture complete sequence state");
        }
        return data;
    }

    static bool restore_sequence(llama_context * ctx, int id, const std::vector<uint8_t> & data, const char * label) {
        if (!ctx) { return data.empty(); }
        const size_t loaded = llama_state_seq_set_data_ext(ctx, data.data(), data.size(), id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (loaded != data.size()) {
            SRV_ERR("continuation %s state load size mismatch, slot=%d, expected=%zu, loaded=%zu\n", label, id, data.size(), loaded);
            return false;
        }
        const auto actual = capture_sequence(ctx, id);
        if (actual != data) {
            size_t first = 0;
            while (first < std::min(actual.size(), data.size()) && actual[first] == data[first]) { ++first; }
            size_t count = 0;
            for (size_t i = 0; i < std::min(actual.size(), data.size()); ++i) { count += actual[i] != data[i]; }
            SRV_ERR("continuation %s readback mismatch, slot=%d, expected=%zu, actual=%zu, first=%zu, differing=%zu\n",
                    label, id, data.size(), actual.size(), first, count);
            return false;
        }
        return true;
    }

    std::map<int, std::vector<uint8_t>> continuation_other_states(int exclude) const {
        std::map<int, std::vector<uint8_t>> result;
        if (!server_test_hook("LLAMA_VERIFY_CONTINUATION_ISOLATION")) { return result; }
        for (const auto & slot : slots) {
            if (slot.id != exclude && slot.is_processing()) { result[slot.id] = capture_sequence(ctx_tgt, slot.id); }
        }
        return result;
    }

    void verify_continuation_isolation(const std::map<int, std::vector<uint8_t>> & before, const char * operation) const {
        for (const auto & entry : before) {
            const auto actual = capture_sequence(ctx_tgt, entry.first);
            if (actual != entry.second) {
                size_t first = 0;
                while (first < std::min(actual.size(), entry.second.size()) && actual[first] == entry.second[first]) { ++first; }
                SRV_ERR("%s changed unrelated seq=%d, before=%zu, after=%zu, first=%zu\n",
                        operation, entry.first, entry.second.size(), actual.size(), first);
                throw std::runtime_error("continuation isolation verification failed");
            }
        }
    }

    bool page_out_paused(int preferred_id = -1, bool capacity_page = false) {
        dft_join(__LINE__);
        server_slot * victim = nullptr;
        std::vector<int> load(pipeline_count, 0);
        for (const auto & slot : slots) {
            if (slot.is_processing() && !slot.resident_paused) { load[pipeline_lane_of(slot.id)]++; }
        }
        for (auto & slot : slots) {
            if (!slot.resident_paused || !slot.task || (slot.borrowed_generation && !capacity_page) ||
                    (preferred_id >= 0 && slot.id != preferred_id)) { continue; }
            if (!victim || load[pipeline_lane_of(slot.id)] < load[pipeline_lane_of(victim->id)] ||
                    (load[pipeline_lane_of(slot.id)] == load[pipeline_lane_of(victim->id)] &&
                     slot.prompt.n_tokens() < victim->prompt.n_tokens())) { victim = &slot; }
        }
        if (!victim) { return false; }
        pipeline_drain();
        // Finish every read of this sequence before capturing or unmapping its pages.
        llama_synchronize(ctx_tgt);
        if (victim->ctx_dft) { llama_synchronize(victim->ctx_dft); }
        const auto other_states = continuation_other_states(victim->id);
        suspended_request saved;
        saved.target = capture_sequence(ctx_tgt, victim->id);
        saved.draft = capture_sequence(victim->ctx_dft, victim->id);
        saved.restore_spec = common_speculative_checkpoint(victim->spec, victim->id);
        saved.generation = ++suspension_generation;
        saved.prompt_hash = prompt_hash(*victim);
        saved.generated = victim->generated.text;
        saved.sent = victim->n_sent_text;
        saved.n_gen = victim->stats.n_gen;
        saved.task_id = victim->task->id;
        saved.capacity_page = capacity_page;
        // Reserve before moving the continuation. An allocation failure cannot lose it.
        suspended_requests.reserve(suspended_requests.size() + 1);
        server_slot fresh;
        fresh.id = victim->id;
        fresh.ctx_tgt = victim->ctx_tgt;
        fresh.ctx_dft = victim->ctx_dft;
        fresh.mem = victim->mem;
        fresh.spec = victim->spec;
        fresh.pipeline_lane = victim->pipeline_lane;
        fresh.n_ctx = victim->n_ctx;
        fresh.mctx = victim->mctx;
        fresh.prompt.tokens.has_mtmd = victim->prompt.tokens.has_mtmd;
        fresh.callback_on_release = victim->callback_on_release;
        fresh.callback_on_reset = victim->callback_on_reset;
        fresh.borrowed_generation = saved.generation;
        saved.slot.reset(new server_slot(std::move(*victim)));
        *victim = std::move(fresh);
        suspended_requests.push_back(std::move(saved));
        victim->mem.seq_rm(victim->id, -1, -1);
        llama_memory_trim(llama_get_memory(ctx_tgt));
        if (victim->ctx_dft) { llama_memory_trim(llama_get_memory(victim->ctx_dft)); }
        verify_continuation_isolation(other_states, "page out");
        SRV_INF("%s borrowed slot=%d, continuation generation=%" PRIu64 "\n", capacity_page ? "capacity" : "VIP", victim->id, victim->borrowed_generation);
        return true;
    }

    uint64_t resident_tokens(int exclude = -1) const {
        uint64_t used = 0;
        for (const auto & slot : slots) {
            if (slot.id != exclude) {
                used += llama_memory_seq_n_cells(llama_get_memory(ctx_tgt), slot.id);
            }
        }
        return used;
    }

    int canonical_prefill_tokens(const server_slot & slot) const {
        if (!pipeline_on || !slot.task || !slot.task->need_sampling() || slot.task->tokens.has_media()) { return 0; }
        const int remaining = slot.task->n_tokens() - slot.prompt.n_tokens();
        if (remaining <= 0) { return 0; }
        // Leave room for generation rows so peers cannot clip the prompt quantum.
        const int room = std::max(1, int(llama_n_batch(ctx_tgt)) - params_base.n_outputs_max);
        const int quantum = std::max(1, std::min({int(llama_n_ubatch(ctx_tgt)),
                    std::max(1, params_base.pipe_prefill_chunk), room, n_ctx}));
        return std::min(remaining, quantum);
    }

    uint64_t generation_reserve() const {
        uint64_t needed = 0;
        uint64_t prefill = 0;
        for (const auto & slot : slots) {
            if (!slot.is_processing() || slot.resident_paused) { continue; }
            if (slot.state == SLOT_STATE_GENERATING) {
                needed += 1 + std::max(0, slot.get_n_draft_max());
            } else if (slot.state == SLOT_STATE_STARTED || slot.state == SLOT_STATE_PROCESSING_PROMPT) {
                prefill = std::max<uint64_t>(prefill, std::max(1, canonical_prefill_tokens(slot)));
            }
        }
        return needed + prefill;
    }

    void ensure_generation_capacity() {
        if (!params_base.kv_spill || (!params_base.kv_dynamic && !params_base.kv_unified)) { return; }
        if (resident_tokens() + generation_reserve() <= (uint64_t) n_ctx) { return; }
        pipeline_drain();
        while (resident_tokens() + generation_reserve() > (uint64_t) n_ctx) {
            if (try_clear_idle_slots()) { continue; }
            server_slot * victim = nullptr;
            int runnable = 0;
            for (auto & slot : slots) {
                if (!slot.is_processing()) { continue; }
                runnable += !slot.resident_paused;
                if ((slot.state != SLOT_STATE_GENERATING && slot.state != SLOT_STATE_PROCESSING_PROMPT) ||
                        llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id) < 0) { continue; }
                if (!victim || slot.task->params.priority < victim->task->params.priority ||
                        (slot.task->params.priority == victim->task->params.priority &&
                         (slot.resident_paused > victim->resident_paused ||
                          (slot.resident_paused == victim->resident_paused && slot.prompt.n_tokens() < victim->prompt.n_tokens())))) { victim = &slot; }
            }
            // Preserve an actor that can advance and eventually free the pool.
            if (!victim || (!victim->resident_paused && runnable < 2)) { return; }
            victim->resident_paused = true;
            try {
                if (!page_out_paused(victim->id, true)) { victim->resident_paused = false; return; }
            } catch (...) {
                victim->resident_paused = false;
                throw;
            }
        }
    }

    void resume_paused() {
        if (queue_tasks.has_pending_vip()) { return; }
        const int budget = ordinary_budget();
        bool changed = false;
        if (suspended_requests.empty() && std::none_of(slots.begin(), slots.end(),
                    [](const server_slot & slot) { return slot.resident_paused; })) { return; }
        dft_join(__LINE__);
        const bool resident_eligible = active_ordinary() < budget && std::any_of(slots.begin(), slots.end(),
                [](const server_slot & slot) { return slot.resident_paused; });
        const bool paged_eligible = std::any_of(suspended_requests.begin(), suspended_requests.end(),
                [&](const suspended_request & saved) {
                    const auto & old = *saved.slot;
                    return !slots.at(old.id).is_processing() &&
                            (old.task->params.priority > 0 || active_ordinary() < budget) &&
                            !(saved.capacity_page && old.task->params.priority <= 0 && active_vip() > 0);
                });
        // A paused task that cannot resume must not drain unrelated in-flight
        // lanes on every scheduling round (notably while a VIP owns the budget).
        if (!resident_eligible && !paged_eligible) { return; }
        pipeline_drain();
        std::stable_sort(suspended_requests.begin(), suspended_requests.end(), [](const suspended_request & a, const suspended_request & b) {
            const int ap = a.slot->task->params.priority, bp = b.slot->task->params.priority;
            return ap != bp ? ap > bp : a.generation > b.generation;
        });
        for (auto it = suspended_requests.begin(); it != suspended_requests.end();) {
            auto & saved = *it;
            auto & dest = slots.at(saved.slot->id);
            auto & old = *saved.slot;
            if (dest.is_processing() || (old.task->params.priority <= 0 && active_ordinary() >= budget) ||
                    (saved.capacity_page && old.task->params.priority <= 0 && active_vip() > 0)) { ++it; continue; }
            if (dest.borrowed_generation != saved.generation) {
                const bool nested = std::any_of(suspended_requests.begin(), suspended_requests.end(), [&](const suspended_request & child) {
                    return child.slot->id == old.id && child.slot->borrowed_generation == saved.generation;
                });
                if (nested) { ++it; continue; }
            }
            if (params_base.kv_dynamic || params_base.kv_unified) {
                const uint64_t need = old.prompt.n_tokens();
                const uint64_t margin = std::min<uint64_t>(256, need < (uint64_t) n_ctx ? ((uint64_t) n_ctx - need) / 16 : 0);
                while (resident_tokens(dest.id) + need + margin > (uint64_t) n_ctx && try_clear_idle_slots()) {}
                if (resident_tokens(dest.id) + need + margin > (uint64_t) n_ctx) { ++it; continue; }
            }
            const bool guard = dest.borrowed_generation == saved.generation && old.task &&
                    old.task->id == saved.task_id && prompt_hash(old) == saved.prompt_hash &&
                    old.generated.text == saved.generated && old.n_sent_text == saved.sent && old.stats.n_gen == saved.n_gen;
            if (!guard) {
                send_error(old, "continuation ownership verification failed", ERROR_TYPE_SERVER);
                old.callback_on_reset(old);
                dest.borrowed_generation = 0;
                it = suspended_requests.erase(it);
                continue;
            }
            bool restored = false;
            const auto other_states = continuation_other_states(dest.id);
            try {
                if (dest.ctx_dft && dest.ctx_dft != old.ctx_dft) {
                    llama_memory_seq_rm(llama_get_memory(dest.ctx_dft), dest.id, -1, -1);
                }
                restored = restore_sequence(ctx_tgt, dest.id, saved.target, "target") &&
                        restore_sequence(old.ctx_dft, dest.id, saved.draft, "draft");
            } catch (const std::exception & e) {
                SRV_ERR("continuation restore failed: %s\n", e.what());
            }
            verify_continuation_isolation(other_states, "restore");
            if (!restored) {
                old.mem.seq_rm(dest.id, -1, -1);
                send_error(old, "continuation KV verification failed", ERROR_TYPE_SERVER);
                old.callback_on_reset(old);
                dest.borrowed_generation = 0;
                it = suspended_requests.erase(it);
                continue;
            }
            if (saved.restore_spec) { saved.restore_spec(); }
            changed = true;
            const auto borrowed_parent = old.borrowed_generation;
            dest = std::move(old);
            dest.borrowed_generation = borrowed_parent;
            dest.resident_paused = false;
            if (dest.smpl && dest.task->params.sampling.backend_sampling && !pipeline_on && !dest.can_speculate() &&
                    !(dest.task->params.sampling.n_probs > 0 && !dest.task->params.post_sampling_probs)) {
                llama_set_sampler(ctx_tgt, dest.id, common_sampler_get(dest.smpl.get()));
            }
            SRV_INF("continuation verified, slot=%d, task=%" PRId64 ", generation=%" PRIu64 "\n", dest.id, dest.task->id, saved.generation);
            it = suspended_requests.erase(it);
        }
        while (active_ordinary() < budget) {
            std::vector<int> load(pipeline_count, 0);
            for (const auto & slot : slots) {
                if (slot.is_processing() && !slot.resident_paused) { ++load[pipeline_lane_of(slot.id)]; }
            }
            server_slot * next = nullptr;
            for (auto & slot : slots) {
                if (slot.resident_paused && (!next || load[pipeline_lane_of(slot.id)] < load[pipeline_lane_of(next->id)])) {
                    next = &slot;
                }
            }
            if (!next) { break; }
            next->resident_paused = false;
            changed = true;
            SLT_INF(*next, "%s", "VIP resident resume\n");
        }
        if (changed && !active_vip()) { queue_tasks.pop_deferred_task(-1); }
    }

    std::vector<pipeline_lane> pipeline_lanes;
    std::deque<int> pipeline_fifo;
    int pipeline_filter = -1;
    int pipeline_prefill_slot = -1;
    size_t pipeline_prefill_next = 0;
    bool pipeline_settling = false;
    int pipeline_count = 1;
    // Opt-in (LLAMA_PIPE_PREFILL_LANES=K): K extra lanes carry only prompt chunks; the
    // pipeline_count decode lanes carry only generation rows (CUDA-graph replay), and
    // ready generation tickets may retire ahead of unfinished prompt tickets.
    int pipeline_prefill_lanes = 0;
    uint64_t pipeline_prompt_overtakes = 0;
    // Test-only fault injection: report prompt tickets not-ready until this long after issue
    // (simulates a slow prompt car so QA fixtures exercise generation overtaking). 0 = off.
    const int64_t pipeline_test_prompt_hold_us = server_test_hook("LLAMA_PIPE_TEST_PROMPT_HOLD_US");
    bool pipeline_on = false;
    bool pipeline_async_drafts = false;
    // Experimental ready-only coalescing. One main-thread owner for all mutable
    // draft state; target tickets remain asynchronous and independently owned.
    bool pipeline_ready_draft_batch = false;
    bool pipeline_stepwise_drafts = false;
    uint64_t pipeline_draft_batch_calls = 0, pipeline_draft_batch_lanes = 0;
    uint64_t pipeline_draft_batch_sequences = 0, pipeline_draft_batch_max_sequences = 0;
    int64_t pipeline_draft_batch_us = 0;
    std::map<size_t, uint64_t> pipeline_draft_batch_histogram;
    int64_t pipeline_profile_issue_us = 0;
    int64_t pipeline_profile_wait_us = 0;
    int64_t pipeline_profile_process_us = 0;
    uint64_t pipeline_profile_count = 0;
    int64_t pipeline_issue_total_us = 0, pipeline_issue_calls = 0;
    int64_t pipeline_drain_us = 0, pipeline_drain_calls = 0;
    uint64_t pipeline_drain_nonempty = 0;
    uint64_t pipeline_prefill_cars = 0, pipeline_decode_cars = 0, pipeline_mixed_cars = 0;
    uint64_t pipeline_prompt_payload = 0, pipeline_decode_payload = 0;
    uint64_t pipeline_admission_calls = 0, pipeline_admission_drain_us = 0;
    uint64_t pipeline_admission_skipped = 0, pipeline_admission_late_drains = 0;
    ggml_backend_buffer_t checkpoint_pinned = nullptr;
    uint64_t checkpoint_pinned_captures = 0;
    uint64_t checkpoint_mtp_seq_captures = 0;
    // stream-ordered async checkpoint capture
    // diagnostics builds: LLAMA_PIPE_CHECKPOINT_VERIFY=1 compares every capture with a synchronous copy
    const bool checkpoint_capture_verify = server_test_hook("LLAMA_PIPE_CHECKPOINT_VERIFY") != 0;
    static constexpr int checkpoint_async_max_bufs = 6;
    struct ckpt_async_buf { ggml_backend_buffer_t buf = nullptr; size_t size = 0; bool busy = false; };
    struct ckpt_async_item { int slot_id; common_prompt_checkpoint * ckpt; int buf; int64_t handle; size_t size; };
    std::vector<ckpt_async_buf> ckpt_async_bufs;
    std::vector<ckpt_async_item> ckpt_async_items;
    uint64_t ckpt_async_enqueued = 0, ckpt_async_immediate = 0, ckpt_async_verified = 0, ckpt_async_mismatches = 0;
    uint64_t ckpt_async_barriers = 0, ckpt_async_settled = 0, ckpt_async_pool_waits = 0;
    static constexpr size_t checkpoint_recycle_max = 4; // recycled host checkpoint buffers
    std::vector<std::vector<uint8_t>> ckpt_recycle_pool;
    uint64_t ckpt_recycle_put = 0, ckpt_recycle_hit = 0, ckpt_recycle_miss = 0;
    void ckpt_recycle_push(std::vector<uint8_t> & v) {
        if (checkpoint_recycle_max == 0 || v.empty() || ckpt_recycle_pool.size() >= checkpoint_recycle_max) { return; }
        ckpt_recycle_pool.push_back(std::move(v)); v = std::vector<uint8_t>(); ++ckpt_recycle_put;
    }
    void ckpt_recycle_take(std::vector<uint8_t> & dst, size_t size) {
        if (checkpoint_recycle_max == 0) { return; }
        for (size_t i = ckpt_recycle_pool.size(); i-- > 0; ) {
            if (ckpt_recycle_pool[i].capacity() >= size) {
                dst = std::move(ckpt_recycle_pool[i]); ckpt_recycle_pool.erase(ckpt_recycle_pool.begin() + i); ++ckpt_recycle_hit; return;
            }
        }
        ++ckpt_recycle_miss;
    }
    uint8_t * ckpt_async_base(int b) { return (uint8_t *) ggml_backend_buffer_get_base(ckpt_async_bufs[b].buf); }
    void ckpt_async_finish(size_t i, bool copy) {
        auto it = ckpt_async_items[i];
        if (copy) { memcpy(it.ckpt->data_tgt.data(), ckpt_async_base(it.buf), it.size); }
        ckpt_async_bufs[it.buf].busy = false;
        ckpt_async_items.erase(ckpt_async_items.begin() + i);
    }
    void ckpt_async_poll(int slot_id, bool block) {
        if (block) { for (auto & it : ckpt_async_items) { if (slot_id < 0 || it.slot_id == slot_id) { ++ckpt_async_barriers; break; } } }
        for (size_t i = 0; i < ckpt_async_items.size(); ) {
            auto & it = ckpt_async_items[i];
            if (slot_id >= 0 && it.slot_id != slot_id) { ++i; continue; }
            const int32_t r = llama_state_seq_async_wait(ctx_tgt, it.handle, block);
            if (r == 0) { ++i; continue; }
            GGML_ASSERT(r == 1);
            ckpt_async_finish(i, true);
        }
    }
    void ckpt_async_settle(const common_prompt_checkpoint * p) {
        for (size_t i = 0; i < ckpt_async_items.size(); ++i) {
            if (ckpt_async_items[i].ckpt != p) { continue; }
            GGML_ASSERT(llama_state_seq_async_wait(ctx_tgt, ckpt_async_items[i].handle, true) == 1);
            ckpt_async_finish(i, false);
            ++ckpt_async_settled;
            return;
        }
    }
    // Pinned host memory of the target's first device (pageable CPU memory if the backend has none).
    ggml_backend_buffer_type_t pinned_host_buft() const {
        ggml_backend_dev_t dev = llama_model_n_devices(model_tgt) > 0 ? llama_model_get_device(model_tgt, 0) : nullptr;
        ggml_backend_buffer_type_t buft = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        return buft ? buft : ggml_backend_cpu_buffer_type();
    }
    int ckpt_async_acquire(size_t size) {
        for (int pass = 0; pass < 3; ++pass) {
            for (int b = 0; b < (int) ckpt_async_bufs.size(); ++b) {
                if (!ckpt_async_bufs[b].busy && ckpt_async_bufs[b].size >= size) { ckpt_async_bufs[b].busy = true; return b; }
            }
            if ((int) ckpt_async_bufs.size() < checkpoint_async_max_bufs) {
                ckpt_async_buf nb;
                nb.size = size + size / 8;
                nb.buf = ggml_backend_buft_alloc_buffer(pinned_host_buft(), nb.size);
                if (nb.buf) { nb.busy = true; ckpt_async_bufs.push_back(nb); return (int) ckpt_async_bufs.size() - 1; }
            }
            for (int b = 0; b < (int) ckpt_async_bufs.size(); ++b) {
                if (!ckpt_async_bufs[b].busy && ckpt_async_bufs[b].size < size) {
                    ggml_backend_buffer_free(ckpt_async_bufs[b].buf);
                    ckpt_async_bufs[b].size = size + size / 8;
                    ckpt_async_bufs[b].buf = ggml_backend_buft_alloc_buffer(pinned_host_buft(), ckpt_async_bufs[b].size);
                    GGML_ASSERT(ckpt_async_bufs[b].buf != nullptr);
                    ckpt_async_bufs[b].busy = true; return b;
                }
            }
            if (pass == 0) { ckpt_async_poll(-1, false); }
            else if (!ckpt_async_items.empty()) {
                ++ckpt_async_pool_waits;
                GGML_ASSERT(llama_state_seq_async_wait(ctx_tgt, ckpt_async_items.front().handle, true) == 1);
                ckpt_async_finish(0, true);
            }
        }
        return -1;
    }
    // MTP admissions may skip the drain only with the draft executor + ready-batch single driver
    uint64_t pipeline_admission_mtp_skipped = 0;
    bool admission_drain_skippable(const server_task & task) {
        if (task.type != SERVER_TASK_TYPE_COMPLETION) { return false; }
        const bool mtp_ok = dft_exec && pipeline_ready_draft_batch;
        const bool vip_free = !vip_pause_needed();
        if ((params_base.vip_enable && task.params.priority > 0 && !vip_free) || (task.params.mtp && !mtp_ok) || system_cache_limit != 0) { return false; }
        if (!suspended_requests.empty() || (queue_tasks.has_pending_vip() && !vip_free)) { return false; }
        bool mtp_seen = task.params.mtp;
        for (const auto & slot : slots) {
            if (slot.resident_paused) { return false; }
            if (slot.is_processing() && slot.task && slot.task->params.mtp) {
                if (!mtp_ok) { return false; }
                mtp_seen = true;
            }
        }
        for (const auto & lane : pipeline_lanes) {
            if (lane.shared_draft_staged && !mtp_ok) { return false; }
            mtp_seen |= lane.shared_draft_staged;
        }
        if (mtp_seen && task.params.mtp) { ++pipeline_admission_mtp_skipped; }
        return true;
    }
    uint64_t pipeline_ready_reorders = 0;
    uint64_t pipeline_compact_moves = 0;
    std::vector<int> pipeline_compact_ids_cache, pipeline_compact_slots_cache, pipeline_compact_lanes_cache, pipeline_compact_weights_cache;
    uint64_t pipeline_compact_plan_cache_hits = 0;
    int64_t pipeline_compact_plan_us = 0, pipeline_compact_plan_calls = 0;
    uint64_t pipeline_work_physical_improvements = 0, pipeline_work_physical_fallbacks = 0;
    int64_t speculative_sampler_backups = 0, speculative_sampler_backup_us = 0;
    uint64_t speculative_sampler_backup_elided = 0;

    int64_t pipeline_work_physical_us = 0, pipeline_work_physical_calls = 0;
    std::vector<int64_t> pipeline_work_plan_signature, pipeline_work_round_signature;
    std::vector<uint64_t> pipeline_work_issued;
    std::vector<bool> pipeline_work_speculative;
    bool pipeline_work_mixed = false;
    uint64_t pipeline_work_moves = 0, pipeline_work_deferrals = 0, pipeline_work_fused_cars = 0;
    uint64_t prompt_checkpoint_saves = 0, prompt_checkpoint_bytes = 0;
    int64_t prompt_checkpoint_save_us = 0;

    uint64_t pipeline_draft_degrade_events = 0, pipeline_draft_degraded_slots = 0;

    void pipeline_degrade_drafts(const std::vector<int> & ids) {
        dft_join(__LINE__);
        ++pipeline_draft_degrade_events;
        for (const int id : ids) {
            auto & slot = slots.at(id);
            if (slot.mtp_degraded || !slot.spec) { continue; }
            slot.mtp_degraded = true;
            slot.prompt.mtp_ready = false;
            slot.prompt.speculative_state.clear();
            if (slot.ctx_dft) { llama_memory_seq_rm(llama_get_memory(slot.ctx_dft), slot.id, -1, -1); }
            ++pipeline_draft_degraded_slots;
            SLT_WRN(slot, "%s", "draft memory failure: continuing this request without speculative drafts\n");
        }
        update_speculative_masks();
    }

    void update_speculative_masks() {
        dft_join(__LINE__);
        for (const auto & slot : slots) {
            common_speculative_set_enabled(spec.get(), slot.id,
                    slot.can_speculate() && slot.spec == spec.get());
        }
    }

    const int64_t pipeline_test_fail_catchup = server_test_hook("LLAMA_PIPE_TEST_FAIL_CATCHUP");
    int64_t pipeline_catchup_calls = 0;
    const int64_t pipeline_test_fail_draftstep = server_test_hook("LLAMA_PIPE_TEST_FAIL_DRAFTSTEP");
    int64_t pipeline_draftstep_calls = 0;
    uint64_t pipeline_draft_step_failures = 0;

    // draft executor state
    std::unique_ptr<dft_executor> dft_exec;
    bool dft_route = false;                        // true while pipeline_step() runs: submissions go to the worker
    std::unique_ptr<std::atomic<bool>[]> dft_done; // per slot: staged draft finished (release by worker, acquire by scheduler)
    std::vector<int> dft_w_active;                 // worker-owned: slots with pending drafts
    std::mutex dft_fail_mutex;
    std::vector<int> dft_fail_ids;                 // worker -> scheduler: slots whose drafting failed (degrade)
    std::vector<int> dft_fail_pending;             // scheduler-owned: failures waiting until their slots are idle
    static constexpr int DFT_RING = 32;
    std::vector<std::vector<float>> dft_hbuf;      // copied target NextN rows per submitted catch-up
    std::vector<common_batch> dft_bbuf;            // copied batch entries per submitted catch-up
    uint64_t dft_h_submitted = 0;
    std::atomic<uint64_t> dft_h_done{0};
    int32_t dft_n_embd = 0;
    int64_t dft_copy_us = 0;
    uint64_t dft_copy_rows = 0, dft_ring_waits = 0, dft_catchups = 0, dft_direct_catchups = 0;

    mutable std::map<int, std::pair<uint64_t, int64_t>> dft_join_sites; // source line -> (waits, wait us); scheduler thread only
    void dft_join(int site = 0) const {
        if (!dft_exec) { return; }
        const int64_t dt = dft_exec->join();
        if (dt > 0) { auto & e = dft_join_sites[site]; ++e.first; e.second += dt; }
    }

    // Ordered access to spec/ctx_dft: queued on the worker inside pipeline_step(), else run here after a join.
    void dft_run(std::function<void()> fn) {
        if (dft_exec && dft_route) { dft_exec->push(std::move(fn)); return; }
        dft_join(__LINE__);
        fn();
    }

    // slot.mem.seq_rm with the draft half ordered through the executor
    void dft_mem_seq_rm(server_slot & slot, llama_pos p0, llama_pos p1) {
        if (!dft_exec || !dft_route || !slot.mem.ctx_dft) {
            dft_join(__LINE__);
            slot.mem.seq_rm(slot.id, p0, p1);
            return;
        }
        if (!llama_memory_seq_rm(llama_get_memory(slot.mem.ctx_tgt), slot.id, p0, p1)) {
            GGML_ABORT("failed to remove sequence %d with p0=%d, p1=%d\n", slot.id, p0, p1);
        }
        dft_exec->push([ctx = slot.mem.ctx_dft, id = slot.id, p0, p1]() {
            if (!llama_memory_seq_rm(llama_get_memory(ctx), id, p0, p1)) {
                GGML_ABORT("failed to remove draft sequence %d with p0=%d, p1=%d\n", id, p0, p1);
            }
        });
    }

    // Submit the draft catch-up of a settled target car. The NextN rows are copied while the
    // car's output ticket is selected; the worker never reads ctx_tgt.
    void dft_submit_catchup(const common_batch & view) {
        std::vector<std::pair<int, bool>> mask;
        mask.reserve(slots.size());
        for (const auto & slot : slots) { mask.emplace_back(slot.id, slot.can_speculate() && slot.spec == spec.get()); }
        bool any = false;
        std::vector<int> ids;
        for (const auto & token : view.tokens) {
            auto & slot = slots[token.seq_id];
            slot.prompt.mtp_ready = slot.can_speculate();
            if (!slot.prompt.mtp_ready) { slot.prompt.speculative_state.clear(); }
            any |= mask[token.seq_id].second;
            if (std::find(ids.begin(), ids.end(), slot.id) == ids.end()) { ids.push_back(slot.id); }
        }
        const bool fail = pipeline_test_fail_catchup > 0 && ++pipeline_catchup_calls == pipeline_test_fail_catchup;
        int idx = -1;
        if (any && !fail) {
            if (view.has_embd()) {
                // embedding entries carry non-owning views; keep them on the scheduler
                ++dft_direct_catchups;
                dft_join(__LINE__);
                for (const auto & m : mask) { common_speculative_set_enabled(spec.get(), m.first, m.second); }
                if (!common_speculative_process(spec.get(), view)) {
                    std::lock_guard<std::mutex> lock(dft_fail_mutex);
                    dft_fail_ids.insert(dft_fail_ids.end(), ids.begin(), ids.end());
                }
                return;
            }
            const auto t0 = ggml_time_us();
            if (dft_h_submitted - dft_h_done.load(std::memory_order_acquire) >= (uint64_t) DFT_RING) {
                ++dft_ring_waits;
                while (dft_h_submitted - dft_h_done.load(std::memory_order_acquire) >= (uint64_t) DFT_RING) { std::this_thread::yield(); }
            }
            idx = (int) (dft_h_submitted % DFT_RING);
            const int n = view.size();
            const float * base = llama_get_embeddings_nextn(ctx_tgt);
            if (!base) { throw std::runtime_error("target NextN rows are unavailable for the draft catch-up"); }
            if (dft_catchups < 64) {
                // the catch-up indexes rows both ways; they must be the same contiguous rows
                for (int k = 0; k < n; ++k) {
                    if (!mask[view.tokens[k].seq_id].second) { continue; }
                    if (llama_get_embeddings_nextn_ith(ctx_tgt, k) != base + (size_t) k * dft_n_embd) {
                        throw std::runtime_error("target NextN rows are not contiguous");
                    }
                }
            }
            dft_bbuf[idx].tokens = view.tokens;
            auto & h = dft_hbuf[idx];
            h.resize((size_t) n * dft_n_embd);
            std::memcpy(h.data(), base, h.size() * sizeof(float));
            dft_copy_rows += n;
            dft_copy_us += ggml_time_us() - t0;
            ++dft_catchups;
        }
        ++dft_h_submitted;
        dft_exec->push([this, mask = std::move(mask), ids = std::move(ids), idx, fail]() {
            for (const auto & m : mask) { common_speculative_set_enabled(spec.get(), m.first, m.second); }
            bool ok = !fail;
            if (idx >= 0) {
                pe_w(PE_CATCHUP_B, dft_bbuf[idx].size());
                ok = common_speculative_process_tgt_embd(spec.get(), dft_bbuf[idx], dft_hbuf[idx].data());
                pe_w(PE_CATCHUP_E, dft_bbuf[idx].size());
            }
            dft_h_done.fetch_add(1, std::memory_order_release);
            if (!ok) {
                std::lock_guard<std::mutex> lock(dft_fail_mutex);
                dft_fail_ids.insert(dft_fail_ids.end(), ids.begin(), ids.end());
            }
        });
    }

    // Worker: one head step over every pending draft (the ready batch), then publish finished slots.
    bool dft_pump_step() {
        auto * sp = spec.get();
        std::vector<int> drafting_now;
        for (const int id : dft_w_active) {
            if (common_speculative_get_draft_params(sp, id).drafting) { drafting_now.push_back(id); }
        }
        if (!drafting_now.empty()) {
            const int n = (int) drafting_now.size();
            try {
                // QA hook, as on the scheduler path: only at the start of a step (all drafts empty)
                if (pipeline_test_fail_draftstep > 0 && std::all_of(drafting_now.begin(), drafting_now.end(), [&](int id) {
                            auto & dp = common_speculative_get_draft_params(sp, id); return !dp.result || dp.result->empty(); }) &&
                        ++pipeline_draftstep_calls == pipeline_test_fail_draftstep) {
                    throw std::runtime_error("test draft step failure");
                }
                pe_w(PE_LAUNCH_B, 0, n);
                common_speculative_draft_step(sp);
                pe_w(PE_COLLECT_E, 0, n);
                ++pipeline_draft_batch_calls;
                pipeline_draft_batch_sequences += n;
                pipeline_draft_batch_max_sequences = std::max<uint64_t>(pipeline_draft_batch_max_sequences, n);
                ++pipeline_draft_batch_histogram[n];
            } catch (const std::exception & e) {
                // as on the scheduler path: drop this round's drafts, the longest-context request gives up drafting
                int victim = -1;
                size_t victim_len = 0;
                for (const int id : drafting_now) {
                    auto & dp = common_speculative_get_draft_params(sp, id);
                    dp.drafting = false;
                    if (dp.result) { dp.result->clear(); }
                    if (dp.result_q) { dp.result_q->clear(); }
                    const size_t len = dp.prompt ? dp.prompt->size() : 0;
                    if (victim < 0 || len > victim_len) { victim = id; victim_len = len; }
                }
                ++pipeline_draft_step_failures;
                SRV_WRN("draft step failed on the draft executor (%s): %d drafts dropped, degrading slot %d\n", e.what(), n, victim);
                if (victim >= 0) {
                    std::lock_guard<std::mutex> lock(dft_fail_mutex);
                    dft_fail_ids.push_back(victim);
                }
            }
        }
        const auto first = std::remove_if(dft_w_active.begin(), dft_w_active.end(), [&](int id) {
            if (common_speculative_get_draft_params(sp, id).drafting) { return false; }
            dft_done[id].store(true, std::memory_order_release);
            return true;
        });
        dft_w_active.erase(first, dft_w_active.end());
        return !dft_w_active.empty();
    }

    // Scheduler: surface worker errors, apply deferred degrades, refresh lane readiness.
    void dft_poll() {
        if (auto e = dft_exec->take_error()) { std::rethrow_exception(e); }
        {
            std::lock_guard<std::mutex> lock(dft_fail_mutex);
            dft_fail_pending.insert(dft_fail_pending.end(), dft_fail_ids.begin(), dft_fail_ids.end());
            dft_fail_ids.clear();
        }
        if (!dft_fail_pending.empty()) {
            std::vector<int> now, later;
            for (const int id : dft_fail_pending) {
                bool busy = slots.at(id).pipeline_owner >= 0;
                for (const auto & lane : pipeline_lanes) {
                    if (!lane.shared_draft_staged) { continue; }
                    for (const auto * slot : lane.drafting) { busy |= slot->id == id; }
                }
                auto & dst = busy ? later : now;
                if (std::find(dst.begin(), dst.end(), id) == dst.end()) { dst.push_back(id); }
            }
            dft_fail_pending = std::move(later);
            if (!now.empty()) { pipeline_degrade_drafts(now); }
        }
        for (auto & lane : pipeline_lanes) {
            if (lane.shared_draft_staged && !lane.shared_draft_ready) {
                lane.shared_draft_ready = std::all_of(lane.drafting.begin(), lane.drafting.end(),
                        [&](const server_slot * slot) { return dft_done[slot->id].load(std::memory_order_acquire); });
            }
        }
    }

    bool process_speculative_batch(const common_batch & view) {
        // QA hook: report the N-th draft catch-up as failed (draft memory of its requests is purged on degrade)
        if (pipeline_test_fail_catchup > 0 && ++pipeline_catchup_calls == pipeline_test_fail_catchup) { return false; }
        update_speculative_masks();
        for (const auto & token : view.tokens) {
            auto & slot = slots[token.seq_id];
            slot.prompt.mtp_ready = slot.can_speculate();
            if (!slot.prompt.mtp_ready) { slot.prompt.speculative_state.clear(); }
        }
        return common_speculative_process(spec.get(), view);
    }

    void refresh_speculative_prompt_states() {
        std::set<int> seen;
        for (const auto & token : batch.tokens) {
            if (!seen.insert(token.id_slot).second) { continue; }
            auto & slot = slots[token.id_slot];
            if (slot.prompt.mtp_ready) {
                common_speculative_get_state(slot.spec, slot.id, slot.prompt.speculative_state);
            }
        }
    }

    void draft_speculative_batch() {
        dft_join(__LINE__);
        common_speculative_draft(spec.get());
    }

    int pipeline_lane_of(int id) const {
        return slots.at(id).pipeline_lane >= 0 ? slots.at(id).pipeline_lane :
                id * pipeline_count / std::max(1, params_base.n_parallel);
    }

    // Non-unified KV: a ubatch spans adjacent KV streams, so the planners order cars by stream (slot id when the
    // memory keeps no per-sequence streams). Streams belong to the engine, slot ids to the client.
    int pipeline_stream_of(int id) const {
        const int32_t stream = llama_memory_seq_stream(llama_get_memory(ctx_tgt), id);
        return stream >= 0 ? stream : id;
    }

    uint64_t pipeline_place_swaps = 0, pipeline_place_kept = 0, pipeline_place_refused = 0, pipeline_place_lost = 0;
    int64_t pipeline_place_us = 0;

    // LLAMA_KV_PLACEMENT=0 disables stream placement for diagnostic comparisons.
    const bool kv_placement = [] { const char * v = getenv("LLAMA_KV_PLACEMENT"); return !v || atoi(v) != 0; }();

    void pipeline_place_slot(server_slot & slot, bool /* speculative */, int priority) {
        // A burst's membership is not known at admission. Pack streams at a settled generation boundary.
        if (!kv_placement || slot.pipeline_owner >= 0) { return; }
        ++pipeline_place_kept;
        pipeline_follow_lane(slot, priority);
    }

    uint64_t pipeline_defrag_passes = 0, pipeline_defrag_moves = 0, pipeline_defrag_refused = 0;
    int64_t pipeline_defrag_us = 0;
    std::vector<int> pipeline_defrag_signature;

    bool pipeline_defrag_members(std::vector<int> & signature, std::vector<int> & mtp, std::vector<int> & ar) {
        signature.clear(); mtp.clear(); ar.clear();
        for (const auto & slot : slots) {
            // Protected resident requests cannot be used as destinations or moved by this pass.
            if (slot.resident_paused || slot.borrowed_generation) { return false; }
            if (!slot.is_processing()) { continue; }
            if (slot.state != SLOT_STATE_GENERATING || !slot.task || slot.task->params.priority > 0 ||
                    slot.task->is_parent() || slot.task->is_child()) { return false; }
            const int stream = llama_memory_seq_stream(llama_get_memory(ctx_tgt), slot.id);
            if (stream < 0) { return false; }
            const bool speculative = slot.can_speculate();
            signature.insert(signature.end(), { slot.id, stream, speculative ? 1 : 0 });
            (speculative ? mtp : ar).push_back(slot.id);
        }
        return true;
    }

    void pipeline_defrag_streams() {
        if (!kv_placement) { return; }
        std::vector<int> signature, mtp, ar;
        if (!pipeline_defrag_members(signature, mtp, ar) || signature == pipeline_defrag_signature) { return; }
        auto * mem = llama_get_memory(ctx_tgt);
        const int n = params_base.n_parallel;
        auto outside = [&](int id, bool spec, int count) {
            const int stream = llama_memory_seq_stream(mem, id);
            return spec ? stream >= count : stream < n - count;
        };
        bool needed = false;
        for (const int id : mtp) { needed |= outside(id, true, mtp.size()); }
        for (const int id : ar)  { needed |= outside(id, false, ar.size()); }
        if (!needed) { pipeline_defrag_signature = signature; return; }

        const int64_t started = ggml_time_us();
        pipeline_drain();
        dft_join(__LINE__);
        if (!pipeline_defrag_members(signature, mtp, ar)) { return; }
        GGML_ASSERT(pipeline_fifo.empty());
        ++pipeline_defrag_passes;
        bool refused = false;
        for (const bool spec : { true, false }) {
            auto ids = spec ? mtp : ar;
            std::sort(ids.begin(), ids.end(), [&](int a, int b) {
                return spec ? pipeline_stream_of(a) > pipeline_stream_of(b) : pipeline_stream_of(a) < pipeline_stream_of(b);
            });
            for (const int id : ids) {
                if (!outside(id, spec, ids.size())) { continue; }
                auto & slot = slots.at(id);
                server_slot * partner = nullptr;
                int best = n + 1;
                for (auto & other : slots) {
                    if (other.id == id || other.pipeline_owner >= 0 || other.resident_paused || other.borrowed_generation) { continue; }
                    // The opposite class may occupy our destination; both live states follow their sequence IDs.
                    if (other.is_processing() && other.can_speculate() == spec) { continue; }
                    const int stream = llama_memory_seq_stream(mem, other.id);
                    if (stream < 0) { continue; }
                    const int rank = spec ? stream : n - 1 - stream;
                    if (rank < int(ids.size()) && rank < best) { best = rank; partner = &other; }
                }
                if (!partner || slot.pipeline_owner >= 0) { ++pipeline_defrag_refused; refused = true; continue; }
                for (auto * member : { &slot, partner }) {
                    if (member->ckpt_barrier) { member->ckpt_barrier(member->id); }
                }
                const int result = llama_memory_seq_swap_stream(mem, slot.id, partner->id);
                if (result == 0) { ++pipeline_defrag_refused; refused = true; continue; }
                if (result < 0) {
                    ++pipeline_place_lost;
                    for (auto * member : { &slot, partner }) {
                        if (member->is_processing()) {
                            send_error(*member, "KV stream exchange failed; request state unavailable", ERROR_TYPE_SERVER);
                            member->release();
                        }
                        member->prompt_clear();
                    }
                    pipeline_defrag_signature.clear();
                    pipeline_compact_ids_cache.clear();
                    pipeline_defrag_us += ggml_time_us() - started;
                    return;
                }
                if (ctx_dft && llama_memory_seq_swap_stream(llama_get_memory(ctx_dft), slot.id, partner->id) < 0) {
                    pipeline_degrade_drafts({ slot.id, partner->id });
                }
                ++pipeline_place_swaps;
                ++pipeline_defrag_moves;
                SLT_INF(slot, "settled stream exchange with slot %d, stream=%d\n", partner->id, pipeline_stream_of(slot.id));
            }
        }
        pipeline_compact_ids_cache.clear();
        pipeline_compact_slots_cache.clear();
        pipeline_compact_lanes_cache.clear();
        pipeline_compact_weights_cache.clear();
        if (refused || !pipeline_defrag_members(signature, mtp, ar)) { pipeline_defrag_signature.clear(); }
        else { pipeline_defrag_signature = std::move(signature); }
        pipeline_defrag_us += ggml_time_us() - started;
    }

    uint64_t pipeline_follow_moves = 0;

    // A starting slot has no ticket, fifo entry or draft in flight: its lane can change freely.
    void pipeline_follow_lane(server_slot & slot, int priority) {
        if (!pipeline_on || priority > 0 || slot.pipeline_owner >= 0) { return; }
        const int stream = pipeline_stream_of(slot.id);
        std::vector<int> load(pipeline_count, 0);
        int active = 0, left = -1, right = -1;
        for (const auto & other : slots) {
            if (other.id == slot.id || !other.is_processing() || other.resident_paused) { continue; }
            const int lane = pipeline_lane_of(other.id);
            ++load[lane]; ++active;
            if (other.task && other.task->params.priority > 0) { continue; }
            const int s = pipeline_stream_of(other.id);
            if (s == stream - 1) { left = lane; }
            if (s == stream + 1) { right = lane; }
        }
        const int share = (active + 1 + pipeline_count - 1) / pipeline_count;
        int lane = int(std::min_element(load.begin(), load.end()) - load.begin());
        for (const int candidate : { left, right }) {
            if (candidate >= 0 && load[candidate] < share) { lane = candidate; break; }
        }
        const int previous = pipeline_lane_of(slot.id);
        if (lane == previous) { return; }
        slot.pipeline_lane = lane;
        ++pipeline_follow_moves;
        SLT_INF(slot, "pipeline lane %d -> %d at launch (kv stream %d)\n", previous, lane, stream);
    }

    int pipeline_work_weight(const server_slot & slot) const {
        // Use the requested cap, not the changing tail budget: placement must
        // not churn every time an actor approaches its output/context boundary.
        return 1 + (slot.can_speculate() ? std::max(0, std::min(
                params_base.speculative.draft.n_max, slot.task->params.speculative.draft.n_max)) : 0);
    }

    std::vector<int64_t> pipeline_work_membership() const {
        int first_weight = 0;
        int first_priority = -1;
        bool heterogeneous = false, priority_mix = false, speculative = false;
        for (const auto & slot : slots) {
            if (!slot.is_processing() || slot.resident_paused) { continue; }
            if (slot.state != SLOT_STATE_GENERATING) { return {}; }
            const int weight = pipeline_work_weight(slot);
            speculative |= weight > 1;
            const int priority = slot.task->params.priority;
            if (first_priority < 0) { first_priority = priority; }
            else { priority_mix |= priority != first_priority; }
            if (!first_weight) { first_weight = weight; }
            else { heterogeneous |= weight != first_weight; }
        }
        // Ordinary C15 and equal-depth MTP keep the accepted scheduling path,
        // with no signature allocations or placement search in the hot loop.
        if (!heterogeneous && !(speculative && priority_mix)) { return {}; }
        // {stream, slot, task, lane, weight} in stream order
        std::vector<std::array<int64_t, 5>> entries;
        for (const auto & slot : slots) {
            if (!slot.is_processing() || slot.resident_paused) { continue; }
            if (slot.state != SLOT_STATE_GENERATING) { return {}; }
            entries.push_back({pipeline_stream_of(slot.id), slot.id, slot.task->id,
                    pipeline_lane_of(slot.id), pipeline_work_weight(slot)});
        }
        std::sort(entries.begin(), entries.end());
        std::vector<int64_t> result;
        for (const auto & e : entries) { result.insert(result.end(), e.begin(), e.end()); }
        return result;
    }

    bool pipeline_rebalance_work() {
        auto membership = pipeline_work_membership();
        if (membership.empty()) { return false; }
        const auto heterogeneous = [](const std::vector<int64_t> & key) {
            for (size_t i = 9; i < key.size(); i += 5) {
                if (key[i] != key[4]) { return true; }
            }
            return false;
        };
        if (!heterogeneous(membership)) { return false; }
        if (membership == pipeline_work_plan_signature) { return true; }
        const auto plan = [&](const std::vector<int64_t> & key) {
            std::vector<int> ids, current, weights; // ids: streams
            for (size_t i = 0; i < key.size(); i += 5) {
                ids.push_back(key[i]); current.push_back(key[i+3]); weights.push_back(key[i+4]);
            }
            auto destination = pipeline_work_balanced_plan(ids, current, weights, pipeline_count);
            if (llama_n_rs_seq(ctx_tgt) > 0) {
                const int capacity = int(llama_pipeline_verify_ubatch(ctx_tgt));
                if (std::all_of(weights.begin(), weights.end(), [&](int weight) {
                        return weight <= capacity && weight <= int(llama_n_rs_seq(ctx_tgt)) + 1;
                    })) {
                    // First retain accepted row balancing, then reduce physical
                    // complete-tail graphs. Do not row-balance that result again.
                    server_wall_timer timer(pipeline_work_physical_us, pipeline_work_physical_calls);
                    bool converged = false;
                    auto physical = pipeline_mixed_physical_compact_plan(ids,
                            destination.empty() ? current : destination, weights, pipeline_count, capacity, &converged);
                    if (!converged) { ++pipeline_work_physical_fallbacks; }
                    if (!physical.empty()) {
                        destination = std::move(physical);
                        ++pipeline_work_physical_improvements;
                    }
                }
            }
            return destination == current ? std::vector<int>{} : destination;
        };
        if (!plan(membership).empty()) {
            // Keep the accepted complete-ticket/draft migration boundary.
            pipeline_drain();
            membership = pipeline_work_membership();
            if (membership.empty() || !heterogeneous(membership)) { return false; }
            const auto destination = plan(membership);
            for (size_t i = 0; i < destination.size(); ++i) {
                auto & slot = slots.at(membership[5*i + 1]);
                if (pipeline_lane_of(slot.id) == destination[i]) { continue; }
                if (!pipeline_move_slot(slot, destination[i])) { break; }
                ++pipeline_work_moves;
            }
        }
        // Placement runs once per membership/layout change, never per token.
        pipeline_work_plan_signature = pipeline_work_membership();
        return true;
    }

    void pipeline_refresh_work_rounds() {
        if (!pipeline_on) { return; }
        const auto key = pipeline_work_membership();
        if (key == pipeline_work_round_signature) { return; }
        pipeline_work_round_signature = key;
        pipeline_work_issued.assign(pipeline_count, 0);
        pipeline_work_speculative.assign(pipeline_count, false);
        int priority = 0;
        for (size_t i = 0; i < key.size(); i += 5) {
            priority = std::max(priority, slots.at(key[i+1]).task->params.priority);
        }
        bool ordinary = false, speculative = false;
        for (size_t i = 0; i < key.size(); i += 5) {
            // A background MTP must never throttle a traditional VIP. Only
            // highest-priority MTP lanes provide pacing credits; background
            // speculative cars are subject to those credits as well.
            if (key[i+4] > 1 && slots.at(key[i+1]).task->params.priority == priority) {
                pipeline_work_speculative[key[i+3]] = true; speculative = true;
            }
            else { ordinary = true; }
        }
        pipeline_work_mixed = ordinary && speculative;
    }

    bool pipeline_move_slot(server_slot & slot, int lane) {
        dft_join(__LINE__);
        GGML_ASSERT(slot.pipeline_owner < 0 && pipeline_fifo.empty());
        const int previous = pipeline_lane_of(slot.id);
        if (previous == lane) { return true; }
        slot.pipeline_lane = lane;
        SLT_INF(slot, "pipeline lane move %d -> %d, paused=%d, mtp=%d\n", previous, lane, slot.resident_paused, slot.can_speculate());
        return true;
    }

    void pipeline_compact_lanes() {
        // Non-unified KV splits a car at each gap in its KV streams. ids: streams in order, slot ids alongside.
        std::vector<int> slot_ids;
        auto membership = [&](std::vector<int> & ids, std::vector<int> & current, std::vector<int> & weights) {
            ids.clear(); current.clear(); weights.clear(); slot_ids.clear();
            std::vector<std::array<int, 4>> entries;
            for (const auto & slot : slots) {
                if (!slot.is_processing() || slot.resident_paused) { continue; }
                if (slot.state != SLOT_STATE_GENERATING) { return false; }
                entries.push_back({pipeline_stream_of(slot.id), slot.id, pipeline_lane_of(slot.id), pipeline_work_weight(slot)});
            }
            std::sort(entries.begin(), entries.end());
            for (const auto & e : entries) {
                ids.push_back(e[0]); slot_ids.push_back(e[1]); current.push_back(e[2]); weights.push_back(e[3]);
            }
            return true;
        };
        // lane k takes the k-th block in stream order: the shared draft batch merges lanes adjacent in issue order
        auto in_stream_order = [&](std::vector<int> lanes) {
            std::vector<int> map(pipeline_count, -1);
            int next = 0;
            for (const int lane : lanes) { if (map[lane] < 0) { map[lane] = next++; } }
            for (int lane = 0; lane < pipeline_count; ++lane) { if (map[lane] < 0) { map[lane] = next++; } }
            for (auto & lane : lanes) { lane = map[lane]; }
            return lanes;
        };
        auto plan_runs = [&](const std::vector<int> & ids, const std::vector<int> & current, const std::vector<int> & weights) {
            server_wall_timer plan_timer(pipeline_compact_plan_us, pipeline_compact_plan_calls);
            // Fewer stream runs imply fewer physical splits only when each
            // lane's complete verification rows fit one physical ubatch.
            const uint64_t capacity = llama_pipeline_verify_ubatch(ctx_tgt);
            const bool homogeneous = weights.empty() || std::all_of(weights.begin(), weights.end(),
                    [&](int weight) { return weight == weights.front(); });
            if (homogeneous && !weights.empty() &&
                    llama_n_rs_seq(ctx_tgt) > 0 && weights.front() <= int(capacity) &&
                    weights.front() <= int(llama_n_rs_seq(ctx_tgt)) + 1) {
                return pipeline_physical_compact_plan(ids, current, pipeline_count, weights.front(), int(capacity));
            }
            const uint64_t rows = weights.empty() ? 0 : uint64_t(weights.front()) *
                    ((ids.size() + pipeline_count - 1) / pipeline_count);
            if (!homogeneous || rows > capacity) {
                return pipeline_balanced_compact_plan(ids, current, pipeline_count);
            }
            return pipeline_balanced_stream_plan(ids, current, pipeline_count);
        };
        auto plan = [&](const std::vector<int> & ids, const std::vector<int> & current, const std::vector<int> & weights) {
            const auto runs = plan_runs(ids, current, weights);
            auto ordered = in_stream_order(runs.empty() ? current : runs);
            return ordered == current ? std::vector<int>{} : ordered;
        };
        std::vector<int> ids, current, weights;
        if (!membership(ids, current, weights)) { return; }
        if (ids == pipeline_compact_ids_cache && slot_ids == pipeline_compact_slots_cache &&
                current == pipeline_compact_lanes_cache && weights == pipeline_compact_weights_cache) {
            ++pipeline_compact_plan_cache_hits;
            return;
        }
        auto destination = plan(ids, current, weights);
        if (!destination.empty()) {
            pipeline_drain();
            // Settling a car may finish a request. Recheck before moving it.
            if (!membership(ids, current, weights)) { return; }
            destination = plan(ids, current, weights);
            for (size_t i = 0; i < destination.size(); ++i) {
                if (current[i] == destination[i]) { continue; }
                if (!pipeline_move_slot(slots.at(slot_ids[i]), destination[i])) {
                    // A deferred transfer must be eligible for another attempt.
                    pipeline_compact_ids_cache.clear();
                    pipeline_compact_lanes_cache.clear();
                    pipeline_compact_weights_cache.clear();
                    return;
                }
                current[i] = destination[i];
                ++pipeline_compact_moves;
            }
        }
        pipeline_compact_ids_cache = std::move(ids);
        pipeline_compact_slots_cache = std::move(slot_ids);
        pipeline_compact_lanes_cache = std::move(current);
        pipeline_compact_weights_cache = std::move(weights);
    }

    void pipeline_rebalance() {
        if (!pipeline_on) { return; }
        pipeline_defrag_streams();
        if (pipeline_rebalance_work()) { return; }
        auto loads = [&]() {
            std::vector<int> result(pipeline_count, 0);
            for (const auto & slot : slots) {
                if (slot.is_processing() && !slot.resident_paused) { ++result[pipeline_lane_of(slot.id)]; }
            }
            return result;
        };
        auto load = loads();
        if (*std::max_element(load.begin(), load.end()) - *std::min_element(load.begin(), load.end()) <= 1) {
            pipeline_compact_lanes();
            return;
        }
        // Move only at a complete ticket and draft boundary. Paused slots own no lane capacity.
        pipeline_drain();
        load = loads();
        while (true) {
            const int from = std::max_element(load.begin(), load.end()) - load.begin();
            const int to   = std::min_element(load.begin(), load.end()) - load.begin();
            if (load[from] - load[to] <= 1) { break; }
            server_slot * next = nullptr;
            for (auto & slot : slots) {
                if (slot.is_processing() && !slot.resident_paused && pipeline_lane_of(slot.id) == from &&
                        (!next || slot.prompt.n_tokens() < next->prompt.n_tokens())) { next = &slot; }
            }
            if (!next || !pipeline_move_slot(*next, to)) { break; }
            --load[from];
            ++load[to];
        }
        pipeline_compact_lanes();
    }

    // MTP prompt chunks may use prompt-only lanes (single shared draft driver)
    bool pipeline_mtp_prefill_lanes = false;

    bool pipeline_prefill_allowed() const {
        // Successive chunks of one sequence must use the same ordered streams.
        if (!params_base.pipe_fuse) { return false; }
        bool active = false;
        for (const auto & slot : slots) {
            if (!slot.is_processing() || slot.resident_paused) { continue; }
            if ((slot.can_speculate() && !pipeline_async_drafts) ||
                    slot.task->is_parent() || slot.task->is_child() ||
                    slot.task->tokens.has_media() || !slot.task->need_sampling() || !slot.lora.empty()) { return false; }
            active = true;
        }
        return active;
    }

    uint64_t checkpoint_skipped_nocache = 0;
    uint64_t pipeline_checkpoint_holds = 0;
    // true if the slot's next prompt chunk may create a tail checkpoint while one of its own
    // prompt chunks is still in flight (the capture would then need a full synchronize).
    bool pipeline_hold_for_checkpoint(const server_slot & slot) const {
        if (params_base.n_ctx_checkpoints <= 0 || !slot.task) { return false; }
        if (!slot.task->params.cache_prompt) { return false; }   // no checkpoint will be made
        const int64_t remaining = int64_t(slot.task->n_tokens()) - int64_t(slot.prompt.n_tokens());
        const int64_t n_ub = llama_n_ubatch(ctx_tgt);
        if (remaining > n_ub + 8) { return false; }   // tail checkpoints sit at 4 + n_ubatch and 4 tokens from the end
        for (const auto & lane : pipeline_lanes) {
            if (lane.in_flight && lane.prefill_slot == slot.id) { return true; }
        }
        return false;
    }

    int pipeline_next_prefill(int lane_id) {
        if (!pipeline_prefill_allowed()) { return -1; }
        if (pipeline_prefill_lanes > 0 && lane_id < pipeline_count) { return -1; }
        int in_flight = 0;
        for (const auto & lane : pipeline_lanes) {
            in_flight += lane.in_flight && lane.prefill_slot >= 0;
        }
        if (in_flight >= std::max(1, params_base.pipe_pf_inflight)) { return -1; }
        for (int pass = 0; pass < 2; ++pass)
        for (size_t i = 0; i < slots.size(); ++i) {
            const size_t index = (pipeline_prefill_next + i) % slots.size();
            const auto & slot = slots[index];
            if (!slot.resident_paused && slot.task &&
                    (slot.state == SLOT_STATE_STARTED || slot.state == SLOT_STATE_PROCESSING_PROMPT)) {
                if (pass == 0 && pipeline_hold_for_checkpoint(slot)) { ++pipeline_checkpoint_holds; continue; }
                pipeline_prefill_next = (index + 1) % slots.size();
                return slot.id;
            }
        }
        return -1;
    }

    bool pipeline_prefill_lane_issuable() const {
        if (!pipeline_prefill_allowed()) { return false; }
        int in_flight = 0;
        for (const auto & lane : pipeline_lanes) { in_flight += lane.in_flight && lane.prefill_slot >= 0; }
        if (in_flight >= std::max(1, params_base.pipe_pf_inflight)) { return false; }
        return std::any_of(slots.begin(), slots.end(), [this](const server_slot & slot) {
            return !slot.resident_paused && slot.task &&
                (slot.state == SLOT_STATE_STARTED || slot.state == SLOT_STATE_PROCESSING_PROMPT);
        });
    }

    bool pipeline_can_run() const {
        if (!pipeline_on) {
            return false;
        }
        bool any = false;
        const bool allow_prefill = pipeline_prefill_allowed();
        for (const auto & slot : slots) {
            if (!slot.is_processing() || slot.resident_paused) {
                continue;
            }
            if (allow_prefill && (slot.state == SLOT_STATE_STARTED || slot.state == SLOT_STATE_PROCESSING_PROMPT ||
                        (slot.state == SLOT_STATE_DONE_PROMPT && slot.pipeline_owner >= 0))) {
                any = true;
                continue;
            }
            if (slot.state != SLOT_STATE_GENERATING || !slot.task->need_sampling() ||
                    !slot.inp_embd.empty() || !slot.lora.empty() ||
                    slot.prompt.n_tokens() + std::max(1, slot.get_n_draft_max()) + 2 >= slot.n_ctx ||
                    slot.task->is_parent() || slot.task->is_child()) {
                return false;
            }
            any = true;
        }
        return any;
    }

    bool pipeline_issue(int lane_id) {
        server_wall_timer issue_timer(pipeline_issue_total_us, pipeline_issue_calls);
        auto & lane = pipeline_lanes[lane_id];
        GGML_ASSERT(!lane.in_flight);
        if (pipeline_work_mixed) {
            // Completion/cancellation can change membership within this tick.
            pipeline_refresh_work_rounds();
            if (pipeline_work_mixed && !pipeline_work_round_allowed(
                    pipeline_work_issued, pipeline_work_speculative, lane_id)) {
                ++pipeline_work_deferrals;
                return false;
            }
        }
        // A settled lane can be submitted again within this scheduling tick.
        // Recheck the budget for every car, including that immediate reissue.
        ensure_generation_capacity();
        pipeline_filter = lane_id;
        batch = {};
        batch.init(ctx_tgt, llama_n_batch(ctx_tgt), llama_model_n_embd_inp(model_tgt));
        if (lane.shared_draft_staged) {
            GGML_ASSERT(pipeline_ready_draft_batch && lane.shared_draft_ready);
            lane.shared_draft_staged = lane.shared_draft_ready = false;
            batch.slot_batched = lane.slot_batched;
        } else {
            prepare_decode_generations(lane.generating, lane.drafting);
            if (!lane.drafting.empty() && pipeline_async_drafts) {
                // Stage the lane; the shared draft driver batches every staged lane's drafts.
                GGML_ASSERT(!lane.generating.empty());
                lane.slot_batched = lane.generating.front();
                lane.shared_draft_staged = true;
                lane.shared_draft_ready = false;
                { uint64_t m = 0; for (auto * sl : lane.generating) { if (sl->id < 64) { m |= 1ull << sl->id; } }
                  pe(PE_STAGE, lane_id, 0, 0, (int) lane.drafting.size(), 0, m); }
                pipeline_filter = -1;
                return true;
            } else if (!lane.drafting.empty()) {
                draft_speculative_batch();
            }
        }
        finish_drafts(lane.drafting);
        pipeline_prefill_slot = pipeline_next_prefill(lane_id);
        lane.prefill_slot = -1;
        lane.prefill_task = -1;
        lane.prefill_output = -1;
        lane.slot_ids.clear();
        finish_pre_decode(lane.generating);
        lane.generating.clear();
        lane.drafting.clear();
        lane.slot_batched = nullptr;
        if (batch.size() > 0) {
            bool ordinary_rows = false, speculative_rows = false;
            for (const auto & token : batch.tokens) {
                if (slots.at(token.id_slot).can_speculate()) { speculative_rows = true; }
                else { ordinary_rows = true; }
            }
            if (ordinary_rows && speculative_rows) { ++pipeline_work_fused_cars; }
            const size_t prompt_payload = std::count_if(batch.tokens.begin(), batch.tokens.end(),
                    [](const server_batch::token & token) { return token.is_prompt; });
            pipeline_prompt_payload += prompt_payload;
            pipeline_decode_payload += batch.size() - prompt_payload;
            if (prompt_payload) { ++pipeline_prefill_cars; }
            else { ++pipeline_decode_cars; }
            if (prompt_payload && prompt_payload < size_t(batch.size())) { ++pipeline_mixed_cars; }
            batch.render(0, batch.size());
            if (pipeline_prefill_slot >= 0 && std::all_of(batch.tokens.begin(), batch.tokens.end(),
                    [](const server_batch::token & token) { return token.output; })) {
                // A one-token prompt tail uses the decode streams. Fence its earlier prompt cars first.
                llama_synchronize(ctx_tgt);
            }
            llama_set_decode_pipelane(ctx_tgt, lane_id);
            metrics_pre_decode();
            const auto issue_start = ggml_time_us();
            ++pipeline_nonempty_issues;
            const int pe_nprompt = (int) std::count_if(batch.tokens.begin(), batch.tokens.end(), [](const server_batch::token & token) { return token.is_prompt; });
            const uint64_t pe_slots = pe_mask(batch);
            int pe_nseq = 0; for (uint64_t m = pe_slots; m; m &= m - 1) { ++pe_nseq; }
            pe(PE_ISSUE_B, lane_id, batch.size(), pe_nprompt, pe_nseq, 0, pe_slots);
            const int ret = pipeline_test_fail_issue > 0 && pipeline_nonempty_issues == pipeline_test_fail_issue
                    ? -2 : llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get());
            pipeline_profile_issue_us += ggml_time_us() - issue_start;
            pe(PE_ISSUE_E, lane_id, batch.size(), pe_nprompt, pe_nseq, ret == 0 ? llama_decode_ticket_last(ctx_tgt) : 0, pe_slots);
            if (ret != 0) {
                pipeline_filter = -1;
                if (ret == 1 || ret == -2) {
                    // nothing of this car is retained by the lane; its slots are terminated by the caller
                    std::vector<int> ids;
                    for (const auto & token : batch.tokens) {
                        if (std::find(ids.begin(), ids.end(), token.id_slot) == ids.end()) { ids.push_back(token.id_slot); }
                    }
                    lane.slot_ids.clear();
                    lane.prefill_slot = lane.prefill_output = -1;
                    lane.prefill_task = -1;
                    lane.ticket = 0;
                    pipeline_prefill_slot = -1;
                    llama_decode_ticket_select(ctx_tgt, 0);
                    llama_set_decode_pipelane(ctx_tgt, -1);
                    throw pipeline_car_failure(ret, std::move(ids));
                }
                throw std::runtime_error("pipeline decode failed: " + std::to_string(ret));
            }
            lane.ticket = llama_decode_ticket_last(ctx_tgt);
            if (!llama_decode_ticket_retain(ctx_tgt, lane.ticket)) {
                throw std::runtime_error("cannot retain pipeline output ticket");
            }
            for (const auto & token : batch.tokens) {
                auto & slot = slots[token.id_slot];
                if (std::find(lane.slot_ids.begin(), lane.slot_ids.end(), slot.id) == lane.slot_ids.end()) {
                    lane.slot_ids.push_back(slot.id);
                }
                if (token.is_prompt) {
                    lane.prefill_slot = slot.id;
                    lane.prefill_task = slot.task->id;
                    if (token.output) { lane.prefill_output = slot.i_batch; }
                    if (lane.prefill_output < 0) { continue; }
                }
                auto & owner = slot.pipeline_owner;
                GGML_ASSERT(owner < 0 || owner == lane_id);
                owner = lane_id;
            }
            lane.batch = std::move(batch);
            lane.in_flight = true;
            lane.issued_us = ggml_time_us();
            if (pipeline_work_mixed) { ++pipeline_work_issued[lane_id]; }
            pipeline_fifo.push_back(lane_id);
        }
        pipeline_filter = -1;
        pipeline_prefill_slot = -1;
        return true;
    }

    int pipeline_settle(int ready_lane = -1) {
        GGML_ASSERT(!pipeline_fifo.empty());
        const int lane_id = ready_lane >= 0 ? ready_lane : pipeline_fifo.front();
        const auto position = std::find(pipeline_fifo.begin(), pipeline_fifo.end(), lane_id);
        GGML_ASSERT(position != pipeline_fifo.end());
        auto & lane = pipeline_lanes[lane_id];
        pe(PE_SETTLE_B, lane_id, lane.batch.size(), 0, 0, lane.ticket, pe_mask(lane.batch));
        const auto wait_start = ggml_time_us();
        if (!llama_decode_ticket_select(ctx_tgt, lane.ticket)) {
            throw std::runtime_error("pipeline output ticket is unavailable");
        }
        pipeline_profile_wait_us += ggml_time_us() - wait_start;
        pe(PE_WAIT_E, lane_id, 0, 0, 0, lane.ticket);
        const auto process_start = ggml_time_us();
        if (position != pipeline_fifo.begin()) { ++pipeline_ready_reorders; }
        pipeline_fifo.erase(position);
        lane.in_flight = false;
        pipeline_filter = lane_id;
        pipeline_settling = true;
        batch = std::move(lane.batch);
        try {
            if (lane.prefill_slot >= 0) {
                const auto & slot = slots.at(lane.prefill_slot);
                if (!slot.task || slot.task->id != lane.prefill_task) {
                    throw std::runtime_error("prefill ticket task ownership changed");
                }
            }
            metrics_post_decode(0, batch.size(), true);
            std::vector<int> degrade_ids;
            pe(PE_CATCHUP_B, lane_id, batch.size());
            bool pe_catchup_ok = true;
            if (dft_exec && dft_route) { dft_submit_catchup(batch.view); }
            else { pe_catchup_ok = !spec || process_speculative_batch(batch.view); }
            pe(PE_CATCHUP_E, lane_id, batch.size());
            if (!pe_catchup_ok) {
                // target results of this car stay valid; drop drafting for its requests after verification
                for (const auto & token : batch.tokens) {
                    if (std::find(degrade_ids.begin(), degrade_ids.end(), token.id_slot) == degrade_ids.end()) { degrade_ids.push_back(token.id_slot); }
                }
            }
            post_decode(batch.size(), 0);
            pe(PE_POST_E, lane_id, batch.size());
            if (!degrade_ids.empty()) { pipeline_degrade_drafts(degrade_ids); }
        } catch (...) {
            llama_decode_ticket_select(ctx_tgt, 0);
            llama_decode_ticket_release(ctx_tgt, lane.ticket);
            lane.ticket = 0;
            pipeline_settling = false;
            pipeline_filter = -1;
            throw;
        }
        llama_decode_ticket_select(ctx_tgt, 0);
        if (!llama_decode_ticket_release(ctx_tgt, lane.ticket)) {
            throw std::runtime_error("pipeline output ticket retention lost");
        }
        lane.ticket = 0;
        for (const auto & token : batch.tokens) {
            if (slots[token.id_slot].pipeline_owner == lane_id) { slots[token.id_slot].pipeline_owner = -1; }
        }
        pipeline_settling = false;
        pipeline_filter = -1;
        pipeline_profile_process_us += ggml_time_us() - process_start;
        pe(PE_SETTLE_E, lane_id);
        ++pipeline_profile_count;
        return lane_id;
    }

    void pipeline_update_draft_ready() {
        for (auto & lane : pipeline_lanes) {
            if (lane.shared_draft_staged && !lane.shared_draft_ready) {
                lane.shared_draft_ready = std::all_of(lane.drafting.begin(), lane.drafting.end(),
                        [](const server_slot * slot) {
                            return !common_speculative_get_draft_params(slot->spec, slot->id).drafting;
                        });
            }
        }
    }

    // a memory failure confined to one target car
    struct pipeline_car_failure : std::runtime_error {
        int ret;
        std::vector<int> slot_ids;
        pipeline_car_failure(int r, std::vector<int> ids)
            : std::runtime_error("pipeline decode failed: " + std::to_string(r)), ret(r), slot_ids(std::move(ids)) {}
    };
    const int64_t pipeline_test_fail_issue = server_test_hook("LLAMA_PIPE_TEST_FAIL_ISSUE");
    int64_t pipeline_nonempty_issues = 0;
    uint64_t pipeline_oom_isolations = 0, pipeline_oom_terminated = 0;

    void pipeline_flush_ready_drafts() {
        if (!pipeline_ready_draft_batch) { return; }
        if (dft_exec) { dft_poll(); return; }
        size_t n_lanes = 0;
        std::set<int> sequences;
        for (const auto & lane : pipeline_lanes) {
            if (!lane.shared_draft_staged || lane.shared_draft_ready) { continue; }
            ++n_lanes;
            for (const auto * slot : lane.drafting) {
                GGML_ASSERT(slot->spec == spec.get() && slot->ctx_dft == ctx_dft);
                if (common_speculative_get_draft_params(slot->spec, slot->id).drafting) {
                    GGML_ASSERT(sequences.insert(slot->id).second);
                }
            }
        }
        if (!n_lanes) { return; }
        // No asynchronous worker shares this context. Nothing waits for a lane
        // that has not completed target verification, or for a batch-size goal.
        const auto start = ggml_time_us();
        try {
            // QA hook: fail the N-th draft step as if its memory commit had failed
            // (only at the start of a step: with every draft empty no step is active, matching the driver's reset state)
            if (pipeline_test_fail_draftstep > 0 && std::all_of(sequences.begin(), sequences.end(), [this](int id) {
                        auto & dp = common_speculative_get_draft_params(spec.get(), id); return !dp.result || dp.result->empty(); }) &&
                    ++pipeline_draftstep_calls == pipeline_test_fail_draftstep) {
                throw std::runtime_error("test draft step failure");
            }
            common_speculative_draft_step(spec.get());
        } catch (const std::exception & e) {
            // The step state was reset by the speculative driver; partial drafts of this round are discarded so
            // every lane verifies with an empty draft (plain target decode). The longest-context drafting
            // request gives up speculation to return its draft KV pages; others draft again next round.
            int victim = -1;
            for (const int id : sequences) {
                auto & dp = common_speculative_get_draft_params(spec.get(), id);
                dp.drafting = false;
                if (dp.result) { dp.result->clear(); }
                if (dp.result_q) { dp.result_q->clear(); }
                if (victim < 0 || slots.at(id).prompt.tokens.size() > slots.at(victim).prompt.tokens.size()) { victim = id; }
            }
            ++pipeline_draft_step_failures;
            SRV_WRN("draft step failed (%s): %zu drafts dropped, degrading slot %d\n", e.what(), sequences.size(), victim);
            if (victim >= 0) { pipeline_degrade_drafts({victim}); }
            pipeline_update_draft_ready();
            return;
        }
        pipeline_draft_batch_us += ggml_time_us() - start;
        ++pipeline_draft_batch_calls;
        pipeline_draft_batch_lanes += n_lanes;
        pipeline_draft_batch_sequences += sequences.size();
        pipeline_draft_batch_max_sequences = std::max<uint64_t>(pipeline_draft_batch_max_sequences, sequences.size());
        ++pipeline_draft_batch_histogram[sequences.size()];
        for (auto & lane : pipeline_lanes) {
            if (lane.shared_draft_staged && !lane.shared_draft_ready) {
                lane.shared_draft_ready = std::all_of(lane.drafting.begin(), lane.drafting.end(),
                        [](const server_slot * slot) {
                            return !common_speculative_get_draft_params(slot->spec, slot->id).drafting;
                        });
            }
        }
    }

    void pipeline_drain() {
        dft_join(__LINE__);
        do { pipeline_flush_ready_drafts(); } while (std::any_of(pipeline_lanes.begin(), pipeline_lanes.end(),
                [](const pipeline_lane & lane) { return lane.shared_draft_staged && !lane.shared_draft_ready; }));
        server_wall_timer drain_timer(pipeline_drain_us, pipeline_drain_calls);
        if (!pipeline_fifo.empty()) { ++pipeline_drain_nonempty; }
        while (!pipeline_fifo.empty()) {
            pipeline_settle();
        }
        for (int i = 0; i < (int) pipeline_lanes.size(); ++i) {
            auto & lane = pipeline_lanes[i];
            if (lane.shared_draft_staged) {
                GGML_ASSERT(lane.shared_draft_ready);
                lane.shared_draft_staged = lane.shared_draft_ready = false;
                pipeline_filter = i;
                // Preserve completed drafts for the next conventional round, but
                // do not append unverified tokens while entering a pause boundary.
                finish_drafts(lane.drafting);
                lane.generating.clear();
                lane.drafting.clear();
                lane.slot_batched = nullptr;
            }
        }
        pipeline_filter = -1;
        llama_set_decode_pipelane(ctx_tgt, -1);
    }

    void pipeline_step_ready_draft_batch() {
        bool progressed = false;
        // Retire only tickets already ready. Prompt tickets retain FIFO order;
        // pure generation tickets may retire out of order as in the old path.
        const bool decode_only = std::none_of(slots.begin(), slots.end(), [](const server_slot & slot) {
            return slot.is_processing() && !slot.resident_paused && slot.state != SLOT_STATE_GENERATING;
        }) && std::all_of(pipeline_fifo.begin(), pipeline_fifo.end(), [&](int id) {
            return pipeline_lanes[id].prefill_slot < 0;
        });
        for (int count = 0; count < (int) pipeline_lanes.size() && !pipeline_fifo.empty(); ++count) {
            int ready = -1;
            bool seen_prompt = false;
            for (const int id : pipeline_fifo) {
                const bool prompt = pipeline_lanes[id].prefill_slot >= 0;
                if (pipeline_mtp_prefill_lanes && !decode_only) {
                    if ((!prompt || !seen_prompt) && llama_decode_ticket_ready(ctx_tgt, pipeline_lanes[id].ticket)) {
                        if (!prompt && seen_prompt) { ++pipeline_prompt_overtakes; }
                        ready = id; break;
                    }
                    seen_prompt |= prompt;
                    continue;
                }
                if (llama_decode_ticket_ready(ctx_tgt, pipeline_lanes[id].ticket)) { ready = id; break; }
                if (!decode_only) { break; }
            }
            if (ready < 0) { break; }
            pipeline_settle(ready);
            progressed = true;
        }
        for (int i = 0; i < pipeline_count; ++i) {
            auto & lane = pipeline_lanes[i];
            if (lane.in_flight || lane.shared_draft_staged) { continue; }
            const bool runnable = std::any_of(slots.begin(), slots.end(), [&](const server_slot & slot) {
                return slot.state == SLOT_STATE_GENERATING && !slot.resident_paused &&
                        slot.pipeline_owner < 0 && pipeline_lane_of(slot.id) == i;
            });
            const bool prefill = pipeline_prefill_allowed() && std::any_of(slots.begin(), slots.end(), [](const server_slot & slot) {
                return !slot.resident_paused && (slot.state == SLOT_STATE_STARTED || slot.state == SLOT_STATE_PROCESSING_PROMPT);
            });
            if (runnable || (prefill && pipeline_prefill_lanes == 0)) { progressed |= pipeline_issue(i); }
        }
        for (int i = pipeline_count; i < (int) pipeline_lanes.size(); ++i) {
            // prompt-only lanes, issued only when a chunk can actually be assigned
            auto & lane = pipeline_lanes[i];
            if (!lane.in_flight && pipeline_prefill_lane_issuable()) { progressed |= pipeline_issue(i); }
        }
        progressed |= std::any_of(pipeline_lanes.begin(), pipeline_lanes.end(),
                [](const pipeline_lane & lane) { return lane.shared_draft_staged && !lane.shared_draft_ready; });
        pipeline_flush_ready_drafts();
        for (int i = 0; i < pipeline_count; ++i) {
            if (pipeline_lanes[i].shared_draft_staged && pipeline_lanes[i].shared_draft_ready) {
                progressed |= pipeline_issue(i);
            }
        }
        // Backpressure only when no work can be issued now. Never wait for an
        // unfinished target ticket merely to enlarge a ready draft batch.
        if (!progressed && !pipeline_fifo.empty()) {
            pipeline_settle(); progressed = true;
        }
        if (!progressed) { pe(PE_SLEEP); std::this_thread::sleep_for(std::chrono::microseconds(100)); }
    }

    void pipeline_step() {
        if (pipeline_ready_draft_batch && std::any_of(slots.begin(), slots.end(), [](const server_slot & slot) {
                return slot.is_processing() && !slot.resident_paused && slot.can_speculate();
            })) {
            pipeline_step_ready_draft_batch();
            return;
        }
        bool progressed = false;
        for (int i = 0; i < (int) pipeline_lanes.size(); ++i) {
            auto & lane = pipeline_lanes[i];
            if (i >= pipeline_count) {
                // Prompt-only lane: issue only when a chunk can actually be assigned.
                if (!lane.in_flight && pipeline_prefill_lane_issuable()) { progressed |= pipeline_issue(i); }
                continue;
            }
            if (!lane.in_flight) {
                const bool runnable = std::any_of(slots.begin(), slots.end(), [&](const server_slot & slot) {
                    return slot.state == SLOT_STATE_GENERATING && !slot.resident_paused &&
                            slot.pipeline_owner < 0 && pipeline_lane_of(slot.id) == i;
                });
                const bool prefill = pipeline_prefill_allowed() && std::any_of(slots.begin(), slots.end(), [](const server_slot & slot) {
                    return !slot.resident_paused && (slot.state == SLOT_STATE_STARTED || slot.state == SLOT_STATE_PROCESSING_PROMPT);
                });
                if (runnable || (prefill && pipeline_prefill_lanes == 0)) {
                    progressed |= pipeline_issue(i);
                }
            }
        }
        if (!pipeline_fifo.empty()) {
            int ready_lane = -1;
            // Decode tickets own their outputs and touch disjoint sequences.
            // A completed lane can therefore commit and start its next draft
            // without waiting for another lane's GPU dependency. Prompt cars
            // retain FIFO order, including their recurrent/cache boundaries.
            const bool decode_only = std::none_of(slots.begin(), slots.end(), [](const server_slot & slot) {
                return slot.is_processing() && !slot.resident_paused && slot.state != SLOT_STATE_GENERATING;
            }) && std::all_of(pipeline_fifo.begin(), pipeline_fifo.end(), [&](int lane) {
                return pipeline_lanes[lane].prefill_slot < 0;
            });
            bool seen_prompt = false;
            for (const int lane_id : pipeline_fifo) {
                auto & lane = pipeline_lanes[lane_id];
                if (pipeline_prefill_lanes > 0 && !decode_only) {
                    // Prompt tickets retire strictly in FIFO order among themselves (their
                    // recurrent/cache boundaries); generation tickets touch disjoint
                    // sequences and may retire as soon as they are ready.
                    const bool prompt = lane.prefill_slot >= 0;
                    const bool held = prompt && pipeline_test_prompt_hold_us > 0 &&
                            ggml_time_us() - lane.issued_us < pipeline_test_prompt_hold_us;
                    if ((!prompt || !seen_prompt) && !held && llama_decode_ticket_ready(ctx_tgt, lane.ticket)) {
                        if (!prompt && seen_prompt) { ++pipeline_prompt_overtakes; }
                        ready_lane = lane_id;
                        break;
                    }
                    seen_prompt |= prompt;
                    continue;
                }
                if (llama_decode_ticket_ready(ctx_tgt, lane.ticket)) {
                    ready_lane = lane_id;
                    break;
                }
                if (!decode_only) { break; }
            }
            if (pipeline_prefill_lanes > 0 && !decode_only) {
                // Never block on the FIFO head here: a long prompt car would hold back
                // generation tickets that finish earlier. Poll instead.
                if (ready_lane >= 0) {
                    const int lane = pipeline_settle(ready_lane);
                    if (lane < pipeline_count || pipeline_prefill_lane_issuable()) { pipeline_issue(lane); }
                    progressed = true;
                }
            } else {
                const int lane = pipeline_settle(ready_lane);
                pipeline_issue(lane);
                progressed = true;
            }
        }
        if (!progressed) { std::this_thread::sleep_for(std::chrono::microseconds(100)); }
    }


    llama_model   * model_dft = nullptr;
    llama_context * ctx_dft   = nullptr;

    common_speculative_init_result_ptr spec_init;

    common_context_seq_rm_type ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_context_seq_rm_type ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    common_speculative_ptr spec;

    bool add_bos_token = true;

    int32_t n_ctx; // total context for all clients / slots

    // set to llama_model_n_swa(model)
    // if swa_full is enabled, this is set to 0 to simulate a non-SWA model
    int32_t n_swa;

    // slots / clients
    std::vector<server_slot> slots;

    int trace = 0;        // env: LLAMA_TRACE
    int slots_debug = 0;  // env: LLAMA_SERVER_SLOTS_DEBUG
    int slots_n_diff = 0; // env: LLAMA_SERVER_SLOTS_N_DIFF

    int n_empty_consecutive = 0;

    std::unique_ptr<server_prompt_cache> prompt_cache;

    struct system_prefix_state {
        server_prompt prompt;
        server_prompt_data data;
        uint32_t hits = 0;

        size_t size() const {
            return sizeof(*this) + 2 * sizeof(void *) + prompt.tokens.size() * sizeof(llama_token) +
                    prompt.speculative_state.capacity() + data.main.capacity() + data.drft.capacity();
        }
    };

    std::list<system_prefix_state> system_prefixes;
    size_t system_cache_limit = 0;
    size_t system_cache_bytes = 0;

    uint64_t system_cache_inplace = 0, system_cache_misses = 0, system_cache_waits = 0;
    uint64_t system_cache_hits = 0, system_cache_hit_tokens = 0, system_cache_restore_failures = 0;
    uint64_t system_cache_restore_total_us = 0, system_cache_publish_total_us = 0;
    uint64_t system_cache_publishes = 0, system_cache_publish_failures = 0, system_cache_evictions = 0;

    int system_prefix_length(const server_task & task) const {
        if (!system_cache_limit || !task.params.cache_prompt || task.type != SERVER_TASK_TYPE_COMPLETION ||
                task.is_parent() || task.is_child() || !task.need_sampling() || task.tokens.has_media() ||
                !params_base.lora_adapters.empty() || !task.params.lora.empty() || llama_model_n_swa(model_tgt) > 0) { return 0; }
        bool has_system = false;
        for (const auto & span : task.params.message_spans.spans) {
            if (span.role == COMMON_CHAT_ROLE_SYSTEM) { has_system = true; }
            if (span.role == COMMON_CHAT_ROLE_USER) {
                return has_system && span.pos > 0 && span.pos < task.tokens.size() ? int(span.pos) : 0;
            }
            if (span.role == COMMON_CHAT_ROLE_ASSISTANT || span.role == COMMON_CHAT_ROLE_TOOL) { return 0; }
        }
        return 0;
    }

    bool system_prefix_matches(const server_tokens & cached, const server_task & task, int boundary) const {
        return cached.size() == size_t(boundary) && cached.get_common_prefix(task.tokens) == size_t(boundary);
    }

    system_prefix_state * find_system_prefix(const server_task & task, int boundary) {
        for (auto & entry : system_prefixes) {
            if ((!task.params.mtp || entry.prompt.mtp_ready) && system_prefix_matches(entry.prompt.tokens, task, boundary)) {
                return &entry;
            }
        }
        return nullptr;
    }

    void wake_system_waiters() {
        for (int i = 0; i < params_base.n_parallel; ++i) { queue_tasks.pop_deferred_task(-1); }
    }

    bool wait_system_prefix(const server_task & task) {
        const int boundary = system_prefix_length(task);
        if (!boundary || find_system_prefix(task, boundary)) { return false; }
        for (const auto & producer : slots) {
            if (!producer.is_processing() || producer.resident_paused || !producer.task ||
                    producer.task->params.priority < task.params.priority ||
                    (task.params.mtp && !producer.task->params.mtp) ||
                    producer.prompt.n_tokens() > boundary || system_prefix_length(*producer.task) != boundary) { continue; }
            if (producer.task->tokens.get_common_prefix(task.tokens) >= size_t(boundary)) {
                ++system_cache_waits;
                SRV_DBG("system cache: task %" PRId64 " waits for prefix producer %" PRId64 "\n", task.id, producer.task->id);
                return true;
            }
        }
        return false;
    }

    void load_system_prefix(server_slot & slot) {
        dft_join(__LINE__);
        slot.system_cache_prefix_tokens = system_prefix_length(*slot.task);
        const int boundary = slot.system_cache_prefix_tokens;
        if (!boundary) { return; }
        const auto common = slot.prompt.tokens.get_common_prefix(slot.task->tokens);
        const auto next = slot.prompt.tokens.pos_next(common);
        const auto first = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
        const auto last = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);
        // Keep a complete in-place continuation only when no rewind is needed.
        // A longer token prefix alone cannot prove recurrent state is reusable:
        // its checkpoint may cut a canonical prefill chunk at another position.
        if (common >= size_t(boundary) && common == slot.prompt.tokens.size() &&
                first >= 0 && first < next && last == next - 1 &&
                (!slot.task->params.mtp || slot.prompt.mtp_ready)) { ++system_cache_inplace; return; }
        if (boundary >= slot.n_ctx || slot.task->n_tokens() > slot.n_ctx) { return; }
        auto * entry = find_system_prefix(*slot.task, boundary);
        if (!entry) {
            ++system_cache_misses;
            // A cold/evicted SYSTEM key must use the same canonical prefill
            // path as a fresh slot, rather than rewind an unrelated user tail.
            slot.prompt_clear();
            return;
        }
        if (params_base.kv_dynamic) {
            // A host snapshot restores the whole prefix at once. When the
            // shared device pool cannot admit it plus normal scheduling room,
            // let ordinary chunked prefill perform the existing spill policy.
            const uint64_t needed = uint64_t(boundary) + generation_reserve();
            while (resident_tokens(slot.id) + needed > uint64_t(n_ctx) && try_clear_idle_slots()) {}
            if (resident_tokens(slot.id) + needed > uint64_t(n_ctx)) {
                SLT_INF(slot, "%s\n", "system cache deferred: insufficient shared KV room; use chunked prefill");
                slot.prompt_clear();
                return;
            }
        }
        const auto started = ggml_time_us();
        try {
            slot.prompt_clear();
            if (llama_state_seq_set_data_ext(ctx_tgt, entry->data.main.data(), entry->data.main.size(), slot.id, 0) != entry->data.main.size()) {
                throw std::runtime_error("target prefix restore failed");
            }
            if (slot.task->params.mtp) {
                if (!slot.ctx_dft || entry->data.drft.empty() ||
                        llama_state_seq_set_data_ext(slot.ctx_dft, entry->data.drft.data(), entry->data.drft.size(), slot.id, 0) != entry->data.drft.size()) {
                    throw std::runtime_error("draft prefix restore failed");
                }
                common_speculative_set_state(slot.spec, slot.id, entry->prompt.speculative_state);
            }
            if (slot.ckpt_barrier) { slot.ckpt_barrier(slot.id); }
            slot.prompt = entry->prompt.clone();
            if (!slot.task->params.mtp) {
                slot.prompt.mtp_ready = false;
                slot.prompt.speculative_state.clear();
            }
            entry->hits = std::min<uint32_t>(1000000, entry->hits + 1);
            slot.system_cache_restored_tokens = boundary;
            slot.system_cache_restore_us = ggml_time_us() - started;
            ++system_cache_hits;
            system_cache_hit_tokens += boundary;
            SLT_INF(slot, "system cache hit: tokens=%d, restore_ms=%.3f, entries=%zu, bytes=%zu\n",
                    boundary, (ggml_time_us() - started) / 1000.0, system_prefixes.size(), system_cache_bytes);
        } catch (const std::exception & e) {
            ++system_cache_restore_failures;
            slot.prompt_clear();
            SLT_WRN(slot, "system cache fallback: %s\n", e.what());
        }
        system_cache_restore_total_us += ggml_time_us() - started;
    }

    void publish_system_prefix(server_slot & slot, int boundary) {
        if (!boundary || slot.prompt.n_tokens() != boundary || find_system_prefix(*slot.task, boundary)) { return; }
        dft_join(__LINE__);
        const auto started = ggml_time_us();
        try {
            // The first user car has not been submitted yet. All earlier cars
            // for this sequence end at or before this boundary.
            const auto last = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);
            if (last != boundary - 1) { throw std::runtime_error("prefix device position mismatch"); }
            system_prefix_state entry;
            const bool mtp = slot.task->params.mtp;
            const size_t main_size = llama_state_seq_get_size_ext(ctx_tgt, slot.id, 0);
            const size_t draft_size = mtp ? llama_state_seq_get_size_ext(slot.ctx_dft, slot.id, 0) : 0;
            const size_t rows = size_t(llama_n_batch(ctx_tgt)) + 1;
            const size_t width = size_t(llama_model_n_embd_inp(model_tgt));
            if (mtp && (width > system_cache_limit / sizeof(float) / rows)) {
                throw std::runtime_error("draft prefix temporary exceeds host cache budget");
            }
            const size_t spec_bound = mtp ? rows * width * sizeof(float) + 32 : 0;
            size_t needed = sizeof(entry) + 2 * sizeof(void *);
            // Reserve temporary key and speculative buffers before allocating.
            for (const auto component : {size_t(boundary) * sizeof(llama_token) * 2, spec_bound, main_size, draft_size}) {
                if (component > system_cache_limit || needed > system_cache_limit - component) {
                    throw std::runtime_error("prefix exceeds host cache budget");
                }
                needed += component;
            }
            while (system_cache_bytes > system_cache_limit - needed) {
                auto victim = system_prefixes.begin();
                for (auto it = system_prefixes.begin(); it != system_prefixes.end(); ++it) {
                    const long double value = static_cast<long double>(it->hits + 1) / it->size();
                    const long double best = static_cast<long double>(victim->hits + 1) / victim->size();
                    if (value < best) { victim = it; }
                }
                SRV_INF("system cache evict: tokens=%d, hits=%u, bytes=%zu\n", victim->prompt.n_tokens(), victim->hits, victim->size());
                ++system_cache_evictions;
                system_cache_bytes -= victim->size();
                system_prefixes.erase(victim);
            }
            if (mtp && (!common_speculative_get_state(slot.spec, slot.id, entry.prompt.speculative_state) ||
                    entry.prompt.speculative_state.capacity() > spec_bound)) {
                throw std::runtime_error("draft prefix state unavailable or exceeds reservation");
            }
            llama_tokens tokens;
            tokens.reserve(boundary);
            for (int i = 0; i < boundary; ++i) { tokens.push_back(slot.task->tokens[i]); }
            entry.prompt.tokens = server_tokens(tokens, false);
            entry.prompt.mtp_ready = mtp;
            entry.data.main.resize(main_size);
            entry.data.drft.resize(draft_size);
            if (!main_size || llama_state_seq_get_data_ext(ctx_tgt, entry.data.main.data(), main_size, slot.id, 0) != main_size ||
                    (mtp && llama_state_seq_get_data_ext(slot.ctx_dft, entry.data.drft.data(), draft_size, slot.id, 0) != draft_size)) {
                throw std::runtime_error("prefix snapshot failed");
            }
            for (auto it = system_prefixes.begin(); it != system_prefixes.end();) {
                if (system_prefix_matches(it->prompt.tokens, *slot.task, boundary)) {
                    entry.hits = std::max(entry.hits, it->hits);
                    system_cache_bytes -= it->size();
                    it = system_prefixes.erase(it);
                } else { ++it; }
            }
            const size_t entry_size = entry.size();
            system_prefixes.push_back(std::move(entry));
            system_cache_bytes += entry_size;
            ++system_cache_publishes;
            SRV_INF("system cache publish: tokens=%d, mtp=%d, snapshot_ms=%.3f, entries=%zu, bytes=%zu\n",
                    boundary, mtp, (ggml_time_us() - started) / 1000.0, system_prefixes.size(), system_cache_bytes);
        } catch (const std::exception & e) {
            ++system_cache_publish_failures;
            SLT_WRN(slot, "system cache save skipped: %s\n", e.what());
        }
        system_cache_publish_total_us += ggml_time_us() - started;
        // A failed or over-budget snapshot must not strand deferred followers.
        wake_system_waiters();
    }

    server_metrics metrics;

    // queued prompt stats - llama_decode() is async, so the timing is only valid after a sync
    // note: kept out of server_metrics, which is copied as-is into the task result
    int64_t  t_decode_start  = 0; // start of the last submitted decode
    int64_t  t_prompt_start  = 0; // start of the oldest queued prompt decode
    uint64_t n_prompt_queued = 0;

    json json_ui_settings = json::object();

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    std::string model_name; // name of the loaded model, to be used by API
    std::set<std::string> model_aliases; // additional names for the model
    std::set<std::string> model_tags;    // informational tags

    bool sleeping = false;

    int64_t t_last_load_progress_ms = 0;

    void destroy() {
        if (dft_exec) { dft_exec->join(); dft_exec.reset(); }
        system_prefixes.clear();
        system_cache_bytes = 0;
        if (ctx_tgt) { llama_synchronize(ctx_tgt); }
        GGML_ASSERT(ctx_tgt || ckpt_async_items.empty());
        while (!ckpt_async_items.empty()) {
            ckpt_async_settle(ckpt_async_items.back().ckpt);
        }
        for (auto & buffer : ckpt_async_bufs) {
            ggml_backend_buffer_free(buffer.buf);
        }
        ckpt_async_bufs.clear();
        if (checkpoint_pinned) {
            ggml_backend_buffer_free(checkpoint_pinned);
            checkpoint_pinned = nullptr;
        }
        ckpt_recycle_pool.clear();
        pipeline_fifo.clear();
        pipeline_lanes.clear();
        pipeline_ready_draft_batch = false;
        suspended_requests.clear();
        spec.reset();
        spec_init.reset();

        ctx_dft   = nullptr;
        model_dft = nullptr;

        llama_init.reset();

        ctx_tgt = nullptr;
        model_tgt = nullptr;

        mtmd_free(mctx);
        mctx = nullptr;
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            if (callback_state) {
                callback_state(SERVER_STATE_SLEEPING, {});
                // note: for sleeping == false, event is emitted by load_model()
            }
            SRV_INF("%s", "server is entering sleeping state\n");
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    struct load_progress_data {
        server_context_impl * ctx;
        std::string stage;
        std::vector<std::string> stages;
        int64_t t_last_load_progress_ms = 0;
        load_progress_data(server_context_impl * ctx, const std::string & stage) : ctx(ctx), stage(stage) {}
    };
    static bool load_progress_callback(float progress, void * user_data) {
        auto * d = static_cast<load_progress_data *>(user_data);
        GGML_ASSERT(d);
        // always emit the first and final sample; throttle the rest to one per 200ms
        {
            auto & t_last = d->t_last_load_progress_ms;
            const int64_t t_now = ggml_time_ms();
            const bool first = t_last == 0;
            const bool done  = progress >= 1.0f;
            const bool throttled = !first && !done && (t_now - t_last) < 200;
            if (throttled) {
                return true;
            }
            t_last = t_now;
        }
        if (d->ctx->callback_state) {
            d->ctx->callback_state(SERVER_STATE_LOADING, {
                {"stages", d->stages},
                {"current", d->stage},
                {"value", progress},
            });
        }
        return true;
    }

    // load the model and initialize llama_context
    // this may also be called to resume from sleeping state
    bool load_model(common_params & params) {
        system_prefixes.clear();
        system_cache_bytes = 0;
        system_cache_limit = 0;
        if (params.system_cache_mib > 0) {
            system_cache_limit = size_t(params.system_cache_mib) * 1024 * 1024;
            // Cross-slot restores require independent sequence streams. Unified
            // attention changes the physical row layout on restore and does not
            // preserve exact cold-prefill output on all backends.
            if (params.kv_unified) {
                SRV_WRN("%s\n", "system cache disabled: unified KV has no exact cross-slot restore guarantee");
                system_cache_limit = 0;
            }
            SRV_INF("system cache: host budget=%zu MiB, no expiry, LFU per byte\n", system_cache_limit / 1024 / 1024);
        }
        if (params.kv_spill && !params.kv_dynamic && !params.kv_unified) {
            SRV_WRN("%s\n", "kv spill disabled: requires --kv-unified or --kv-dynamic (a shared KV pool)");
        }
        load_progress_data load_progress_text  (this, "text_model");
        load_progress_data load_progress_mmproj(this, "mmproj_model");
        load_progress_data load_progress_spec  (this, "spec_model");

        const bool is_resume = sleeping;

        params_base = params;
        const auto output_limits = server_output_limits(params_base);
        params_base.n_outputs_max = output_limits.total;
        params_base.n_outputs_max_per_seq = output_limits.per_seq;

        const bool has_mmproj = !params.mmproj.path.empty();
        const bool has_draft = params.speculative.has_dft();
        const bool spec_mtp = std::find(params_base.speculative.types.begin(),
                                        params_base.speculative.types.end(),
                                        COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end();
        const bool has_spec = has_draft || spec_mtp;

        if (callback_state) {
            std::vector<std::string> stages = {"text_model"};
            if (has_spec) {
                stages.push_back("spec_model");
            }
            if (has_mmproj) {
                stages.push_back("mmproj_model");
            }
            load_progress_text.stages   = stages;
            load_progress_mmproj.stages = stages;
            load_progress_spec.stages   = stages;

            // trigger 0% progress
            load_progress_callback(0.0f, &load_progress_text);
        }


        SRV_INF("loading model '%s'\n", params.model.get_name().c_str());
        SRV_TRC("local path '%s'\n", params.model.path.c_str());

        std::string & mmproj_path = params_base.mmproj.path;
        mtmd_context_params mparams = mtmd_context_params_default();
        if (has_mmproj) {
            mparams.use_gpu          = params_base.mmproj_use_gpu;
            mparams.device           = params_base.mmproj_device;
            mparams.print_timings    = false;
            mparams.n_threads        = params_base.cpuparams.n_threads;
            mparams.flash_attn_type  = params_base.flash_attn_type;
            mparams.warmup           = params_base.warmup;
            mparams.image_min_tokens = params_base.image_min_tokens;
            mparams.image_max_tokens = params_base.image_max_tokens;
            mparams.batch_max_tokens = params_base.mtmd_batch_max_tokens;
            mparams.media_marker     = get_media_marker();
            // progress callback
            mparams.progress_callback           = load_progress_callback;
            mparams.progress_callback_user_data = &load_progress_mmproj;
        }

        // get the memory usage of mmproj, also used to check image_max_tokens against n_ubatch
        mtmd_memory_usage mmproj_usage = {{}, -1, false};
        int64_t mmproj_usage_t_us = 0;
        if (has_mmproj) {
            int64_t t_start = ggml_time_us();
            mmproj_usage = mtmd_get_memory_usage(mmproj_path.c_str(), mparams);
            mmproj_usage_t_us = ggml_time_us() - t_start;
        }

        // optionally fit mmproj memory usage
        if (has_mmproj && params_base.fit_params) {
            const auto & mmproj_mem = mmproj_usage.backend_mem_usage;
            const int64_t t_elapsed = mmproj_usage_t_us;
            if (!mmproj_mem.empty()) {
                size_t total = 0;
                for (auto & [dev, size] : mmproj_mem) {
                    total += size;
                }
                SRV_TRC("[mtmd] estimated worst-case memory usage of mmproj is %.2f MiB (took %.2f ms)\n", total / (1024.0 * 1024.0), t_elapsed / 1000.0);
                GGML_ASSERT(!params_base.fit_params_target.empty());
                for (auto & [dev, size] : mmproj_mem) {
                    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                        if (ggml_backend_dev_get(i) == dev) {
                            if (i < params_base.fit_params_target.size()) {
                                SRV_DBG("[mtmd] adding %.2f MiB to fit_params_target for device %s\n", size / (1024.0 * 1024.0), ggml_backend_dev_name(dev));
                                params_base.fit_params_target[i] += size;
                            }
                            break;
                        }
                    }
                }
            } else {
                SRV_ERR("%s", "[mtmd] failed to get memory usage of mmproj\n");
            }
        }

        // note: the draft / MTP context is fitted together with the target model, see common_fit_extra_model

        // attach a progress callback
        {
            params_base.load_progress_callback = load_progress_callback;
            params_base.load_progress_callback_user_data = &load_progress_text;
        }

        llama_init = common_init_from_params(params_base);

        model_tgt = llama_init->model();
        ctx_tgt   = llama_init->context();

        if (model_tgt == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        if (ctx_tgt == nullptr) {
            SRV_ERR("failed to create_context with model '%s'\n", params_base.model.path.c_str());
            return false;
        }

        vocab = llama_model_get_vocab(model_tgt);

        try {
            decision.init(model_tgt);
        } catch (const std::exception & e) {
            SRV_ERR("failed to init decision model: %s\n", e.what());
            return false;
        }

        n_ctx = llama_n_ctx(ctx_tgt);

        add_bos_token = llama_vocab_get_add_bos(vocab);

        char native_mtp_arch[32] = {};
        llama_model_meta_val_str(model_tgt, "general.architecture", native_mtp_arch, sizeof(native_mtp_arch));
        const int native_mtp_lanes = params_base.pipe_lanes < 0 ? llama_pipeline_stages(ctx_tgt) : params_base.pipe_lanes;
        const bool native_mtp_pipeline = spec_mtp && !has_draft && native_mtp_lanes > 1 &&
                !params_base.kv_unified && llama_decode_pipelined_supported(ctx_tgt) &&
                llama_model_n_devices(model_tgt) > 1 && llama_model_n_layer_nextn(model_tgt) > 0 &&
                (std::strcmp(native_mtp_arch, "qwen35") == 0 || std::strcmp(native_mtp_arch, "qwen35moe") == 0);
        // The pipelined MTP drafts run on the target model's resident NextN heads
        // (no replica model), so every layer including NextN must be on a GPU.
        if (native_mtp_pipeline && (params_base.split_mode != LLAMA_SPLIT_MODE_LAYER ||
                (params_base.n_gpu_layers >= 0 &&
                 params_base.n_gpu_layers < llama_model_n_layer(model_tgt) + llama_model_n_layer_nextn(model_tgt)))) {
            SRV_ERR("%s", "pipelined native MTP requires --split-mode layer with all layers (incl. NextN) offloaded\n");
            return false;
        }

        if (has_spec) {
            // spec_mtp doesn't use load a model internally, so we report 0.0 and 1.0 manually
            load_progress_callback(0.0f, &load_progress_spec);
            load_progress_spec.t_last_load_progress_ms = 0;  // reset so internal cbs aren't delayed

            {
                common_params params_dft = common_base_params_to_speculative(params_base);
                if (spec_mtp && (has_draft || native_mtp_pipeline) && params_base.pipe_lanes != 1 && llama_model_n_devices(model_tgt) > 1) {
                    params_dft.devices = { llama_model_get_device(model_tgt, 0), nullptr };
                    params_dft.split_mode = LLAMA_SPLIT_MODE_LAYER;
                    params_dft.main_gpu = 0;
                    std::fill(std::begin(params_dft.tensor_split), std::end(params_dft.tensor_split), 0.0f);
                }
                // progress callback
                params_dft.load_progress_callback           = load_progress_callback;
                params_dft.load_progress_callback_user_data = &load_progress_spec;

                spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
                model_dft = spec_init->model();
                ctx_dft   = spec_init->context();


                if (has_draft && model_dft == nullptr) {
                    SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                    return false;
                }

                if (ctx_dft == nullptr) {
                    SRV_ERR("%s", "failed to create MTP context\n");
                    return false;
                }

                params_base.speculative.draft.ctx_tgt = ctx_tgt;
                params_base.speculative.draft.ctx_dft = ctx_dft;
            }

            load_progress_callback(1.0f, &load_progress_spec);
        }

        if (has_mmproj) {
            if (callback_state) {
                callback_state(SERVER_STATE_LOADING, {{"stage", "mmproj_model"}});
            }

            if (!is_resume) {
                mtmd_helper_log_set(common_log_default_callback, nullptr);
            }

            // the image must fit in one ubatch if the model needs non-causal attention on it
            // a non-causal model also has the text of the prompt in that ubatch, leave half of it to the text
            {
                const int n_ubatch = llama_n_ubatch(ctx_tgt);
                const bool is_mixed = use_mixed_batch();
                if (is_mixed || mmproj_usage.use_non_causal) {
                    const int n_max = is_mixed ? n_ubatch / 2 : n_ubatch;
                    if (mmproj_usage.image_max_tokens > n_max) {
                        SRV_WRN("cap image_max_tokens (original=%d) to %d (n_ubatch = %d) because %s\n", mmproj_usage.image_max_tokens, n_max, n_ubatch,
                                is_mixed ? "the model processes the prompt in one ubatch" : "model needs non-causal attention on image");
                        SRV_WRN("%s\n", "increase n_ubatch (-ub) to increase vision token budget");
                        mparams.image_max_tokens = n_max;
                        mparams.image_min_tokens = std::min(mparams.image_min_tokens, n_max);
                    }
                }
            }

            mctx = mtmd_init_from_file(mmproj_path.c_str(), model_tgt, mparams);
            if (mctx == nullptr) {
                SRV_ERR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
                return false;
            }
            SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());

            init_opt.video_params.fps_target = params_base.video_fps;
            init_opt.video_params.timestamp_interval_ms = params_base.video_timestamp_interval_ms;
            init_opt.video_params.ffmpeg_bin_dir = params_base.video_ffmpeg_bin_dir.empty()
                                ? nullptr : params_base.video_ffmpeg_bin_dir.c_str();

            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
            }
        }

        if (!llama_memory_can_shift(llama_get_memory(ctx_tgt))) {
            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
            }
        }

        if (llama_model_n_swa(model_tgt) == 0) {
            if (params_base.swa_full) {
                params_base.swa_full = false;
                SRV_WRN("%s\n", "swa_full is not supported by this model, it will be disabled");
            }
        }

        n_swa = params_base.swa_full ? 0 : llama_model_n_swa(model_tgt);

        // Necessary similarity of prompt for slot selection
        slot_prompt_similarity = params_base.slot_prompt_similarity;

        const int n_ctx_train = llama_model_n_ctx_train(model_tgt);

        {
            // note: the capping itself is done in n_ctx_slot(), here we only report it
            const int n_ctx_seq = llama_n_ctx_seq(ctx_tgt);

            if (params_base.kv_unified_per_slot > 0) {
                if (n_ctx_seq > params_base.kv_unified_per_slot) {
                    SRV_INF("capping per-slot context (%d) to --kv-unified-per-slot (%d)\n",
                            n_ctx_seq, params_base.kv_unified_per_slot);
                } else if (params_base.kv_unified_per_slot > n_ctx_seq) {
                    // cap is above the per-slot pool capacity, so it can never bind
                    SRV_WRN(
                        "--kv-unified-per-slot (%d) exceeds the per-slot pool capacity (%d) - cap has no effect, "
                        "slots are limited to %d (raise the KV pool with -c, or unset -c to size it to "
                        "n_parallel * kv_unified_per_slot)\n",
                        params_base.kv_unified_per_slot, n_ctx_seq, n_ctx_seq);
                }
            }

            const int n_ctx_capped = params_base.kv_unified_per_slot > 0 ?
                std::min(n_ctx_seq, params_base.kv_unified_per_slot) : n_ctx_seq;

            if (n_ctx_capped > n_ctx_train) {
                SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n",
                        n_ctx_capped, n_ctx_train);
            }
        }

        slots.clear();

        ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }

        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            SRV_TRC("%s", "speculative decoding will use checkpoints\n");
        }

        // setup slots
        SRV_INF("initializing, n_slots = %d, n_ctx_slot = %d, kv_unified = '%s'\n",
                params_base.n_parallel, n_ctx_slot(), params_base.kv_unified ? "true" : "false");

        // initialize slots
        for (int i = 0; i < params_base.n_parallel; i++) {
            slots.emplace_back();
        }

        // try speculative decoding
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            try {
                spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
            } catch (const std::exception & e) {
                SRV_ERR("failed to initialize speculative decoding context: %s\n", e.what());
                if (params_base.speculative.has_synth()) {
                    return false;
                }
            }
        }

        if (ctx_dft) {
            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft);
        }

        if (spec) {
            SRV_TRC("%s", "speculative decoding context initialized\n");
        } else {
            spec_init.reset();
            ctx_dft   = nullptr;
            model_dft = nullptr;
        }

        if (!spec && params_base.speculative.has_synth()) {
            SRV_ERR("%s", "synthetic acceptance requires an initialized speculative decoding context\n");
            return false;
        }

        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot & slot = slots[i];

            slot.id      = i;
            slot.ctx_tgt = ctx_tgt;
            slot.ckpt_barrier = [this](int sid) { if (!ckpt_async_items.empty()) { ckpt_async_poll(sid, true); } };
            if (checkpoint_recycle_max > 0) { slot.ckpt_recycle = [this](std::vector<uint8_t> & v) { ckpt_recycle_push(v); }; }
            slot.ctx_dft = ctx_dft;
            slot.mem.init(ctx_tgt, ctx_dft);
            slot.spec    = spec.get();
            slot.n_ctx   = n_ctx_slot();

            slot.mctx                   = mctx;
            slot.prompt.tokens.has_mtmd = mctx != nullptr;

            SLT_TRC(slot, "new slot, n_ctx = %d\n", slot.n_ctx);

            slot.callback_on_release = [this](int id_slot) {
                // The continuation owns this slot against ordinary queued work,
                // but a waiting VIP must still be allowed to take over.
                queue_tasks.pop_deferred_task(id_slot, slots[id_slot].borrowed_generation != 0);
            };

            slot.callback_on_reset = [this](const server_slot & slot) {
                // flush the generated token stats before reset()
                if (slot.stats.n_gen > 0) {
                    metrics_on_prediction(slot);
                }
                // context-length histogram: count completions by (prompt + generated) token total
                // 64 buckets: 0-1k, 1-2k, ..., 63-64k, >=64k
                const int32_t total_ctx = (int32_t)(slot.prompt.n_tokens() + slot.stats.n_gen);
                if (total_ctx > 0) {
                    const int bk = (total_ctx < 65536) ? (total_ctx / 1024) : 63;
                    dash_ctx_hist[bk]++;
                }
            };

            slot.reset();
        }

        // Apply pending draft graph flags before enabling owned input snapshots.
        // Shared-KV assistants keep the existing copy policy.
        if (spec_mtp && ctx_dft && llama_get_ctx_other(ctx_dft) != ctx_tgt) {
            llama_set_input_snapshots(ctx_dft, true);
        }

        pipeline_count = params_base.pipe_lanes < 0 ? llama_pipeline_stages(ctx_tgt) : params_base.pipe_lanes;
        pipeline_count = std::max(1, std::min(pipeline_count, llama_pipeline_stages(ctx_tgt) + 2));
        pipeline_on = pipeline_count > 1 && !params_base.kv_unified && llama_decode_pipelined_supported(ctx_tgt);
        if (pipeline_on) {
            if (!llama_set_decode_pipeline_peer(ctx_tgt, ctx_dft)) {
                SRV_ERR("%s", "ParaFlow could not bind the draft device plan\n");
                return false;
            }
            llama_set_decode_pipeline_lanes(ctx_tgt, pipeline_count);
            if (!llama_set_decode_pipelined(ctx_tgt, true)) {
                pipeline_on = false;
                SRV_ERR("%s", "ParaFlow pipeline activation failed; server startup aborted\n");
                return false;
            }
            pipeline_lanes.resize(pipeline_count);
            for (auto & slot : slots) { slot.pipeline_lane = pipeline_lane_of(slot.id); }
            {
                // prompt-only lanes: default one more than the decode lanes
                const int requested = std::clamp(params_base.pipe_prefill_lanes < 0 ? pipeline_count + 1 : params_base.pipe_prefill_lanes, 0, 8);
                const bool mtp_lanes_ok = spec && spec_mtp;
                if (requested > 0 && (!spec || mtp_lanes_ok) && params_base.pipe_fuse) {
                    pipeline_mtp_prefill_lanes = mtp_lanes_ok;
                    if (mtp_lanes_ok) { SRV_INF("%s", "MTP prompt chunks use prompt-only lanes (shared ready-batch draft driver)\n"); }
                    pipeline_prefill_lanes = requested;
                    pipeline_lanes.resize(pipeline_count + pipeline_prefill_lanes);
                    SRV_INF("pipeline prefill-only lanes=%d decode lanes=%d\n", pipeline_prefill_lanes, pipeline_count);
                } else if (requested > 0) {
                    SRV_WRN("%s", "prompt-only lanes disabled: they require --pipe-fuse and no non-MTP speculative decoding\n");
                }
            }
            if (spec_mtp && (has_draft || native_mtp_pipeline) && spec) {
                if (pipeline_count > llama_model_n_devices(model_tgt)) {
                    SRV_ERR("%s", "MTP pipeline lanes exceed available target devices\n");
                    return false;
                }
                // One draft context and driver serve every lane: drafts of all staged
                // lanes run as one batched head step (ready-batch, stepwise).
                if (!ctx_dft || llama_get_ctx_other(ctx_dft) == ctx_tgt ||
                        llama_model_n_layer_nextn(llama_get_model(ctx_dft)) != 1 ||
                        !common_speculative_supports_draft_step(spec.get())) {
                    SRV_ERR("%s", "pipelined MTP requires one independent single-head MTP draft context\n");
                    return false;
                }
                pipeline_ready_draft_batch = true;
                pipeline_stepwise_drafts = true;
                pipeline_async_drafts = true;
                SRV_INF("MTP ready-batch drafts: contexts=1 drivers=1 target_lanes=%d\n", pipeline_count);
            }
            SRV_INF("asynchronous pipeline enabled, lanes=%d\n", pipeline_count);
            // The executor thread runs draft steps concurrently with target scheduling. It needs
            // partial draft-KV removal; otherwise drafts run on the scheduler thread.
            if (pipeline_ready_draft_batch && ctx_dft_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_FULL &&
                    ctx_dft_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_RS) {
                dft_n_embd = llama_model_n_embd_out(model_tgt);
                dft_done.reset(new std::atomic<bool>[slots.size()]);
                for (size_t i = 0; i < slots.size(); ++i) { dft_done[i].store(true); }
                dft_hbuf.resize(DFT_RING);
                dft_bbuf.resize(DFT_RING);
                dft_exec = std::make_unique<dft_executor>();
                dft_exec->pump = [this]() { return dft_pump_step(); };
                dft_exec->start();
                SRV_INF("MTP draft executor enabled (n_embd=%d, ring=%d)\n", dft_n_embd, DFT_RING);
            }
        }

        {
            const char * LLAMA_TRACE = getenv("LLAMA_TRACE");
            trace = LLAMA_TRACE ? atoi(LLAMA_TRACE) : 0;

            if (trace) {
                SRV_WRN("LLAMA_TRACE = %d\n", trace);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("LLAMA_SERVER_SLOTS_DEBUG = %d\n", slots_debug);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_N_DIFF = getenv("LLAMA_SERVER_SLOTS_N_DIFF");
            slots_n_diff = LLAMA_SERVER_SLOTS_N_DIFF ? atoi(LLAMA_SERVER_SLOTS_N_DIFF) : 0;

            if (slots_n_diff) {
                SRV_WRN("LLAMA_SERVER_SLOTS_N_DIFF = %d\n", slots_n_diff);
            }
        }

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx_tgt);
            const int32_t n_embd  = llama_model_n_embd_inp(model_tgt);
            batch.init(ctx_tgt, std::max(n_batch, params_base.n_parallel), n_embd);
        }

        // decision requests rarely repeat a state, the prompt cache is opt-in for decision models
        if (decision.type != COMMON_DECISION_TYPE_NONE && !params_base.cache_ram_explicit) {
            SRV_INF("%s", "decision model: prompt cache disabled by default, use `--cache-ram N` to enable it\n");
            params_base.cache_ram_mib    = 0;
            params_base.cache_idle_slots = false;
        }

        if (params_base.cache_ram_mib != 0) {
            if (params_base.cache_ram_mib < 0) {
                SRV_TRC("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_TRC("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            SRV_TRC("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx);
        } else {
            SRV_TRC("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
        SRV_TRC("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

        if (params_base.n_ctx_checkpoints > 0) {
            SRV_TRC("context checkpoints enabled, max = %d, min spacing = %d\n",
                    params_base.n_ctx_checkpoints, params_base.checkpoint_min_step);
        } else {
            SRV_TRC("%s", "context checkpoints disabled\n");
        }

        if (!params_base.model_alias.empty()) {
            // backward compat: use first alias as model name
            model_name = *params_base.model_alias.begin();
        } else if (!params_base.model.get_name().empty()) {
            model_name = params_base.model.get_name();
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name = model_path.filename().string();
        }

        model_aliases = params_base.model_alias;
        model_tags    = params_base.model_tags;

        // propagate new defaults back to caller
        params = params_base;

        if (!is_resume) {
            return init();
        }

        if (callback_state) {
            callback_state(SERVER_STATE_READY, {});
        }

        return true;
    }

    // unlike load_model(), this is only called once during initialization
    bool init() {
        GGML_ASSERT(ctx_tgt   != nullptr);
        GGML_ASSERT(model_tgt != nullptr);

        GGML_ASSERT(!sleeping);

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task, bool is_yielding) {
            return process_single_task(std::move(task), is_yielding);
        });
        queue_tasks.on_update_slots([this]() {
            const int64_t t_enter = dash_hist ? paraflow_wall_ms() : 0;
            update_slots();
            if (dash_hist) { dash_hist_record(t_enter); }
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });

        metrics.init();

        if (params_base.dash) {
            const std::string cfg = dash_make_config();
            std::lock_guard<std::mutex> lock(dash_mutex);
            dash_config = cfg;
            dash_hist.reset(new paraflow_dash_bucket[PARAFLOW_DASH_BUCKETS]);
            dash_hist_start = paraflow_wall_ms() / PARAFLOW_DASH_BUCKET_MS;
            dash_hist_prompt = metrics.n_prompt_observed; dash_hist_predict = metrics.n_predict_observed;
            SRV_INF("%s", "dashboard enabled at /paraflow/dash\n");
        }

        if (params_base.cache_idle_slots) {
            if (params_base.cache_ram_mib == 0) {
                SRV_WRN("%s", "--cache-idle-slots requires --cache-ram, disabling\n");
                params_base.cache_idle_slots = false;
            } else {
                if (params_base.kv_unified) {
                    SRV_TRC("%s", "idle slots will be saved to prompt cache and cleared upon starting a new task\n");
                } else {
                    // without a unified KV cache, clearing a slot frees no reusable room, so we only
                    // publish a RAM-cache copy of idle slots (their KV stays in VRAM) [TAG_IDLE_SLOT_CLEAR]
                    SRV_TRC("%s", "idle slots will be saved to prompt cache upon starting a new task\n");
                }
                SRV_DBG("%s", "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__\n");
            }
        }

        {
            const std::string & cfg = params_base.ui_config_json;
            if (!cfg.empty()) {
                try {
                    json json_settings = json::parse(cfg);
                    json_ui_settings = json_settings;
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse UI config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;
            bool enable_thinking = false;

            try {
                chat_templates = common_chat_templates_init(model_tgt, params_base.chat_template);

                SRV_TRC("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

                // thinking is enabled if:
                // 1. It's not explicitly disabled via --reasoning off
                // 2. The chat template supports it
                const bool template_supports_thinking = params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
                enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;
                SRV_TRC("%s: chat template, thinking = %d\n", __func__, enable_thinking);
            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR("%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            // IMPORTANT: chat_params is reused across sleeping / resuming states,
            //            never store llama_context/llama_model pointers in chat_params,
            //            as they may be invalidated after sleeping
            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* allow_image           */ mctx ? mtmd_support_vision(mctx) : false,
                /* allow_audio           */ mctx ? mtmd_support_audio (mctx) : false,
                /* allow_video           */ mctx ? mtmd_helper_support_video(mctx) : false,
                /* enable_thinking       */ enable_thinking,
                /* reasoning_budget      */ params_base.sampling.reasoning_budget_tokens,
                /* reasoning_budget_msg  */ params_base.sampling.reasoning_budget_message,
                /* media_path            */ params_base.media_path,
                /* force_pure_content    */ params_base.force_pure_content_parser
            };

            {
                auto caps = common_chat_templates_get_caps(chat_params.tmpls.get());
                auto it = params_base.default_template_kwargs.find("preserve_reasoning");
                bool supported = caps.at("supports_preserve_reasoning");
                bool specified = params_base.preserve_reasoning_specified;
                // note: the kwarg is enabled by default if not specified explicitly, so check the value
                bool enabled = it != params_base.default_template_kwargs.end() && it->second == "true";
                if (supported) {
                    SRV_TRC("preserve_reasoning kwarg: %s\n",
                            it == params_base.default_template_kwargs.end() ? "unset (template default)" : it->second.c_str());
                } else {
                    SRV_TRC("%s", "preserve_reasoning kwarg: not supported by template\n");
                }
                if (supported && !specified) {
                    SRV_WRN("%s", "chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable via --no-reasoning-preserve)\n");
                }
                if (supported && !enabled) {
                    SRV_INF("%s", "chat template supports preserving reasoning, consider enabling it via --reasoning-preserve\n");
                }
                if (!supported && specified && enabled) {
                    SRV_WRN("%s", "chat template does NOT support preserving reasoning, --reasoning-preserve has no effect\n");
                }
            }
        }

        return true;
    }

    server_slot * get_slot_by_id(int id_slot) {
        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_slot_by_cmpl_id(const std::string & cmpl_id) {
        if (cmpl_id.empty()) {
            return nullptr;
        }

        for (server_slot & slot : slots) {
            if (slot.is_processing() && slot.task && slot.task->params.oaicompat_cmpl_id == cmpl_id) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_available_slot(const server_task & task) {
        server_slot * ret = nullptr;

        bool update_cache = false;
        std::vector<int> lane_load(pipeline_count, 0);
        for (const auto & current : slots) {
            if (current.is_processing() && !current.resident_paused) { lane_load[pipeline_lane_of(current.id)]++; }
        }
        int min_available_load = INT_MAX;
        for (const auto & current : slots) {
            if (!current.is_processing() && (!current.borrowed_generation || task.params.priority > 0)) {
                min_available_load = std::min(min_available_load, lane_load[pipeline_lane_of(current.id)]);
            }
        }

        // if a specific slot is requested, use it (still goes through cache update logic below)
        if (task.id_slot != -1) {
            ret = get_slot_by_id(task.id_slot);
            if (ret) {
                // a busy slot is returned untouched, the caller defers the task
                if (ret->is_processing() || (ret->borrowed_generation && task.params.priority <= 0)) {
                    return ret;
                }

                SLT_INF(*ret, "selected slot by id (%d)\n", task.id_slot);
            }
        }

        // find the slot that has at least n% prompt similarity
        if (slot_prompt_similarity != 0.0f) {
            float f_sim_best = 0;

            for (server_slot & slot : slots) {
                if (task.id_slot != -1 && slot.id != task.id_slot) {
                    continue;
                }

                // skip the slot if it is not available
                if (slot.is_processing() || (slot.borrowed_generation && task.params.priority <= 0)) {
                    SLT_TRC(slot, " - skipping, is_processing = %d\n", slot.is_processing());
                    continue;
                }

                const auto & tokens = slot.prompt.tokens;

                // skip the slot if it does not contains cached tokens
                if (tokens.empty()) {
                    SLT_TRC(slot, "%s", " - skipping, slot is empty\n");
                    continue;
                }

                if (pipeline_on && task.id_slot < 0 && lane_load[pipeline_lane_of(slot.id)] > min_available_load) {
                    continue;
                }

                // fraction of the Longest Common Prefix length with respect to the input prompt length
                const size_t lcp_len = tokens.get_common_prefix(task.tokens);
                const float f_sim_cur = float(lcp_len) / task.tokens.size();

                SLT_TRC(slot, " - checking sim = %.3f (%zu/%zu) > %.3f\n", f_sim_cur, lcp_len, task.tokens.size(), slot_prompt_similarity);

                // select the current slot if the criteria match
                if (f_sim_cur > f_sim_best && f_sim_cur > slot_prompt_similarity) {
                    f_sim_best = f_sim_cur;

                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                const float f_keep = (f_sim_best*task.tokens.size()) / ret->prompt.tokens.size();

                if (task.id_slot == -1) {
                    SLT_INF(*ret, "selected slot by LCP similarity, f_sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                            f_sim_best, slot_prompt_similarity, f_keep);
                }

                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5f) {
                    update_cache = true;
                }
            }
        }

        // find an empty slot, or the one that has been least recently used
        if (ret == nullptr) {
            const auto free_slots = get_free_slots(1, -1, task.params.priority > 0);

            if (!free_slots.empty()) {
                ret = free_slots[0];

                SLT_INF(*ret, "selected slot by LRU, n_tokens = %d, t_last = %" PRId64 "\n", ret->prompt.n_tokens(), ret->t_last_used);

                update_cache = true;
            }
        }

        if (ret) {
            update_cache = update_cache && prompt_cache;

            // cache prompts only for the tasks that reuse them
            update_cache = update_cache && !task.is_stateless();

            if (update_cache) {
                SRV_TRC("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                ret->prompt_save(*prompt_cache);

                if (!ret->prompt_load(*prompt_cache, task.tokens, task.params.mtp)) {
                    ret->prompt_clear();
                }

                prompt_cache->update();

                SRV_TRC("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
        }

        return ret;
    }

    // return true if at least one slot has been cleared
    // TODO: improve logic
    //       - smarter decision which slot to clear (LRU or longest prompt?)
    //       - move slot to level 2 cache instead of removing?
    //       - instead of purging, try to store and resume later?
    bool try_clear_idle_slots() {
        bool res = false;

        if (!params_base.kv_unified && !params_base.kv_dynamic) {
            return res;
        }

        for (auto & slot : slots) {
            if (slot.is_processing() || slot.borrowed_generation) {
                continue;
            }

            if (slot.prompt.n_tokens() > 0) {
                SRV_WRN("purging slot %d with %zu tokens\n", slot.id, slot.prompt.tokens.size());

                slot.prompt_clear();

                llama_memory_trim(llama_get_memory(ctx_tgt));
                if (slot.ctx_dft) { dft_join(__LINE__); llama_memory_trim(llama_get_memory(slot.ctx_dft)); }
                res = true;

                // clear slots one by one
                break;
            }
        }

        return res;
    }

    std::vector<common_adapter_lora_info> construct_lora_list(const std::map<int, float> & config) const {
        std::vector<common_adapter_lora_info> output = params_base.lora_adapters; // copy
        for (size_t i = 0; i < output.size(); ++i) {
            auto it = config.find(i);
            if (it != config.end()) {
                output[i].scale = it->second;
            } else {
                output[i].scale = 0.0f;
            }
        }
        return output;
    }

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        dft_join(__LINE__);
        if (params_base.kv_dynamic && task.n_tokens() >= n_ctx) {
            send_error(task, "prompt exceeds the shared physical KV capacity", ERROR_TYPE_EXCEED_CONTEXT_SIZE);
            return false;
        }
        if (task.params.mtp && !slot.spec) {
            send_error(task, "MTP is not configured on this server", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        if (task.params.mtp && !slot.prompt.mtp_ready) {
            // Conventional requests do not build draft KV. Rebuild this prompt
            // once when switching modes rather than using stale draft state.
            slot.prompt_clear();
        }
        if (!task.is_parent() && !task.is_child()) {
            pipeline_place_slot(slot, task.params.mtp, task.params.priority);
        }
        // process per-request lora adapters
        if (!task.params.lora.empty()) {
            auto task_loras = construct_lora_list(task.params.lora);
            if (!are_lora_equal(task_loras, slot.lora)) {
                // if lora has changed, check to see if the cache should be cleared
                if (lora_should_clear_cache(slot.lora, task_loras)) {
                    SLT_TRC(slot, "clearing cache for lora change. %zu loras -> %zu loras\n", slot.lora.size(), task.params.lora.size());
                    if (slot.ckpt_barrier) { slot.ckpt_barrier(slot.id); }
                    slot.prompt.clear();
                } else {
                    SLT_TRC(slot, "keeping cache for alora. %zu target loras\n", task_loras.size());
                }
                slot.lora = task_loras;
            }
        } else {
            slot.lora = params_base.lora_adapters;
        }

        // if using alora, make sure it's only a single one requested and active
        size_t alora_invocation_start = task.tokens.size();
        if (lora_all_alora(slot.lora)) {
            const auto & enabled_ids = lora_get_enabled_ids(slot.lora);
            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.size() != 1) {
                send_error(task, "Cannot run multiple aLoRAs in a single request", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            const auto & lora = slot.lora[enabled_ids[0]].ptr;

            // get the pointer and count for the invocation tokens
            const uint64_t      n_invocation_tokens = llama_adapter_get_alora_n_invocation_tokens(lora);
            const llama_token * invocation_tokens   = llama_adapter_get_alora_invocation_tokens  (lora);

            // scan backwards through the prompt tokens to find the last
            // occurrence of the invocation sequence
            int match_idx = static_cast<int>(n_invocation_tokens) - 1;
            for (int i = task.tokens.size() - 1; i >= 0; --i) {
                // the token in this position matches the next token to find in
                // the invocation sequence
                if (task.tokens[i] == invocation_tokens[match_idx]) {
                    // if it's a full match, we've found the start
                    if (match_idx == 0) {
                        alora_invocation_start = i;
                        break;
                    }
                    // otherwise, check the next token in the sequence
                    --match_idx;
                } else {
                    // no match in this position, so start looking over again
                    match_idx = static_cast<int>(n_invocation_tokens) - 1;
                }
            }

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                SLT_DBG(slot, "alora %zu requested, but not found. deactivating\n", enabled_ids[0]);
                slot.lora[enabled_ids[0]].scale = 0.0f;
            } else {
                SLT_DBG(slot, "alora %zu activated starting at %zu\n", enabled_ids[0], alora_invocation_start);
                slot.alora_invocation_start = alora_invocation_start;
            }
        }

        if (!task.tokens.validate(ctx_tgt)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        // initialize samplers
        if (task.need_sampling()) {
            try {
                slot.smpl.reset(common_sampler_init(model_tgt, task.params.sampling));
            } catch (std::exception & e) {
                std::string err_msg = std::string("Failed to initialize samplers: ") + e.what();
                send_error(task, err_msg, ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            const bool need_pre_sample_logits = task.params.sampling.n_probs > 0 && !task.params.post_sampling_probs;

            bool use_backend_sampling = task.params.sampling.backend_sampling;

            // TODO: getting pre sampling logits is not yet supported with backend sampling
            use_backend_sampling &= !need_pre_sample_logits && !pipeline_on && !task.params.mtp;

            // TODO: tmp until backend sampling is fully implemented
            if (use_backend_sampling) {
                llama_set_sampler(ctx_tgt, slot.id, common_sampler_get(slot.smpl.get()));
            } else {
                llama_set_sampler(ctx_tgt, slot.id, nullptr);
            }

            SLT_TRC(slot, "sampler chain: %s\n", common_sampler_print(slot.smpl.get()).c_str());
            SLT_TRC(slot, "sampler params: \n%s\n", task.params.sampling.print().c_str());

            if (spec && !common_speculative_get_synth_probs(spec.get()).empty()) {
                const uint32_t seed = task.params.sampling.seed == LLAMA_DEFAULT_SEED
                    ? std::random_device{}()
                    : task.params.sampling.seed;
                slot.spec_synth_rng.seed(seed);
            }
        } else {
            slot.smpl.reset();
        }

        // the per-request limit takes priority over the global one
        slot.n_predict_max = task.params.n_predict != -1 ? task.params.n_predict : params_base.n_predict;

        slot.task = std::make_unique<const server_task>(std::move(task));
        slot.system_cache_restored_tokens = 0;
        slot.system_cache_restore_us = 0;

        slot.state = slot.task->is_child()
            ? SLOT_STATE_WAIT_OTHER // wait for the parent to process prompt
            : SLOT_STATE_STARTED;

        load_system_prefix(slot);

        // reset server kill-switch counter
        n_empty_consecutive = 0;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        return true;
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        metrics.n_predict_observed++;
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled = result.tok;
        server_loop_guard::match loop_match;
        if (params_base.loop_guard) {
            slot.loop_guard_tokens.push_back(result.tok);
            slot.loop_guard_generated++;
            while (slot.loop_guard_tokens.size() > (size_t) slot.n_ctx) { slot.loop_guard_tokens.pop_front(); }
            loop_match = server_loop_guard::detect(slot.loop_guard_generated, slot.loop_guard_tokens);
        }

        slot.generated.append(token_str, result.tok);
        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated.text) < slot.generated.text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated.text.size());

            const std::string str_test = slot.generated.text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated.truncate(pos + stop_pos);
                pos = std::min(slot.n_sent_text, slot.generated.text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok) ) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated.text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            slot.add_token(result);
            if (slot.task->params.stream) {
                send_partial_response(slot, result, false);
            }
        }

        if (incomplete) {
            // Delay text until the UTF-8 character is complete, while keeping
            // every generated token visible in the stream and probability log.
            result.text_to_send.clear();
            slot.add_token(result);
            if (slot.task->params.stream) {
                send_partial_response(slot, result, false);
            }
            slot.has_next_token = true;
        }

        if (loop_match && !incomplete) {
            slot.stop = STOP_TYPE_LIMIT;
            slot.has_next_token = false;
            SLT_WRN(slot, "loop guard stopped generation: period=%zu repeats=%zu\n", loop_match.period, loop_match.repeats);
        }

        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_gen = %d, n_ctx = %d\n",
                    slot.prompt.n_tokens(), slot.task->n_tokens(), (int) slot.stats.n_gen, slot.n_ctx);
        }

        // check the limits
        if (slot.stats.n_gen > 0 && slot.has_next_token && !slot.has_budget()) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_gen = %d, n_predict = %d\n", (int) slot.stats.n_gen, slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated.text.size() && (slot.generated.text[pos] == ' ' || slot.generated.text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated.text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated.truncate(pos);

                        SLT_DBG(slot, "stopped by indentation limit, n_gen = %d, n_indent = %d\n", (int) slot.stats.n_gen, n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated.text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 && slot.stats.t_gen_ms() > slot.task->params.t_max_predict_ms) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_gen = %d, t_max_predict_ms = %d ms\n", (int) slot.stats.n_gen, (int) slot.task->params.t_max_predict_ms);
            }
        }

        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(slot, "n_gen = %d, n_remaining = %d, next token: %5d '%s'\n", (int) slot.stats.n_gen, slot.n_remaining(), result.tok, token_str.c_str());

        return slot.has_next_token; // continue
    }

    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) const {
        const size_t n_probs_request = slot.task->params.sampling.n_probs;

        if (post_sampling) {
            const auto * cur_p = common_sampler_get_candidates(slot.smpl.get(), true);
            const size_t max_probs = cur_p->size;
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                // Some samplers do return 0.0 probabilities, others don't.
                // Filter 0.0 probailities, to ensure the behavior is consistent.
                if (cur_p->data[i].p == 0.0) {
                    break;
                }

                result.probs.push_back({
                    cur_p->data[i].id,
                    common_token_to_piece(ctx_tgt, cur_p->data[i].id, special),
                    cur_p->data[i].p
                });
            }
        } else {
            std::vector<llama_token_data> cur = get_token_probabilities(ctx_tgt, idx, n_probs_request);
            const size_t max_probs = cur.size();
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                // set probability for sampled token
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur[i].id,
                    common_token_to_piece(ctx_tgt, cur[i].id, special),
                    cur[i].p
                });
            }
        }
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const server_task_id id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0) {
        SRV_ERR("task id = %" PRId64 ", error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
        }

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress, bool is_begin = false) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.stats.n_prompt_cached;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.time_ms   = slot.stats.t_elapsed_us() / 1000;
        }
        if (is_begin) {
            res->is_begin = true;
        } else {
            res->content = slot.generated.substr(slot.n_sent_text - tkn.text_to_send.size(), tkn.text_to_send.size());
            if (!is_progress && tkn.tok != LLAMA_TOKEN_NULL) { res->tokens.assign(1, tkn.tok); }
        }

        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (!is_progress && slot.task->params.sampling.n_probs > 0 && tkn.tok != LLAMA_TOKEN_NULL) {
            res->prob_output = tkn; // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->stats = slot.stats;
        }

        queue_results.send(std::move(res));
    }

    void send_final_response(server_slot & slot) {
        // A token limit can fall inside a byte-fallback character. Streaming
        // already holds that incomplete suffix back; use the same valid text
        // prefix for the non-streamed response while preserving all token ids.
        slot.generated.truncate(validate_utf8(slot.generated.text));
        if (slot.task->params.stream && slot.n_sent_text < slot.generated.text.size()) {
            completion_token_output tail;
            tail.tok = LLAMA_TOKEN_NULL;
            tail.text_to_send = slot.generated.text.substr(slot.n_sent_text);
            slot.n_sent_text += tail.text_to_send.size();
            send_partial_response(slot, tail, false);
        }
        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;

        res->index = slot.task->index;

        // keep copy of last generated text for debugging purposes
        if (slots_debug) {
            slot.debug_generated_text = slot.generated.text;
        }

        // in stream mode, content and tokens are already in last partial chunk
        if (slot.task->params.stream) {
            res->content     = {};
            res->tokens      = llama_tokens{};
        } else {
            res->content     = std::move(slot.generated);
            res->tokens      = std::move(slot.generated_tokens);
        }
        res->stats           = slot.stats;
        res->prompt          = slot.task->tokens.detokenize(ctx_tgt, true);
        res->response_fields = std::move(slot.task->params.response_fields);

        res->truncated             = slot.truncated;
        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->n_tokens_cached       = slot.prompt.n_tokens();
        res->has_new_line          = slot.has_new_line;
        res->stopping_word         = slot.stopping_word;
        res->stop                  = slot.stop;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(ctx_tgt, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params; // copy the parameters

        queue_results.send(std::move(res));
    }

    void send_embedding(const server_slot & slot, const common_batch & batch) {
        auto res = std::make_unique<server_task_result_embd>();
        res->id        = slot.task->id;
        res->index     = slot.task->index;
        res->n_tokens  = slot.task->n_tokens();
        res->res_type  = slot.task->params.res_type;

        const int n_embd_out = llama_model_n_embd_out(model_tgt);

        std::vector<float> embd_res(n_embd_out, 0.0f);

        for (int i = 0; i < batch.size(); ++i) {
            if (!batch.tokens[i].output || batch.tokens[i].seq_id != slot.id) {
                continue;
            }

            const float * embd = nullptr;
            if (llama_pooling_type(slot.ctx_tgt) == LLAMA_POOLING_TYPE_NONE) {
                embd = llama_get_embeddings_ith(slot.ctx_tgt, i);
            } else {
                embd = llama_get_embeddings_seq(slot.ctx_tgt, batch.tokens[i].seq_id);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.tokens[i].id, batch.tokens[i].seq_id);

                res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
                continue;
            }

            // normalize only when there is pooling
            if (llama_pooling_type(slot.ctx_tgt) != LLAMA_POOLING_TYPE_NONE) {
                common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
                res->embedding.push_back(embd_res);
                break;
            }

            res->embedding.emplace_back(embd, embd + n_embd_out);
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(std::move(res));
    }

    void send_decision(const server_slot & slot, const common_batch & batch, int32_t i_batch) {
        auto res = std::make_unique<server_task_result_decision>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        const auto & decision = slot.task->decision;

        if (!decision.labels.empty()) {
            const float * logits = llama_get_logits_ith(slot.ctx_tgt, i_batch);
            if (logits == nullptr) {
                send_error(slot, "failed to get logits", ERROR_TYPE_SERVER);
                return;
            }
            const int32_t n_vocab = llama_vocab_n_tokens(vocab);
            for (const llama_token label : decision.labels) {
                GGML_ASSERT(label >= 0 && label < n_vocab);
                res->scores.push_back(logits[label]);
            }
            if (!decision.label_groups.empty()) {
                std::vector<float> scores;
                size_t i = 0;
                for (const int32_t n : decision.label_groups) {
                    GGML_ASSERT(n > 0 && i + n <= res->scores.size());
                    scores.push_back(*std::max_element(res->scores.begin() + i, res->scores.begin() + i + n));
                    i += n;
                }
                res->scores = std::move(scores);
            }
        } else {
            // the outputs of this slot in this batch are the last tokens of the prompt
            std::vector<int32_t> idx;
            for (int i = 0; i < batch.size(); ++i) {
                if (batch.tokens[i].output && batch.tokens[i].seq_id == slot.id) {
                    idx.push_back(i);
                }
            }
            const int32_t pos_first = slot.prompt.n_tokens() - (int32_t) idx.size();
            auto get_embd = [&](int32_t pos) -> const float * {
                const int32_t i = pos - pos_first;
                return i >= 0 && i < (int32_t) idx.size() ? llama_get_embeddings_ith(slot.ctx_tgt, idx[i]) : nullptr;
            };

            // joint head (decision model): the scores are the first rows
            for (int32_t i = 0; i < decision.n_scores; i++) {
                const float * embd = i < (int32_t) idx.size() ? llama_get_embeddings_ith(slot.ctx_tgt, idx[i]) : nullptr;
                if (embd == nullptr) {
                    send_error(slot, "failed to get embeddings", ERROR_TYPE_SERVER);
                    return;
                }
                res->scores.push_back(embd[0]);
            }

            const int32_t n_embd_out = llama_model_n_embd_out(model_tgt);
            const int32_t n_pointer  = n_embd_out / 2;
            const float * embd_q = decision.pointer >= 0 ? get_embd(decision.pointer) : nullptr;
            GGML_ASSERT(decision.column >= 0 && decision.column < n_embd_out);

            for (const int32_t marker : decision.markers) {
                const float * embd = get_embd(marker);
                if (embd == nullptr || (decision.pointer >= 0 && embd_q == nullptr)) {
                    send_error(slot, "failed to get embeddings, the question and its options must fit in one batch", ERROR_TYPE_SERVER);
                    return;
                }
                if (decision.pointer < 0) {
                    res->scores.push_back(embd[decision.column]);
                    continue;
                }
                float dot = 0.0f;
                for (int32_t i = 0; i < n_pointer; i++) {
                    dot += embd_q[i] * embd[n_pointer + i];
                }
                res->scores.push_back(dot / sqrtf((float) n_pointer));
            }
        }

        SLT_DBG(slot, "%s", "sending decision result\n");

        queue_results.send(std::move(res));
    }

    void send_rerank(const server_slot & slot, const common_batch & batch) {
        auto res = std::make_unique<server_task_result_rerank>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        for (int i = 0; i < batch.size(); ++i) {
            if (!batch.tokens[i].output || batch.tokens[i].seq_id != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx_tgt, batch.tokens[i].seq_id);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx_tgt, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.tokens[i].id, batch.tokens[i].seq_id);

                res->score = -1e6;
                continue;
            }

            res->score = embd[0];
        }

        SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

        queue_results.send(std::move(res));
    }

    //
    // Functions to process the task
    //

    // tokenize the input if it's set by CLI, return false on error
    bool tokenize_cli_input(server_task & task) {
        try {
            auto & prompt = task.cli_prompt;
            if (mctx != nullptr) {
                task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files, init_opt);
            } else {
                task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true, init_opt)[0]);
            }
            task.cli_prompt.clear();
            task.cli_files.clear();
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to format input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    // idle slots that lose the least by being reused: empty slots first, then the least recently used ones
    std::vector<server_slot *> get_free_slots(size_t n_slots_needed, int exclude_id_slot, bool allow_borrowed = false) {
        std::vector<server_slot *> free_slots, selected;
        std::vector<int> load(pipeline_count, 0);
        for (auto & slot : slots) {
            if (slot.is_processing() && !slot.resident_paused) { load[pipeline_lane_of(slot.id)]++; }
            if (!slot.is_processing() && slot.id != exclude_id_slot && (allow_borrowed || !slot.borrowed_generation)) {
                free_slots.push_back(&slot);
            }
        }
        if (exclude_id_slot >= 0 && !slots[exclude_id_slot].is_processing()) {
            load[pipeline_lane_of(exclude_id_slot)]++;
        }
        while (!free_slots.empty() && selected.size() < n_slots_needed) {
            const auto best = std::min_element(free_slots.begin(), free_slots.end(), [&](const server_slot * a, const server_slot * b) {
                const int la = load[pipeline_lane_of(a->id)], lb = load[pipeline_lane_of(b->id)];
                if (pipeline_on && la != lb) { return la < lb; }
                if ((a->prompt.n_tokens() == 0) != (b->prompt.n_tokens() == 0)) { return a->prompt.n_tokens() == 0; }
                return a->t_last_used < b->t_last_used;
            });
            auto * slot = *best;
            selected.push_back(slot);
            load[pipeline_lane_of(slot->id)]++;
            free_slots.erase(best);
        }
        return selected;
    }

    // launch multiple slots for parent + child tasks
    bool launch_slots_with_parent_task(server_slot & parent_slot, std::vector<server_slot *> & child_slots, server_task && parent_task) {
        GGML_ASSERT(!parent_slot.is_processing());
        GGML_ASSERT(parent_task.is_parent());
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());

        server_task_id id_parent = parent_task.id;

        SRV_TRC("launching slots for parent task id_task = %" PRId64 " with %zu child tasks\n", id_parent, parent_task.child_tasks.size());

        // to be called in case of failure to release all launched slots
        auto release_slots = [this, id_parent]() {
            for (auto & slot : slots) {
                if (slot.is_processing() && (
                        slot.task->id == id_parent ||
                        slot.task->id_parent == id_parent
                )) {
                    slot.release();
                }
            }
        };

        // launch all child tasks first
        size_t idx = 0;
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());
        for (auto * slot : child_slots) {
            // keep the state that the child task is about to overwrite
            if (prompt_cache && slot->prompt_save(*prompt_cache)) {
                prompt_cache->update();
            }

            server_task_id id_child = parent_task.child_tasks[idx].id;
            if (!launch_slot_with_task(*slot, std::move(parent_task.child_tasks[idx]))) {
                SRV_ERR("failed to launch slot with child task, id_task = %" PRId64 "\n", id_child);
                release_slots();
                return false;
            }
            idx++;
        }

        // finally, launch the parent task
        if (!launch_slot_with_task(parent_slot, std::move(parent_task))) {
            SRV_ERR("failed to launch slot with task, id_task = %" PRId64 "\n", id_parent);
            release_slots();
            return false;
        }

        return true;
    }

    // n_tokens_cur: the number of tokens added to the batch for the current slot
    void create_checkpoint(server_slot & slot, const int64_t n_tokens_cur, llama_pos pos_min, llama_pos pos_max) {
        dft_join(__LINE__);
        const auto checkpoint_begin = ggml_time_us();
        const server_task_id id_task = slot.task->id;

        // evict checkpoints within min-step of a previous checkpoint, unless they were
        // created by the current task
        // only when the list is full, otherwise short prompts keep just the oldest checkpoint
        int64_t last = -1;
        for (auto it = slot.prompt.checkpoints.begin();
                slot.prompt.checkpoints.size() + 1 >= (size_t) params_base.n_ctx_checkpoints &&
                it != slot.prompt.checkpoints.end(); ) {
            if (it->id_task != id_task && last >= 0 && it->n_tokens <= last + params_base.checkpoint_min_step) {
                SLT_TRC(slot, "erasing context checkpoint too close to an earlier one (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                        it->pos_min, it->pos_max, it->n_tokens, (float) it->size() / 1024 / 1024);

                ckpt_async_settle(&*it);
                ckpt_recycle_push(it->data_tgt);
                it = slot.prompt.checkpoints.erase(it);
                continue;
            }

            last = it->n_tokens;
            ++it;
        }

        while (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
            // make room for the new checkpoint, if needed
            const auto & cur = slot.prompt.checkpoints.front();

            SLT_WRN(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                    cur.pos_min, cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);

            ckpt_async_settle(&slot.prompt.checkpoints.front());
            ckpt_recycle_push(slot.prompt.checkpoints.front().data_tgt);
            slot.prompt.checkpoints.erase(slot.prompt.checkpoints.begin());
        }

        // replace an existing checkpoint at the same n_tokens instead of appending a duplicate
        {
            const int64_t n_tokens_new = slot.prompt.n_tokens() - n_tokens_cur;
            for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end(); ) {
                if (it->n_tokens == n_tokens_new) {
                    SLT_TRC(slot, "superseding context checkpoint at n_tokens = %" PRId64 "\n", it->n_tokens);
                    ckpt_async_settle(&*it);
                    ckpt_recycle_push(it->data_tgt);
                    it = slot.prompt.checkpoints.erase(it);
                } else {
                    ++it;
                }
            }
        }

        auto & cur = slot.prompt.checkpoints.emplace_back();

        cur.id_task = id_task;

        // [TAG_CHECKPOINTS_FIX_POS_MIN]
        // TODO: here we incorrectly deterimne that the saved checkpoint data covers the [pos_min, pos_max] range
        //       this is not true for SWA models: https://github.com/ggml-org/llama.cpp/pull/24411#issuecomment-4677983225
        cur.update_pos(slot.prompt.n_tokens() - n_tokens_cur, pos_min, pos_max);

        if (pipeline_on) {
            // Capture only this sequence without draining the pipeline: a stream-ordered async copy
            // into a pinned buffer, else a sequence-scoped fence and a pinned synchronous copy.
            if (slot.ctx_dft != nullptr) { ++checkpoint_mtp_seq_captures; }
            const llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
            const size_t size = llama_state_seq_get_size_ext(ctx_tgt, slot.id, flags);
            ckpt_recycle_take(cur.data_tgt, size);
            cur.data_tgt.resize(size);
            bool async_done = false;
            const int b = ckpt_async_acquire(size);
            if (b >= 0) {
                uint8_t * pin = ckpt_async_base(b);
                const int64_t hnd = llama_state_seq_get_data_async(ctx_tgt, pin, size, slot.id, flags);
                if (hnd < 0) { GGML_ABORT("async checkpoint capture failed (size = %zu)\n", size); }
#ifdef GGML_DIAGNOSTICS
                if (hnd > 0 && checkpoint_capture_verify) {
                    llama_synchronize(ctx_tgt);
                    std::vector<uint8_t> reference(size);
                    const size_t m = llama_state_seq_get_data_ext(ctx_tgt, reference.data(), size, slot.id, flags);
                    GGML_ASSERT(llama_state_seq_async_wait(ctx_tgt, hnd, true) == 1);
                    ++ckpt_async_verified;
                    if (m != size || memcmp(reference.data(), pin, size) != 0) {
                        ++ckpt_async_mismatches;
                        SLT_ERR(slot, "async checkpoint capture mismatch (size = %zu, reference = %zu)\n", size, m);
                    }
                    memcpy(cur.data_tgt.data(), pin, size);
                    ckpt_async_bufs[b].busy = false;
                } else
#endif
                if (hnd > 0) {
                    ckpt_async_items.push_back({slot.id, &cur, b, hnd, size});
                    ++ckpt_async_enqueued;
                } else {
                    memcpy(cur.data_tgt.data(), pin, size);
                    ckpt_async_bufs[b].busy = false;
                    ++ckpt_async_immediate;
                }
                async_done = true;
            }
            if (!async_done) {
                if (checkpoint_pinned == nullptr || ggml_backend_buffer_get_size(checkpoint_pinned) < size) {
                    if (checkpoint_pinned) { ggml_backend_buffer_free(checkpoint_pinned); }
                    checkpoint_pinned = ggml_backend_buft_alloc_buffer(pinned_host_buft(), size + size / 8);
                    GGML_ASSERT(checkpoint_pinned != nullptr);
                }
                uint8_t * capture_dst = (uint8_t *) ggml_backend_buffer_get_base(checkpoint_pinned);
                llama_state_seq_prepare_capture(ctx_tgt, slot.id);
                const size_t n = llama_state_seq_get_data_ext_prepared(ctx_tgt, capture_dst, size, slot.id, flags);
                if (n != size) {
                    GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", size, n);
                }
                memcpy(cur.data_tgt.data(), capture_dst, size);
                ++checkpoint_pinned_captures;
#ifdef GGML_DIAGNOSTICS
                if (checkpoint_capture_verify) {
                    std::vector<uint8_t> reference(size);
                    const size_t m = llama_state_seq_get_data_ext(ctx_tgt, reference.data(), size, slot.id, flags);
                    const bool match = m == size && reference == cur.data_tgt;
                    llama_state_seq_capture_record_verify(ctx_tgt, match);
                    if (!match) {
                        SLT_ERR(slot, "checkpoint capture mismatch (size = %zu, reference = %zu)\n", size, m);
                    }
                }
#endif
            }
        } else {
            cur.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        }
        cur.update_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        // stash the draft's speculative state with the checkpoint
        common_speculative_get_state(slot.spec, slot.id, cur.data_spec);
        ++prompt_checkpoint_saves;
        prompt_checkpoint_bytes += cur.size();
        prompt_checkpoint_save_us += ggml_time_us() - checkpoint_begin;

        SLT_TRC(slot,
                "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min,
                cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);
    }

    // checkpoints are appended to the slot save file, after the llama state payload
    // they cannot be recreated from the final state alone (a recurrent state cannot be rewound)
    static constexpr uint32_t SLOT_CKPT_MAGIC   = 0x504b4353; // "SCKP"
    static constexpr uint32_t SLOT_CKPT_VERSION = 1;

    static bool ckpt_read(std::ifstream & ifs, void * dst, size_t size, size_t & n_read) {
        if (!ifs.read((char *) dst, size)) {
            return false;
        }
        n_read += size;
        return true;
    }

    static bool ckpt_read_buf(std::ifstream & ifs, std::vector<uint8_t> & buf, size_t n_avail, size_t & n_read) {
        uint64_t n = 0;
        // check the size against the bytes left in the file before allocating, the size field may be corrupted
        if (!ckpt_read(ifs, &n, sizeof(n), n_read) || n > n_avail - n_read) {
            return false;
        }
        buf.resize(n);
        return n == 0 || ckpt_read(ifs, buf.data(), n, n_read);
    }

    static void ckpt_write(std::ofstream & ofs, const void * src, size_t size, size_t & n_written) {
        ofs.write((const char *) src, size);
        n_written += size;
    }

    static void ckpt_write_buf(std::ofstream & ofs, const std::vector<uint8_t> & buf, size_t & n_written) {
        const uint64_t n = buf.size();
        ckpt_write(ofs, &n, sizeof(n), n_written);
        if (n > 0) {
            ckpt_write(ofs, buf.data(), n, n_written);
        }
    }

    // returns false if the appendix could not be written completely
    bool save_slot_checkpoints(const std::string & filepath, const server_slot & slot, size_t & n_written) const {
        if (slot.ckpt_barrier) { slot.ckpt_barrier(slot.id); }
        n_written = 0;
        if (slot.prompt.checkpoints.empty()) {
            return true;
        }
        std::ofstream ofs(std::filesystem::u8path(filepath), std::ios::binary | std::ios::app);
        if (!ofs) {
            SLT_WRN(slot, "failed to append context checkpoints to '%s'\n", filepath.c_str());
            return false;
        }
        const uint32_t magic   = SLOT_CKPT_MAGIC;
        const uint32_t version = SLOT_CKPT_VERSION;
        const uint32_t count   = (uint32_t) slot.prompt.checkpoints.size();
        ckpt_write(ofs, &magic,   sizeof(magic),   n_written);
        ckpt_write(ofs, &version, sizeof(version), n_written);
        ckpt_write(ofs, &count,   sizeof(count),   n_written);
        for (const auto & cur : slot.prompt.checkpoints) {
            ckpt_write(ofs, &cur.n_tokens, sizeof(cur.n_tokens), n_written);
            ckpt_write(ofs, &cur.pos_min,  sizeof(cur.pos_min),  n_written);
            ckpt_write(ofs, &cur.pos_max,  sizeof(cur.pos_max),  n_written);
            ckpt_write_buf(ofs, cur.data_tgt,  n_written);
            ckpt_write_buf(ofs, cur.data_dft,  n_written);
            ckpt_write_buf(ofs, cur.data_spec, n_written);
        }
        ofs.flush();
        if (!ofs) {
            SLT_WRN(slot, "failed to append context checkpoints to '%s' - the appendix is incomplete\n", filepath.c_str());
            return false;
        }
        SLT_INF(slot, "appended %u context checkpoint(s) (%.3f MiB) to '%s'\n",
                count, (float) n_written / 1024 / 1024, filepath.c_str());
        return true;
    }

    // returns the number of bytes consumed, 0 if there is no usable appendix
    size_t load_slot_checkpoints(const std::string & filepath, size_t offset, server_slot & slot) const {
        if (slot.ckpt_barrier) { slot.ckpt_barrier(slot.id); }
        std::ifstream ifs(std::filesystem::u8path(filepath), std::ios::binary | std::ios::ate);
        const size_t file_size = ifs ? (size_t) ifs.tellg() : 0;
        if (!ifs || file_size < offset || !ifs.seekg(offset)) {
            return 0;
        }
        const size_t n_avail = file_size - offset; // bytes after the llama state payload
        size_t n_read = 0;
        uint32_t magic   = 0;
        uint32_t version = 0;
        uint32_t count   = 0;
        if (!ckpt_read(ifs, &magic, sizeof(magic), n_read) || magic != SLOT_CKPT_MAGIC) {
            return 0;
        }
        if (!ckpt_read(ifs, &version, sizeof(version), n_read) || version != SLOT_CKPT_VERSION ||
            !ckpt_read(ifs, &count,   sizeof(count),   n_read)) {
            SLT_WRN(slot, "invalid context checkpoint appendix in '%s' - ignored\n", filepath.c_str());
            return 0;
        }
        std::list<common_prompt_checkpoint> checkpoints;
        for (uint32_t i = 0; i < count; ++i) {
            common_prompt_checkpoint cur;
            cur.id_task = -1; // not created by a task - marks a checkpoint restored from a slot file
            if (!ckpt_read(ifs, &cur.n_tokens, sizeof(cur.n_tokens), n_read) ||
                !ckpt_read(ifs, &cur.pos_min,  sizeof(cur.pos_min),  n_read) ||
                !ckpt_read(ifs, &cur.pos_max,  sizeof(cur.pos_max),  n_read) ||
                !ckpt_read_buf(ifs, cur.data_tgt,  n_avail, n_read) ||
                !ckpt_read_buf(ifs, cur.data_dft,  n_avail, n_read) ||
                !ckpt_read_buf(ifs, cur.data_spec, n_avail, n_read)) {
                SLT_WRN(slot, "truncated context checkpoint appendix in '%s' - ignored\n", filepath.c_str());
                return 0;
            }
            // a saved checkpoint always holds a target state - an empty blob would roll back without restoring anything
            if (cur.data_tgt.empty()) {
                SLT_WRN(slot, "invalid context checkpoint appendix in '%s' - ignored\n", filepath.c_str());
                return 0;
            }
            checkpoints.push_back(std::move(cur));
            if (checkpoints.size() > (size_t) params_base.n_ctx_checkpoints) {
                checkpoints.pop_front();
            }
        }
        // the slot file does not check the draft context - test-load one draft checkpoint, drop the draft data if it does not fit
        dft_join(__LINE__);
        if (slot.ctx_dft != nullptr && !checkpoints.empty() && !checkpoints.back().data_dft.empty()) {
            const bool ok = checkpoints.back().load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_seq_rm(llama_get_memory(slot.ctx_dft), slot.id, -1, -1);
            if (!ok) {
                SLT_WRN(slot, "draft context checkpoint data in '%s' does not match the draft context - dropped\n", filepath.c_str());
                for (auto & cur : checkpoints) {
                    cur.clear_dft();
                }
            }
        }
        slot.prompt.checkpoints = std::move(checkpoints);
        SLT_INF(slot, "restored %zu context checkpoint(s) from '%s'\n", slot.prompt.checkpoints.size(), filepath.c_str());
        return n_read;
    }

    // the live draft state follows the checkpoints: the full draft context and the speculative state
    // the llama state payload holds the target context only
    static constexpr uint32_t SLOT_DRAFT_MAGIC   = 0x54464453; // "SDFT"
    static constexpr uint32_t SLOT_DRAFT_VERSION = 1;

    // returns false if the draft state could not be written completely
    bool save_slot_draft(const std::string & filepath, const server_slot & slot, size_t & n_written) const {
        dft_join(__LINE__);
        n_written = 0;
        std::vector<uint8_t> data_dft;
        std::vector<uint8_t> data_spec;
        if (slot.ctx_dft != nullptr) {
            data_dft.resize(llama_state_seq_get_size_ext(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_NONE));
            llama_state_seq_get_data_ext(slot.ctx_dft, data_dft.data(), data_dft.size(), slot.id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }
        common_speculative_get_state(slot.spec, slot.id, data_spec);
        if (data_dft.empty() && data_spec.empty()) {
            return true;
        }
        std::ofstream ofs(std::filesystem::u8path(filepath), std::ios::binary | std::ios::app);
        if (!ofs) {
            SLT_WRN(slot, "failed to append the draft state to '%s'\n", filepath.c_str());
            return false;
        }
        const uint32_t magic   = SLOT_DRAFT_MAGIC;
        const uint32_t version = SLOT_DRAFT_VERSION;
        ckpt_write(ofs, &magic,   sizeof(magic),   n_written);
        ckpt_write(ofs, &version, sizeof(version), n_written);
        ckpt_write_buf(ofs, data_dft,  n_written);
        ckpt_write_buf(ofs, data_spec, n_written);
        ofs.flush();
        if (!ofs) {
            SLT_WRN(slot, "failed to append the draft state to '%s' - the draft state is incomplete\n", filepath.c_str());
            return false;
        }
        SLT_INF(slot, "appended the draft state (%.3f MiB) to '%s'\n", (float) n_written / 1024 / 1024, filepath.c_str());
        return true;
    }

    // returns the number of bytes consumed, 0 if there is no usable draft state
    // the draft starts empty unless the file holds a draft state that fits the draft context
    size_t load_slot_draft(const std::string & filepath, size_t offset, server_slot & slot) const {
        dft_join(__LINE__);
        if (slot.ctx_dft != nullptr) {
            llama_memory_seq_rm(llama_get_memory(slot.ctx_dft), slot.id, -1, -1);
        }
        std::ifstream ifs(std::filesystem::u8path(filepath), std::ios::binary | std::ios::ate);
        const size_t file_size = ifs ? (size_t) ifs.tellg() : 0;
        if (!ifs || file_size < offset || !ifs.seekg(offset)) {
            return 0;
        }
        const size_t n_avail = file_size - offset;
        size_t n_read = 0;
        uint32_t magic   = 0;
        uint32_t version = 0;
        if (!ckpt_read(ifs, &magic, sizeof(magic), n_read) || magic != SLOT_DRAFT_MAGIC) {
            return 0;
        }
        std::vector<uint8_t> data_dft;
        std::vector<uint8_t> data_spec;
        if (!ckpt_read(ifs, &version, sizeof(version), n_read) || version != SLOT_DRAFT_VERSION ||
            !ckpt_read_buf(ifs, data_dft,  n_avail, n_read) ||
            !ckpt_read_buf(ifs, data_spec, n_avail, n_read)) {
            SLT_WRN(slot, "invalid draft state in '%s' - ignored\n", filepath.c_str());
            return 0;
        }
        if (!data_dft.empty()) {
            if (slot.ctx_dft == nullptr ||
                llama_state_seq_set_data_ext(slot.ctx_dft, data_dft.data(), data_dft.size(), slot.id, LLAMA_STATE_SEQ_FLAGS_NONE) != data_dft.size()) {
                if (slot.ctx_dft != nullptr) {
                    llama_memory_seq_rm(llama_get_memory(slot.ctx_dft), slot.id, -1, -1);
                }
                SLT_WRN(slot, "draft state in '%s' does not match the draft context - dropped\n", filepath.c_str());
                return n_read;
            }
        }
        if (!data_spec.empty()) {
            common_speculative_set_state(slot.spec, slot.id, data_spec);
        }
        SLT_INF(slot, "restored the draft state (%.3f MiB) from '%s'\n", (float) n_read / 1024 / 1024, filepath.c_str());
        return n_read;
    }

    // returns false to decline the task, it is offered again after the decode is done
    bool process_single_task(server_task && task, bool is_yielding) {
        // tasks read and modify slot, draft and speculative state directly; metrics/slot reads only read counters,
        // and the NEXT_RESPONSE wake-up posted on every scheduler pass does nothing
        if (task.type != SERVER_TASK_TYPE_METRICS && task.type != SERVER_TASK_TYPE_SLOT_GET &&
                task.type != SERVER_TASK_TYPE_NEXT_RESPONSE) { dft_join(__LINE__); }
        // while yielding, an encode / decode is running and only reading the server state is safe
        if (is_yielding && task.type != SERVER_TASK_TYPE_METRICS && task.type != SERVER_TASK_TYPE_SLOT_GET) {
            SRV_DBG("decoding, decline task, id_task = %" PRId64 "\n", task.id);
            return false;
        }

        if (task.type != SERVER_TASK_TYPE_METRICS && task.type != SERVER_TASK_TYPE_SLOT_GET &&
                task.type != SERVER_TASK_TYPE_NEXT_RESPONSE) {
            const auto admission_start = ggml_time_us();
            if (admission_drain_skippable(task)) {
                ++pipeline_admission_skipped;
            } else {
                pipeline_drain();
            }
            if (task.type == SERVER_TASK_TYPE_COMPLETION) {
                ++pipeline_admission_calls;
                pipeline_admission_drain_us += ggml_time_us() - admission_start;
            }
        }
        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
            case SERVER_TASK_TYPE_DECISION:
                {
                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    const server_task_id id_task = task.id;
                    if (wait_system_prefix(task)) {
                        queue_tasks.defer(std::move(task));
                        break;
                    }
                    const bool admission_skip = admission_drain_skippable(task);
                    if (!admission_skip) {
                        pipeline_drain();
                    }
                    if (params_base.vip_enable && task.params.priority > 0) {
                        pause_for_vip();
                    } else {
                        resume_paused();
                        if (params_base.vip_enable && active_ordinary() >= ordinary_budget()) {
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                    }
                    server_slot * slot = get_available_slot(task);
                    if (params_base.vip_enable && task.params.priority > 0 &&
                            (!slot || slot->is_processing()) && page_out_paused(task.id_slot)) {
                        slot = get_available_slot(task);
                    }
                    if (admission_skip && slot && task.params.cache_prompt &&
                            slot->prompt.tokens.get_common_prefix(task.tokens) > 0) {
                        // possible prompt-state reuse (checkpoint restore writes device state): keep the old drain
                        ++pipeline_admission_late_drains;
                        pipeline_drain();
                    }

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %" PRId64 "\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing() || (slot->borrowed_generation && task.params.priority <= 0)) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %" PRId64 "\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t n_child_tasks = task.child_tasks.size();
                        std::vector<server_slot *> child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %" PRId64 "\n", child_slots.size(), n_child_tasks, id_task);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            SRV_ERR("failed to launch slot with parent task, id_task = %" PRId64 "\n", id_task);
                            break; // drop the task
                        }
                    } else if (!launch_slot_with_task(*slot, std::move(task))) {
                        SRV_ERR("failed to launch slot with task, id_task = %" PRId64 "\n", id_task);
                        break; // drop the task
                    }

                    if (params_base.cache_idle_slots) {
                        for (auto & slot : slots) {
                            if (!slot.is_processing()) {
                                SLT_TRC(slot, "%s", "saving idle slot to prompt cache\n");

                                if (slot.prompt_save(*prompt_cache)) {
                                    SLT_DBG(slot, "%s", "__TEST_TAG_CACHE_IDLE_SLOT__\n");
                                    prompt_cache->update();
                                }

                                if (params_base.kv_unified) {
                                    // [TAG_IDLE_SLOT_CLEAR]
                                    slot.prompt_clear();
                                }
                            }
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    pipeline_drain();
                    for (auto it = suspended_requests.begin(); it != suspended_requests.end(); ++it) {
                        if (it->task_id == task.id_target) {
                            auto & current = slots.at(it->slot->id);
                            const auto parent = it->slot->borrowed_generation;
                            if (current.borrowed_generation == it->generation) { current.borrowed_generation = parent; }
                            for (auto & child : suspended_requests) {
                                if (child.slot->id == it->slot->id && child.slot->borrowed_generation == it->generation) {
                                    child.slot->borrowed_generation = parent;
                                }
                            }
                            it->slot->callback_on_reset(*it->slot);
                            suspended_requests.erase(it);
                            break;
                        }
                    }
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            slot.release();
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CONTROL:
                {
                    auto res = std::make_unique<server_task_result_control>();
                    res->id = task.id;

                    server_slot * slot = get_slot_by_cmpl_id(task.params.control_cmpl_id);
                    if (slot == nullptr) {
                        SRV_WRN("control %s on unknown completion id=%s, no live slot\n",
                                task.params.control_action.c_str(), task.params.control_cmpl_id.c_str());
                        res->success = false;
                        res->message = "no active completion for this id";
                        queue_results.send(std::move(res));
                        break;
                    }

                    if (task.params.control_action == "reasoning_end") {
                        // the budget sampler only exists when reasoning control was armed
                        if (!slot->task->params.sampling.reasoning_control) {
                            res->success = false;
                            res->message = "reasoning control not enabled for this completion";
                            queue_results.send(std::move(res));
                            break;
                        }
                        // act on the live slot mid generation, never defer
                        common_sampler_reasoning_budget_force(slot->smpl.get());
                        res->success = true;
                    } else {
                        res->success = false;
                        res->message = "unknown control action";
                    }

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        if (slot.is_processing()) {
                            n_processing_slots++;
                        }
                    }
                    SRV_DBG("n_processing_slots = %d\n", n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
                    res->metrics             = metrics;
                    const auto add_active = [&](const server_slot & slot) {
                        if (slot.stats.n_gen > 0) { res->n_predict_active += slot.stats.n_gen; }
                        else { res->n_prompt_active += slot.stats.n_prompt_processed; }
                    };
                    for (const auto & slot : slots) {
                        if (slot.is_processing()) { add_active(slot); }
                    }
                    for (const auto & saved : suspended_requests) { add_active(*saved.slot); }

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_GET:
                {
                    json slots_data = json::array();

                    int n_idle_slots = 0;

                    for (server_slot & slot : slots) {
                        if (!slot.is_processing()) {
                            n_idle_slots++;
                        }

                        auto data = slot.to_json(slots_debug == 0);
                        if (slots_data.empty()) {
                            data["pipeline_scheduler"] = {
                                {"settled", pipeline_profile_count},
                                {"ready_reorders", pipeline_ready_reorders},
                                {"prefill_lanes", pipeline_prefill_lanes}, {"mtp_prefill_lanes", pipeline_mtp_prefill_lanes}, {"prompt_overtakes", pipeline_prompt_overtakes}, {"test_prompt_hold_us", pipeline_test_prompt_hold_us},
                                {"balanced_lane_compaction", true}, {"compact_moves", pipeline_compact_moves},
                                {"compact_plan_calls", pipeline_compact_plan_calls},
                                {"compact_plan_cache_hits", pipeline_compact_plan_cache_hits},
                                {"compact_plan_us", pipeline_compact_plan_us},
                                {"compact_stream_exchange", true},
                                {"compact_physical_cost", true},
                                {"issue_us", pipeline_profile_issue_us},
                                {"wait_us", pipeline_profile_wait_us},
                                {"process_us", pipeline_profile_process_us},
                                {"issue_total_us", pipeline_issue_total_us}, {"issue_calls", pipeline_issue_calls},
                                {"drain_us", pipeline_drain_us}, {"drain_calls", pipeline_drain_calls},
                                {"drain_nonempty", pipeline_drain_nonempty},
                                {"admission_calls", pipeline_admission_calls}, {"admission_drain_us", pipeline_admission_drain_us}, {"admission_skipped", pipeline_admission_skipped}, {"admission_mtp_skipped", pipeline_admission_mtp_skipped}, {"vip_pause_skipped", pipeline_vip_pause_skipped}, {"checkpoint_pinned_captures", checkpoint_pinned_captures}, {"checkpoint_mtp_seq_captures", checkpoint_mtp_seq_captures}, {"checkpoint_holds", pipeline_checkpoint_holds}, {"admission_late_drains", pipeline_admission_late_drains},
                                {"prefill_cars", pipeline_prefill_cars}, {"decode_cars", pipeline_decode_cars},
                                {"mixed_cars", pipeline_mixed_cars},
                                {"prompt_payload", pipeline_prompt_payload}, {"decode_payload", pipeline_decode_payload},
                            };
                            json draft_histogram = json::object();
                            for (const auto & entry : pipeline_draft_batch_histogram) { draft_histogram[std::to_string(entry.first)] = entry.second; }
                            data["pipeline_scheduler"]["ready_draft_batch"] = {
                                {"enabled", pipeline_ready_draft_batch},
                                {"stepwise", pipeline_stepwise_drafts},
                                {"calls", pipeline_draft_batch_calls}, {"lanes", pipeline_draft_batch_lanes},
                                {"sequences", pipeline_draft_batch_sequences},
                                {"max_sequences", pipeline_draft_batch_max_sequences},
                                {"host_us", pipeline_draft_batch_us},
                                {"oom_isolations", pipeline_oom_isolations}, {"oom_terminated", pipeline_oom_terminated}, {"checkpoint_skipped_nocache", checkpoint_skipped_nocache}, {"draft_degrade_events", pipeline_draft_degrade_events}, {"draft_degraded_slots", pipeline_draft_degraded_slots}, {"draft_step_failures", pipeline_draft_step_failures}, {"draft_executor", dft_exec ? json{
                                    {"tasks", dft_exec->n_tasks}, {"task_us", dft_exec->task_us}, {"steps", dft_exec->n_steps}, {"step_us", dft_exec->step_us},
                                    {"joins", dft_exec->n_joins}, {"join_waits", dft_exec->n_join_waits}, {"join_wait_us", dft_exec->join_wait_us},
                                    {"catchups", dft_catchups}, {"direct_catchups", dft_direct_catchups}, {"copy_rows", dft_copy_rows}, {"copy_us", dft_copy_us},
                                    {"ring_waits", dft_ring_waits}} : json(nullptr)},
                                {"batch_size_histogram", draft_histogram},
                            };
                            if (dft_exec) {
                                json sites = json::object();
                                for (const auto & e : dft_join_sites) { sites[std::to_string(e.first)] = json::array({e.second.first, e.second.second}); }
                                data["pipeline_scheduler"]["ready_draft_batch"]["draft_executor"]["join_sites"] = sites;
                            }
                            data["pipeline_scheduler"]["output_tickets"] = {
                                {"retained", llama_decode_ticket_retained_count(ctx_tgt)},
                                {"in_flight", pipeline_fifo.size()},
                            };
                            std::vector<int> work_load(pipeline_count, 0);
                            for (const auto & current : slots) {
                                if (current.is_processing() && !current.resident_paused) {
                                    work_load[pipeline_lane_of(current.id)] += pipeline_work_weight(current);
                                }
                            }
                            const auto topk = common_sampler_get_direct_topk_stats();
                            data["pipeline_scheduler"]["direct_topk"] = {
                                {"calls", topk.calls},
                                {"nan_fallbacks", topk.nan_fallbacks}, {"input_candidates", topk.input_candidates},
                                {"kept_candidates", topk.kept_candidates}, {"selection_us", topk.selection_us},
                            };
                            data["pipeline_scheduler"]["speculative_sampler_backups"] = {
                                {"saved", speculative_sampler_backups},
                                {"elided", speculative_sampler_backup_elided},
                                {"save_us", speculative_sampler_backup_us},
                            };
                            data["pipeline_scheduler"]["verification_work"] = {
                                {"lane_rows", work_load},
                                {"physical_plan_calls", pipeline_work_physical_calls},
                                {"physical_plan_us", pipeline_work_physical_us},
                                {"physical_plan_improvements", pipeline_work_physical_improvements},
                                {"physical_plan_fallbacks", pipeline_work_physical_fallbacks},
                                {"moves", pipeline_work_moves}, {"fused_cars", pipeline_work_fused_cars},
                                {"mixed", pipeline_work_mixed},
                                {"issued_rounds", pipeline_work_issued}, {"deferrals", pipeline_work_deferrals},
                            };
                            data["pipeline_scheduler"]["context_checkpoints"] = {
                                {"save_calls", prompt_checkpoint_saves}, {"save_us", prompt_checkpoint_save_us},
                                {"serialized_bytes", prompt_checkpoint_bytes},
                            };
                            {
                                uint64_t fast = 0, full = 0, verified = 0, mismatches = 0;
                                llama_state_seq_capture_counts(ctx_tgt, &fast, &full, &verified, &mismatches);
                                data["pipeline_scheduler"]["context_checkpoints"]["seq_capture"] = {
                                    {"fast", fast}, {"full_sync", full}, {"verified", verified}, {"mismatches", mismatches},
                                };
                                uint64_t a_enq = 0, a_sync = 0, a_unmapped = 0, a_out = 0;
                                llama_state_seq_async_counts(ctx_tgt, &a_enq, &a_sync, &a_unmapped, &a_out);
                                data["pipeline_scheduler"]["context_checkpoints"]["async"] = {
                                    {"enqueued", ckpt_async_enqueued}, {"immediate", ckpt_async_immediate},
                                    {"verified", ckpt_async_verified}, {"mismatches", ckpt_async_mismatches},
                                    {"barriers", ckpt_async_barriers}, {"settled", ckpt_async_settled}, {"pool_waits", ckpt_async_pool_waits},
                                    {"pending", ckpt_async_items.size()}, {"buffers", ckpt_async_bufs.size()},
                                    {"recycle_put", ckpt_recycle_put}, {"recycle_hit", ckpt_recycle_hit}, {"recycle_miss", ckpt_recycle_miss}, {"recycle_pool", ckpt_recycle_pool.size()},
                                    {"lib_enqueued", a_enq}, {"lib_sync", a_sync}, {"lib_unmapped", a_unmapped}, {"lib_outstanding", a_out},
                                };
                            }
                            ggml_backend_input_snapshot_stats upload = {};
                            auto add_upload = [&](const ggml_backend_input_snapshot_stats & next) {
                                upload.enabled = upload.enabled || next.enabled;
                                upload.tensors += next.tensors; upload.bytes += next.bytes;
                                upload.fences += next.fences; upload.fence_wait_ns += next.fence_wait_ns;
                                upload.fallbacks += next.fallbacks; upload.allocated_bytes += next.allocated_bytes;
                            };
                            // The executor thread owns the draft context while it runs.
                            if (ctx_dft && !dft_exec) {
                                add_upload(llama_get_input_snapshot_stats(ctx_dft));
                            }
                            auto ubatch_json = [](llama_context * c) {
                                const auto u = llama_get_ubatch_stats(c);
                                return json{{"decodes", u.decodes}, {"ubatches", u.ubatches}, {"tokens", u.tokens}, {"seqs", u.seqs}};
                            };
                            data["pipeline_scheduler"]["ubatch_split"] = {
                                {"target", ubatch_json(ctx_tgt)},
                                {"draft", ctx_dft ? ubatch_json(ctx_dft) : json()},
                            };
                            {
                                const auto k = llama_get_sparse_kv_stats();
                                data["pipeline_scheduler"]["sparse_kv"] = {
                                    {"unmap_syncs", k.unmap_syncs}, {"sync_us", k.sync_us}, {"unmapped", k.unmapped},
                                    {"mapped", k.mapped}, {"recycled", k.recycled},
                                    {"rs_slots", k.rs_slots}, {"rs_moves", k.rs_moves}, {"rs_extra", k.rs_extra},
                                };
                            }
                            data["pipeline_scheduler"]["kv_stream_placement"] = {
                                {"swaps", pipeline_place_swaps}, {"kept", pipeline_place_kept},
                                {"refused", pipeline_place_refused}, {"lost", pipeline_place_lost},
                                {"us", pipeline_place_us}, {"lane_follow", pipeline_follow_moves},
                                {"defrag_passes", pipeline_defrag_passes}, {"defrag_moves", pipeline_defrag_moves},
                                {"defrag_us", pipeline_defrag_us}, {"defrag_refused", pipeline_defrag_refused},
                            };
                            data["pipeline_scheduler"]["draft_input_upload"] = {
                                {"enabled", upload.enabled}, {"tensors", upload.tensors}, {"bytes", upload.bytes},
                                {"fences", upload.fences}, {"fence_wait_ns", upload.fence_wait_ns},
                                {"fallbacks", upload.fallbacks}, {"allocated_bytes", upload.allocated_bytes},
                            };
                            data["system_prefix_cache"] = {
                                {"entries", system_prefixes.size()},
                                {"bytes", system_cache_bytes},
                                {"limit_bytes", system_cache_limit},
                                {"inplace", system_cache_inplace}, {"misses", system_cache_misses}, {"waits", system_cache_waits},
                                {"hits", system_cache_hits}, {"hit_tokens", system_cache_hit_tokens},
                                {"restore_us", system_cache_restore_total_us}, {"restore_failures", system_cache_restore_failures},
                                {"publishes", system_cache_publishes}, {"publish_us", system_cache_publish_total_us},
                                {"publish_failures", system_cache_publish_failures}, {"evictions", system_cache_evictions},
                                {"generated_output", metrics.n_predict_observed},
                                {"computed_input", metrics.n_prompt_observed}, {"cached_input", metrics.n_prompt_cached},
                            };
                        }
                        slots_data.push_back(std::move(data));
                    }
                    SRV_DBG("n_idle_slots = %d\n", n_idle_slots);

                    auto res = std::make_unique<server_task_result_slots>();
                    res->id           = task.id;
                    res->slots_data   = std::move(slots_data);
                    res->n_idle_slots = n_idle_slots;

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %" PRId64 "\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    std::vector<char> packed;
                    try {
                        packed = slot->prompt.tokens.serialize();
                    } catch (const std::exception & err) {
                        send_error(task, err.what(), ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }

                    GGML_ASSERT(packed.size() % sizeof(llama_token) == 0);
                    const size_t nwrite = llama_state_seq_save_file(
                        ctx_tgt, filepath.c_str(), slot->id,
                        reinterpret_cast<const llama_token *>(packed.data()), packed.size() / sizeof(llama_token));
                    if (nwrite == 0) {
                        send_error(task, "Unable to save slot", ERROR_TYPE_SERVER);
                        break;
                    }

                    size_t nwrite_ckpt = 0;
                    if (!save_slot_checkpoints(filepath, *slot, nwrite_ckpt)) {
                        send_error(task, "Unable to save slot: incomplete context checkpoints", ERROR_TYPE_SERVER);
                        break;
                    }

                    size_t nwrite_dft = 0;
                    if (!save_slot_draft(filepath, *slot, nwrite_dft)) {
                        send_error(task, "Unable to save slot: incomplete draft state", ERROR_TYPE_SERVER);
                        break;
                    }

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = true;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nwrite + nwrite_ckpt + nwrite_dft;
                    res->t_ms     = t_save_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %" PRId64 "\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    size_t nread = 0;
                    try {
                        size_t n_packed = 0;
                        llama_tokens packed;
                        nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, nullptr, 0, &n_packed);
                        if (nread != 0) {
                            packed.resize(std::max<size_t>(1, n_packed));
                            nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, packed.data(), packed.size(), &n_packed);
                        }
                        if (nread == 0) {
                            throw std::runtime_error("No available space in KV cache or invalid slot save file");
                        }
                        packed.resize(n_packed);

                        server_tokens restored = server_tokens::deserialize(packed, mctx != nullptr);

                        if (restored.size() > (size_t) slot->n_ctx) {
                            throw std::runtime_error("Restored prompt does not fit in the slot context");
                        }

                        if (!restored.validate(ctx_tgt)) {
                            throw std::runtime_error("Invalid tokens in slot save file");
                        }

                        slot->prompt.clear();
                        slot->prompt.tokens = std::move(restored);
                    } catch (const std::exception & err) {
                        slot->prompt_clear();
                        send_error(task, std::string("Unable to restore slot: ") + err.what(), ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }

                    // nread is the end offset of the llama state payload within the file
                    const size_t nread_ckpt = load_slot_checkpoints(filepath, nread, *slot);
                    const size_t nread_dft  = load_slot_draft(filepath, nread + nread_ckpt, *slot);

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nread + nread_ckpt + nread_dft;
                    res->t_ms     = t_restore_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %" PRId64 "\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    slot->prompt_clear();

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_GET_LORA:
                {
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    auto & loras = params_base.lora_adapters;
                    auto res = std::make_unique<server_task_result_get_lora>();
                    res->id = task.id;
                    for (size_t i = 0; i < loras.size(); ++i) {
                        auto & lora = loras[i];
                        std::string alora_invocation_string = "";
                        const uint64_t n_alora_tokens = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
                        llama_tokens alora_invocation_tokens;
                        if (n_alora_tokens) {
                            const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                            for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                                alora_invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                                alora_invocation_tokens.push_back(alora_tokens[j]);
                            }
                        }
                        res->loras.push_back(server_task_result_get_lora::lora{
                            lora,
                            alora_invocation_string,
                            alora_invocation_tokens,
                        });
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    auto new_loras = construct_lora_list(task.set_lora);
                    // logging
                    for (size_t i = 0; i < new_loras.size(); ++i) {
                        SRV_TRC("set lora adapter idx=%zu scale=%f\n", i, new_loras[i].scale);
                    }
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    params_base.lora_adapters = new_loras;
                    auto res = std::make_unique<server_task_result_apply_lora>();
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;
        }

        return true;
    }

    void iterate(std::vector<server_slot> & slots, std::function<void(server_slot &)> callback) {
        // KV stream order: batches then keep the ubatch order and their outputs need no reordering
        std::vector<std::pair<int, int>> order;
        order.reserve(slots.size());
        for (const auto & slot : slots) { order.emplace_back(pipeline_stream_of(slot.id), slot.id); }
        std::sort(order.begin(), order.end());
        for (const auto & [_, id] : order) {
            auto & slot = slots[id];
            if (slot.resident_paused) { continue; }
            if (pipeline_filter >= 0) {
                if (pipeline_settling) {
                    const auto & ids = pipeline_lanes[pipeline_filter].slot_ids;
                    if (std::find(ids.begin(), ids.end(), slot.id) == ids.end()) { continue; }
                } else if (slot.id != pipeline_prefill_slot) {
                    if (slot.pipeline_owner >= 0 && slot.pipeline_owner != pipeline_filter) { continue; }
                    if (slot.pipeline_owner < 0 && pipeline_lane_of(slot.id) != pipeline_filter) { continue; }
                }
            }
            try {
                callback(slot);
            } catch (const std::exception & e) {
                SLT_ERR(slot, "got exception: %s\n", e.what());
                send_error(slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    void iterate(std::vector<server_slot *> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(*slot);
            } catch (const std::exception & e) {
                SLT_ERR(*slot, "got exception: %s\n", e.what());
                send_error(*slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot->release();
            }
        }
    }

    void abort_all_slots(const std::string & reason) {
        for (auto & slot : slots) {
            if (slot.is_processing()) {
                send_error(slot, reason, ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    // @ngxson : for debugging only
    int64_t t_pre_decode  = 0;
    int64_t t_decode      = 0;
    int64_t t_post_decode = 0;
    int64_t t_sampl       = 0;
    int64_t n_pre_decode  = 0;
    int64_t n_decode      = 0;
    int64_t n_post_decode = 0;
    int64_t n_sampl       = 0;
// #define DEBUG_TIMINGS
#ifdef DEBUG_TIMINGS
    struct scoped_timer {
        int64_t & t;
        int64_t & n;
        int64_t t_start;
        scoped_timer(int64_t & t_, int64_t & n_) : t(t_), n(n_) {
            t_start = ggml_time_us();
        }
        ~scoped_timer() {
            t += ggml_time_us() - t_start;
            n++;
        }
    };
#else
    struct scoped_timer {
        scoped_timer(int64_t &, int64_t &) {}
        ~scoped_timer() {}
    };
#endif

    // Main thread. Copies existing counters only; skipped entirely unless a reader polled in the last 10 s.
    void dash_publish() {
        const int64_t now = ggml_time_us();
        if (now - dash_last_read_us.load(std::memory_order_relaxed) > 10 * 1000 * 1000) { return; }
        if (now - dash_last_pub_us < 200 * 1000) { return; }
        dash_last_pub_us = now;

        auto & d = dash_work;
        d.seq++;
        d.t_us = now;
        d.t_start_us = metrics.t_start;

        d.prompt  = metrics.n_prompt_observed;
        d.predict = metrics.n_predict_observed;
        d.cached  = metrics.n_prompt_cached;
        d.decodes = metrics.n_decode;
        // released requests are folded into metrics; running ones still hold their own draft counts
        d.draft_tokens = metrics.n_draft_tokens; d.draft_accepted = metrics.n_draft_accepted; d.draft_steps = metrics.n_draft_verif_steps;
        d.n_accepted_pos = (int32_t) std::min<size_t>(8, metrics.n_accepted_per_pos.size());
        for (int i = 0; i < 8; ++i) { d.accepted_pos[i] = i < (int) metrics.n_accepted_per_pos.size() ? metrics.n_accepted_per_pos[i] : 0; }
        const auto add_live = [&](const server_slot & slot) {
            d.draft_tokens += slot.stats.n_draft_tokens; d.draft_accepted += slot.stats.n_draft_accepted; d.draft_steps += slot.stats.n_draft_verif_steps;
            for (size_t i = 0; i < slot.n_accepted_per_pos.size() && i < 8; ++i) {
                d.accepted_pos[i] += slot.n_accepted_per_pos[i];
                d.n_accepted_pos = std::max<int32_t>(d.n_accepted_pos, (int32_t) i + 1);
            }
        };

        d.settled = pipeline_profile_count;
        d.cars_prefill = pipeline_prefill_cars; d.cars_decode = pipeline_decode_cars; d.cars_mixed = pipeline_mixed_cars;
        d.payload_prompt = pipeline_prompt_payload; d.payload_decode = pipeline_decode_payload;
        d.issue_us = pipeline_profile_issue_us; d.wait_us = pipeline_profile_wait_us;
        d.process_us = pipeline_profile_process_us; d.drain_us = pipeline_drain_us;
        d.admission_us = pipeline_admission_drain_us; d.admission_skipped = pipeline_admission_skipped;
        d.vip_pause_skipped = pipeline_vip_pause_skipped; d.compact_moves = pipeline_compact_moves;
        d.ckpt_holds = pipeline_checkpoint_holds; d.deferrals = pipeline_work_deferrals;
        d.oom_isolations = pipeline_oom_isolations; d.oom_terminated = pipeline_oom_terminated;
        d.draft_degrade = pipeline_draft_degrade_events; d.draft_failures = pipeline_draft_step_failures;
        if (dft_exec) {
            // join counters are written by this thread only
            d.dft_joins = dft_exec->n_joins; d.dft_join_waits = dft_exec->n_join_waits; d.dft_join_wait_us = dft_exec->join_wait_us;
        }

        d.sc_hits = system_cache_hits; d.sc_hit_tokens = system_cache_hit_tokens; d.sc_misses = system_cache_misses;
        d.sc_evictions = system_cache_evictions; d.sc_failures = system_cache_restore_failures; d.sc_bytes = system_cache_bytes;

        d.n_slots = (int32_t) slots.size();
        d.n_processing = d.n_prompt = d.n_gen = d.n_paused = d.n_vip = 0;
        d.n_rows = 0;
        llama_memory_t mem = llama_get_memory(ctx_tgt);
        for (const auto & slot : slots) {
            const bool busy = slot.is_processing();
            const int32_t cells = (int32_t) llama_memory_seq_n_cells(mem, slot.id);
            if (busy) {
                d.n_processing++;
                d.n_gen    += (slot.state == SLOT_STATE_GENERATING) && !slot.resident_paused;
                d.n_prompt += (slot.state == SLOT_STATE_STARTED || slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) && !slot.resident_paused;
                d.n_paused += slot.resident_paused;
                d.n_vip    += slot.task && slot.task->params.priority > 0;
                add_live(slot);
            }
            if (d.n_rows < (int32_t) (sizeof(d.rows) / sizeof(d.rows[0]))) {
                auto & r = d.rows[d.n_rows++];
                r.id = (int16_t) slot.id; r.state = (int16_t) slot.state;
                r.lane = (int16_t) pipeline_lane_of(slot.id);
                r.stream = (int16_t) llama_memory_seq_stream(mem, slot.id);
                r.n_tokens = busy ? (int32_t) slot.prompt.n_tokens() : 0;
                r.n_cells = cells;
                r.vip = busy && slot.task && slot.task->params.priority > 0;
                r.mtp = busy && slot.can_speculate();
                r.paused = slot.resident_paused; r.pad = 0;
            }
        }
        d.n_susp = 0; d.susp_bytes = 0;
        for (const auto & saved : suspended_requests) {
            add_live(*saved.slot);
            const uint64_t bytes = saved.target.size() + saved.draft.size();
            d.susp_bytes += bytes;
            if (d.n_susp < (int32_t) (sizeof(d.susp) / sizeof(d.susp[0]))) {
                auto & r = d.susp[d.n_susp++];
                r.id = (int16_t) saved.slot->id;
                r.vip = saved.slot->task && saved.slot->task->params.priority > 0;
                r.capacity = saved.capacity_page;
                r.n_tokens = (int32_t) saved.slot->prompt.n_tokens();
                r.bytes = bytes;
            }
        }
        d.n_suspended = (int32_t) suspended_requests.size();
        d.n_deferred  = (int32_t) queue_tasks.queue_tasks_deferred_size();
        d.kv_used = resident_tokens();
        if (d.kv_used > dash_kv_max) dash_kv_max = d.kv_used;
        d.kv_max  = dash_kv_max;
        d.kv_cap  = (uint64_t) n_ctx;
        for (int i = 0; i < 64; ++i) { d.ctx_hist[i] = dash_ctx_hist[i]; }

        // allocator sizes every 2 s; the draft context only while its worker is provably idle
        if (now - d.mem_t_us > 2 * 1000 * 1000) {
            std::unique_lock<std::mutex> dft_lock;
            bool dft_idle = !dft_exec;
            if (dft_exec) {
                dft_lock = std::unique_lock<std::mutex>(dft_exec->mutex, std::try_to_lock);
                dft_idle = dft_lock.owns_lock() && dft_exec->queue.empty() && !dft_exec->busy;
            }
            if (dft_idle) {
                std::map<std::string, paraflow_dash_dev> devs;
                const auto add = [&](ggml_backend_buffer_type_t buft, size_t model_b, size_t ctx_b, size_t comp_b) {
                    if (!buft || ggml_backend_buft_is_host(buft)) { return; }
                    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
                    const std::string name = dev ? ggml_backend_dev_name(dev) : ggml_backend_buft_name(buft);
                    auto & e = devs[name];
                    e.model += model_b; e.context += ctx_b; e.compute += comp_b;
                };
                for (const auto & [buft, mb] : llama_get_memory_breakdown(ctx_tgt)) { add(buft, mb.model, mb.context, mb.compute); }
                for (const auto & [buft, sz] : llama_get_pipeline_aux_memory_breakdown(ctx_tgt)) { add(buft, 0, 0, sz); }
                if (ctx_dft) {
                    const bool shared = llama_get_model(ctx_dft) == llama_get_model(ctx_tgt);
                    for (const auto & [buft, mb] : llama_get_memory_breakdown(ctx_dft)) { add(buft, shared ? 0 : mb.model, mb.context, mb.compute); }
                    for (const auto & [buft, sz] : llama_get_pipeline_aux_memory_breakdown(ctx_dft)) { add(buft, 0, 0, sz); }
                }
                d.n_dev = 0;
                for (auto & [name, e] : devs) {
                    if (d.n_dev >= 16) { break; }
                    auto & o = d.dev[d.n_dev++];
                    snprintf(o.name, sizeof(o.name), "%s", name.c_str());
                    o.model = e.model; o.context = e.context; o.compute = e.compute;
                    const auto it = dash_dev_total.find(name);
                    o.total = it != dash_dev_total.end() ? it->second : 0;
                }
                d.mem_t_us = now;
            }
        }

        if (dash_mutex.try_lock()) { // a reader holds it only for one struct copy; skip this round otherwise
            dash_pub = d;
            dash_mutex.unlock();
        }
    }

    std::map<std::string, uint64_t> dash_dev_total;

    paraflow_dash_bucket & dash_hist_bucket(int64_t k) {
        auto & b = dash_hist[k % PARAFLOW_DASH_BUCKETS];
        if (b.id.load(std::memory_order_relaxed) != (uint32_t) k) { // recycle a bucket last written 6 h ago (or never)
            b.id.store(UINT32_MAX, std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_release);
            for (auto * f : { &b.dp, &b.dd, &b.dd_pure, &b.ng_ms, &b.np_ms, &b.pure_ms, &b.mix_ms }) { f->store(0, std::memory_order_relaxed); }
            b.id.store((uint32_t) k, std::memory_order_release);
        }
        return b;
    }

    // Main thread, after every update_slots() pass. Tokens counted since the previous pass are spread over the
    // time they took: from the end of the previous pass if work was in flight then, else from the start of this
    // pass (an idle loop does not run, so idle time stays empty). Request counts use the state after the pass.
    void dash_hist_record(int64_t t_enter) {
        const int64_t z = paraflow_wall_ms();
        const uint64_t dp = metrics.n_prompt_observed - dash_hist_prompt, dd = metrics.n_predict_observed - dash_hist_predict;
        dash_hist_prompt = metrics.n_prompt_observed; dash_hist_predict = metrics.n_predict_observed;
        uint64_t ng = 0, np = 0;
        bool busy = !suspended_requests.empty();
        for (const auto & slot : slots) {
            if (!slot.is_processing()) { continue; }
            busy = true;
            if (slot.resident_paused) { continue; }
            ng += slot.state == SLOT_STATE_GENERATING;
            np += slot.state == SLOT_STATE_STARTED || slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT;
        }
        int64_t a = dash_hist_prev_busy ? dash_hist_prev_ms : t_enter;
        dash_hist_prev_busy = busy; dash_hist_prev_ms = z;
        if (!dp && !dd && !ng && !np) { return; }
        a = std::max(std::min(a, z), z - PARAFLOW_DASH_BUCKETS * PARAFLOW_DASH_BUCKET_MS);
        const int64_t span = z - a;
        const bool pure = dd > 0 && dp == 0, mixed = dd > 0 && dp > 0;
        const auto add = [](std::atomic<uint32_t> & f, uint64_t v) { f.store(f.load(std::memory_order_relaxed) + (uint32_t) v, std::memory_order_relaxed); };
        uint64_t left_p = dp, left_d = dd;
        for (int64_t k = a / PARAFLOW_DASH_BUCKET_MS; ; ++k) {
            const int64_t bs = k * PARAFLOW_DASH_BUCKET_MS, be = bs + PARAFLOW_DASH_BUCKET_MS;
            const bool last = span == 0 || be >= z;
            const uint64_t ov = span == 0 ? 0 : (uint64_t) (std::min(z, be) - std::max(a, bs));
            const uint64_t sp = last ? left_p : dp * ov / (uint64_t) span, sd = last ? left_d : dd * ov / (uint64_t) span;
            left_p -= sp; left_d -= sd;
            auto & b = dash_hist_bucket(k);
            add(b.dp, sp); add(b.dd, sd); add(b.ng_ms, ng * ov); add(b.np_ms, np * ov);
            if (pure) { add(b.dd_pure, sd); add(b.pure_ms, ov); } else if (mixed) { add(b.mix_ms, ov); }
            if (last) { break; }
        }
    }

    // once at startup, before the serving loop: fixed run configuration and device totals
    std::string dash_make_config() {
        const llama_model * model = llama_get_model(ctx_tgt);
        json devices = json::array();
        for (int i = 0; i < llama_model_n_devices(model); ++i) {
            ggml_backend_dev_t dev = llama_model_get_device(model, i);
            if (!dev) { continue; }
            size_t free_b = 0, total_b = 0;
            ggml_backend_dev_memory(dev, &free_b, &total_b);
            dash_dev_total[ggml_backend_dev_name(dev)] = total_b;
            devices.push_back({{"name", ggml_backend_dev_name(dev)}, {"desc", ggml_backend_dev_description(dev)}, {"total", total_b}});
        }
        std::string path = params_base.model.path;
        const auto slash = path.find_last_of('/');
        if (slash != std::string::npos) { path = path.substr(slash + 1); }
        const auto & spec = params_base.speculative;
        const bool mtp = std::find(spec.types.begin(), spec.types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != spec.types.end();
        return safe_json_to_str({
            {"build", llama_build_info()}, {"model", path},
            {"n_parallel", params_base.n_parallel}, {"n_ctx", n_ctx}, {"n_ctx_slot", slots.empty() ? 0 : slots[0].n_ctx},
            {"n_batch", params_base.n_batch}, {"n_ubatch", params_base.n_ubatch},
            {"pipe_lanes", pipeline_count}, {"prefill_lanes", pipeline_prefill_lanes},
            {"prefill_chunk", params_base.pipe_prefill_chunk}, {"pf_inflight", params_base.pipe_pf_inflight},
            {"system_cache_mib", params_base.system_cache_mib},
            {"kv_dynamic", params_base.kv_dynamic}, {"kv_unified", params_base.kv_unified}, {"kv_spill", params_base.kv_spill},
            {"vip", params_base.vip_enable},
            {"cache_type_k", ggml_type_name(params_base.cache_type_k)}, {"cache_type_v", ggml_type_name(params_base.cache_type_v)},
            {"mtp", mtp}, {"draft_n_max", spec.draft.n_max},
            {"devices", devices},
        });
    }

    void update_slots() {
        if (params_base.dash) { dash_publish(); }
        if (!ckpt_async_items.empty()) { ckpt_async_poll(-1, false); }
        resume_paused();
        ensure_generation_capacity();
        pipeline_rebalance();
        pipeline_refresh_work_rounds();
#ifdef DEBUG_TIMINGS
        static int64_t t_prev = 0;
        int64_t t_start = ggml_time_us();
        if (t_start - t_prev > 5 * 1000 * 1000) { // every 5 seconds
            t_prev = t_start;
            SRV_INF("n_pre_decode      = %" PRId64 "\n", n_pre_decode);
            SRV_INF("avg t_pre_decode  = %f ms\n", (double) t_pre_decode / n_pre_decode / 1000.0);
            SRV_INF("avg t_decode      = %f ms\n", (double) t_decode / n_decode / 1000.0);
            SRV_INF("avg t_post_decode = %f ms\n", (double) t_post_decode / n_post_decode / 1000.0);
            SRV_INF("avg t_sampl       = %f ms\n", (double) t_sampl / n_sampl / 1000.0);
        }
#endif

        // check if all slots are idle
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing()) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_TRC("%s", "all slots are idle\n");
                dft_join(__LINE__);

                metrics_flush_idle();

                return; // skip further processing

            } else {
                queue_tasks.request_update();
            }
        }

        try {
            try {
                if (pipeline_can_run()) {
                    struct route_guard { bool & r; route_guard(bool & r) : r(r) { r = true; } ~route_guard() { r = false; } } route(dft_route);
                    pipeline_step();
                    return;
                }
                pipeline_drain();
            } catch (const pipeline_car_failure & f) {
                // settle every other in-flight car, then terminate only the failed car's requests
                ++pipeline_oom_isolations;
                SRV_WRN("isolating target car memory failure %d: terminating %zu request(s), draining the rest\n", f.ret, f.slot_ids.size());
                pipeline_drain();
                llama_synchronize(ctx_tgt);
                if (ctx_dft) { llama_synchronize(ctx_dft); }
                for (const int id : f.slot_ids) {
                    auto & slot = slots.at(id);
                    slot.pipeline_owner = -1;
                    if (!slot.is_processing()) { continue; }
                    send_error(slot, "insufficient GPU memory to continue this request (decode status " + std::to_string(f.ret) + "); other requests continue", ERROR_TYPE_SERVER);
                    slot.prompt_clear();
                    slot.release();
                    ++pipeline_oom_terminated;
                }
                return;
            }
        } catch (const std::exception & e) {
            llama_synchronize(ctx_tgt);
            pipeline_fifo.clear();
            for (auto & lane : pipeline_lanes) {
                if (lane.ticket) { llama_decode_ticket_release(ctx_tgt, lane.ticket); }
                lane.ticket = 0;
                lane.in_flight = false;
                lane.shared_draft_staged = lane.shared_draft_ready = false;
            }
            for (auto & slot : slots) { slot.pipeline_owner = -1; }
            pipeline_filter = -1;
            pipeline_prefill_slot = -1;
            pipeline_settling = false;
            llama_decode_ticket_select(ctx_tgt, 0);
            abort_all_slots(std::string("pipeline failed: ") + e.what());
            return;
        }

        try {
            scoped_timer t(t_pre_decode, n_pre_decode);
            pre_decode();
        } catch (const std::exception & e) {
            SRV_ERR("pre_decode() failed: %s\n", e.what());
            abort_all_slots("pre_decode() failed: " + std::string(e.what()));

            // the batch is half-built and not rendered, skip now to avoid UB
            return;
        }

        GGML_ASSERT(batch.slot_batched || batch.size() == 0);

        if (batch.slot_batched) {
            auto & slot_batched      = batch.slot_batched;
            auto & alora_scale       = batch.alora_scale;
            auto & alora_disabled_id = batch.alora_disabled_id;

            // TODO @ngxson : alora handling is too messy, need to refactor it to be more clear and maintainable
            // apply lora, only need to do it once per batch
            common_set_adapter_lora(ctx_tgt, slot_batched->lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            llama_set_embeddings(ctx_tgt, slot_batched->need_embd());
        }

        int32_t off_next = 0;
        int32_t n_batch = llama_n_batch(ctx_tgt);
        for (int32_t off = 0; off < batch.size(); off = off_next) {
            const int32_t n_tokens = std::min(n_batch, batch.size() - off);
            try {
                scoped_timer t(t_decode, n_decode);
                // TODO @ngxson : maybe handle n_batch == 1 here instead of inside decode()

                batch.render(off, n_tokens);
                bool ok = decode(n_batch, off);
#ifdef DEBUG_TIMINGS
                llama_synchronize(ctx_tgt);
#endif

                if (ok) {
                    // move the head of the batch forward with the number of tokens we just processed
                    off_next = off + n_tokens;

                    // on successful decode, restore the original batch size
                    n_batch = llama_n_batch(ctx_tgt);
                } else {
                    // try again with the updated n_batch
                    continue;
                }
            } catch (const std::exception & e) {
                SRV_ERR("decode() failed: %s\n", e.what());
                abort_all_slots("decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }

            try {
                scoped_timer t(t_post_decode, n_post_decode);
                post_decode(n_tokens, off);
                llama_decode_ticket_select(ctx_tgt, 0);
            } catch (const std::exception & e) {
                llama_decode_ticket_select(ctx_tgt, 0);
                SRV_ERR("post_decode() failed: %s\n", e.what());
                abort_all_slots("post_decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }
        }
    }

    void prepare_decode_generations(std::vector<server_slot *> & generating, std::vector<server_slot *> & drafting) {
        struct dft_item { common_speculative * spec; int id; bool set; common_speculative_draft_params params; };
        std::vector<dft_item> dft_items;
        if (dft_exec) {
            std::vector<std::tuple<common_speculative *, int, bool>> mask;
            for (const auto & slot : slots) { mask.emplace_back(slot.spec, slot.id, slot.can_speculate()); }
            dft_run([mask = std::move(mask)]() {
                for (const auto & m : mask) { common_speculative_set_enabled(std::get<0>(m), std::get<1>(m), std::get<2>(m)); }
            });
        } else {
            for (const auto & slot : slots) {
                common_speculative_set_enabled(slot.spec, slot.id, slot.can_speculate());
            }
        }
        // apply context-shift if needed
        // TODO: simplify and improve
        iterate(slots, [&](server_slot & slot) {
            if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                if (mctx) {
                    // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                    // we don't support ctx_shift because an image chunk may contains multiple tokens
                    GGML_ABORT("not supported by multimodal");
                }

                if (slot.task->is_parent() || slot.task->is_child()) {
                    send_error(slot, "context shift cannot be used for shared prompt", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                // Shift context
                int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;

                if (add_bos_token) {
                    n_keep += 1;
                }

                n_keep = std::min(slot.n_ctx - 4, n_keep);

                const int n_left    = slot.prompt.n_tokens() - n_keep;
                int       n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                // ref: https://github.com/ggml-org/llama.cpp/pull/24786
                n_discard = std::clamp(n_discard, 0, std::max(0, n_left - 1));

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                dft_join(__LINE__);
                slot.mem.seq_rm (slot.id, n_keep            , n_keep + n_discard);
                slot.mem.seq_add(slot.id, n_keep + n_discard, slot.prompt.tokens.pos_next(), -n_discard);

                // add generated tokens to cache
                // ref: https://github.com/ggml-org/llama.cpp/pull/16818#discussion_r2473269481
                {
                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                    llama_tokens new_tokens = slot.prompt.tokens.get_tokens(); // copy
                    for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                        new_tokens[i - n_discard] = new_tokens[i];
                    }

                    new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                    slot.prompt.clear();
                    slot.prompt.tokens.insert(new_tokens);
                }

                slot.truncated = true;
            }
        });

        // start populating the batch for this iteration
        batch.clear();

        // track if given slot can be batched with slots already in the batch
        auto & slot_batched = batch.slot_batched;

        generating.clear();
        drafting.clear();

        // determine which slots are generating and drafting
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            // check if we can batch this slot with the previous one
            if (!slot_batched) {
                slot_batched = &slot;
            } else if (!slot_batched->can_batch_with(slot)) {
                return;
            }

            generating.push_back(&slot);

            if (spec) {
                if (dft_exec) { dft_items.push_back({ slot.spec, slot.id, false, {} }); }
                else { common_speculative_get_draft_params(slot.spec, slot.id).drafting = false; }

                const bool use_ckpt_tgt = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
                const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

                const int n_draft_max = slot.get_n_draft_max();

                if (n_draft_max > 0) {
                    GGML_ASSERT(slot.can_speculate());

                    // stale candidates: a replay never reads them, a new draft refills them
                    slot.spec_draft_q.clear();

                    if (!slot.spec_draft.empty()) {
                        // we have a previous (partial) draft to reuse
                        if (use_ckpt_tgt) {
                            GGML_ASSERT(!slot.spec_ckpt.empty());
                        }
                    } else {
                        GGML_ASSERT(slot.spec_i_batch.empty());

                        slot.spec_ckpt.update_pos(
                                slot.prompt.n_tokens(),
                                llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id),
                                llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id));

                        if (use_ckpt_dft) {
                            slot.spec_ckpt.update_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        }

                        slot.spec_prompt = slot.prompt.tokens.get_text_tokens();

                        const bool spec_reject = slot.use_spec_rejection();

                        const common_speculative_draft_params dft_params = {
                            /* .drafting = */ true,
                            /* .n_max    = */ n_draft_max,
                            /* .pos0     = */ slot.prompt.tokens.pos_next(),
                            /* .id_last  = */ slot.sampled,
                            /* .prompt   = */ &slot.spec_prompt,
                            /* .result   = */ &slot.spec_draft,
                            /* .result_q = */ spec_reject ? &slot.spec_draft_q : nullptr,
                            /* .temp     = */ slot.task->params.sampling.temp,
                            /* .seed     = */ slot.task->params.sampling.seed,
                        };
                        if (dft_exec) { dft_items.push_back({ slot.spec, slot.id, true, dft_params }); }
                        else { common_speculative_get_draft_params(slot.spec, slot.id) = dft_params; }

                        drafting.push_back(&slot);
                    }
                }
            }
        });

        if (dft_exec && !dft_items.empty()) {
            const bool async = dft_route;
            if (async) {
                for (const auto & it : dft_items) { if (it.set) { dft_done[it.id].store(false, std::memory_order_relaxed); } }
            }
            dft_run([this, items = std::move(dft_items), async]() {
                for (const auto & it : items) {
                    auto & dp = common_speculative_get_draft_params(it.spec, it.id);
                    if (it.set) { dp = it.params; } else { dp.drafting = false; }
                    if (async && it.set) { dft_w_active.push_back(it.id); }
                }
                if (async && !dft_w_active.empty()) { dft_exec->pumping = true; }
            });
        }
    }

    void finish_drafts(std::vector<server_slot *> & drafting) {
        // make checkpoints if needed
        iterate(drafting, [&](server_slot & slot) {
            auto & draft = slot.spec_draft;
            auto & ckpt  = slot.spec_ckpt;

            slot.stats.n_draft_tokens += draft.size();

            // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
            const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

            if (slot.ctx_dft) {
                if (use_ckpt_dft) {
                    GGML_ASSERT(ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                }

                dft_run([ctx = slot.ctx_dft, id = slot.id, p0 = ckpt.pos_max + 1]() {
                    if (!llama_memory_seq_rm(llama_get_memory(ctx), id, p0, -1)) {
                        GGML_ABORT("failed to remove sequence %d\n", id);
                    }
                });
            }

            if (!draft.empty()) {
                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                   (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_tgt));

                const bool use_ckpt_dft =
                   (ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(slot.ctx_dft));

                if (use_ckpt_tgt) {
                    //const int64_t t_start = ggml_time_us();

                    ckpt.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                    //const int64_t t_total = ggml_time_us() - t_start;
                    //printf("checkpoint total: %f ms\n", t_total / 1000.0);

                    SLT_DBG(slot, "created speculative checkpoint (pos_min = %d, pos_max = %d, n_tokens = %d, size = %.3f MiB, draft = %.3f MiB)\n",
                            ckpt.pos_min, ckpt.pos_max, slot.prompt.n_tokens(),
                            (float) ckpt.size() / 1024 / 1024,
                            (float) ckpt.data_dft.size() / 1024 / 1024);
                }

                if (use_ckpt_dft) {
                    ckpt.update_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }
        });

    }

    void finish_pre_decode(std::vector<server_slot *> & generating) {
        auto & slot_batched = batch.slot_batched;
        // update the batch with the sampled/drafted tokens
        iterate(generating, [&](server_slot & slot) {
            slot.handle_last_sampled_token(batch);
        });

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx_tgt);
        int32_t n_ubatch = llama_n_ubatch(ctx_tgt);
        if (pipeline_filter >= 0 && pipeline_prefill_slot >= 0) {
            n_batch = std::min(n_batch, batch.size() + std::max(1, params_base.pipe_prefill_chunk));
        }

        if (params_base.kv_spill && (params_base.kv_dynamic || params_base.kv_unified)) {
            uint64_t generated_cells = 0;
            std::map<int, llama_pos> maxima;
            for (const auto & token : batch.tokens) {
                maxima[token.id_slot] = std::max(maxima.count(token.id_slot) ? maxima[token.id_slot] : -1, token.pos[0]);
            }
            for (const auto & entry : maxima) {
                const auto previous = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), entry.first);
                generated_cells += std::max(0, entry.second - previous);
            }
            const uint64_t used = resident_tokens() + generated_cells;
            const int available = used < (uint64_t) n_ctx ? (int) ((uint64_t) n_ctx - used) : 0;
            n_batch = std::min(n_batch, batch.size() + available);
        }
        auto & alora_scale       = batch.alora_scale;
        auto & alora_disabled_id = batch.alora_disabled_id;

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.size() == 0) {
            bool add_ok = true; // false means the batch is full, skip remaining slots

            iterate(slots, [&](server_slot & slot) {
                if (!add_ok || batch.size() >= n_batch) {
                    return; // batch is full, skip remaining slots
                }

                if (!slot.is_processing()) {
                    return;
                }

                // check if we can batch this slot with the previous one
                if (slot_batched && !slot_batched->can_batch_with(slot)) {
                    return;
                }

                // check if this is a child slot
                if (slot.state == SLOT_STATE_WAIT_OTHER) {
                    SLT_DBG(slot, "%s", "waiting for parent slot to complete\n");
                    return;
                }

                if (pipeline_filter >= 0 && slot.id != pipeline_prefill_slot) { return; }
                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    const auto & input_tokens = slot.task->tokens;

                    // used to determine the number of tokens added to the batch for the current slot
                    const auto n_tokens_prev = batch.size();

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        dft_join(__LINE__); // cache reuse and checkpoint restore touch the draft state directly
                        slot.stats.update_prompt_start();

                        slot.state = SLOT_STATE_PROCESSING_PROMPT;

                        SLT_TRC(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n",
                                slot.n_ctx, slot.task->params.n_keep, slot.task->n_tokens());

                        // print prompt tokens (for debugging)
                        /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        }*/

                        // keep track how many tokens we can reuse from the previous state
                        int n_past = 0;

                        // empty prompt passed -> release the slot and send empty response
                        if (input_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            dft_join(__LINE__);
                            slot.print_timings();
                            send_final_response(slot);
                            slot.release();

                            return;
                        }

                        // TODO: support memory-less logits computation
                        if (slot.task->need_logits() && !llama_get_memory(ctx_tgt)) {
                            send_error(slot, "the current context does not support logits computation. skipping", ERROR_TYPE_SERVER);
                            slot.release();
                            return;
                        }

                        if (!slot.can_split()) {
                            if (slot.task->n_tokens() > n_ubatch) {
                                send_error(slot,
                                           string_format(
                                               "input (%d tokens) is too large to process. increase the physical batch "
                                               "size (current batch size: %d)",
                                               slot.task->n_tokens(), n_ubatch),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                return;
                            }

                            if (slot.task->n_tokens() > slot.n_ctx) {
                                send_error(
                                    slot,
                                    string_format(
                                        "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                        slot.task->n_tokens(), slot.n_ctx),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }
                        } else {
                            if (slot.task->n_tokens() >= slot.n_ctx) {
                                send_error(slot,
                                           string_format("request (%d tokens) exceeds the available context size (%d "
                                                         "tokens), try increasing it",
                                                         slot.task->n_tokens(), slot.n_ctx),
                                           ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }

                            // the outputs of a decision are read from one batch
                            const int32_t n_decision_first = slot.task->type == SERVER_TASK_TYPE_DECISION ? slot.task->decision.pos_first() : -1;
                            if (n_decision_first >= 0 && slot.task->n_tokens() - n_decision_first > n_batch) {
                                send_error(slot,
                                           string_format("the question and its options (%d tokens) are too large to process. "
                                                         "increase the batch size (current batch size: %d)",
                                                         slot.task->n_tokens() - n_decision_first, n_batch),
                                           ERROR_TYPE_INVALID_REQUEST);
                                slot.release();
                                return;
                            }

                            if (slot.task->params.cache_prompt && !slot.task->is_stateless()) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);

                                // the children start from the shared prefix, do not go past it
                                if (slot.task->n_tokens_shared > 0) {
                                    n_past = std::min(n_past, slot.task->n_tokens_shared);
                                }

                                // the outputs of a decision are read from the batch, so its first position must be evaluated
                                if (n_decision_first >= 0) {
                                    n_past = std::min(n_past, n_decision_first);
                                }

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", n_past, slot.alora_invocation_start);
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

                                const auto n_cache_reuse = slot.task->params.n_cache_reuse;

                                const bool can_cache_reuse =
                                    llama_memory_can_shift(llama_get_memory(ctx_tgt)) &&
                                    !slot.prompt.tokens.has_mtmd;

                                if (!can_cache_reuse && n_cache_reuse > 0) {
                                    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", n_cache_reuse);
                                }

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (can_cache_reuse && n_cache_reuse > 0) {
                                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                                    size_t head_c = n_past; // cache
                                    size_t head_p = n_past; // current prompt

                                    if (mctx) {
                                        // we should never reach this
                                        GGML_ABORT("not supported by multimodal");
                                    }

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, n_past = %d\n", n_cache_reuse, n_past);

                                    while (head_c < slot.prompt.tokens.size() &&
                                           head_p < input_tokens.size()) {

                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.prompt.tokens.size() &&
                                               head_p + n_match < input_tokens.size()       &&
                                               slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t) n_cache_reuse) {
                                            SLT_TRC(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx_tgt, prompt_tokens[i]).c_str());
                                            //}

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            slot.mem.seq_rm (slot.id, head_p, head_c);
                                            slot.mem.seq_add(slot.id, head_c, head_c + n_match, kv_shift);

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.prompt.tokens.set_token(head_p + i, slot.prompt.tokens[head_c + i]);
                                                n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new n_past = %d\n", n_past);
                                }
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                            }

                            llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);

                            // ref: https://github.com/ggml-org/llama.cpp/pull/24110
                            const bool has_new_tokens = (n_past < slot.task->n_tokens());

                            // the first evaluated token attends to the positions > pos_eval - n_swa, so a SWA state
                            // starting at pos_eval - n_swa + 1 is enough, while a state without SWA must end before pos_eval
                            const llama_pos pos_eval = pos_next - (has_new_tokens ? 0 : 1);

                            // a state or a checkpoint is useful when its pos_min is below this threshold
                            const auto pos_min_thold = std::max(0, n_swa > 0 ? pos_eval - n_swa + 2 : pos_eval);
                            if (!ckpt_async_items.empty()) { ckpt_async_poll(slot.id, true); }

                            if (n_past > 0 && n_past <= slot.prompt.n_tokens()) {
                                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                                if (pos_min == -1) {
                                    SLT_ERR(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min);
                                    GGML_ABORT("pos_min == -1, but n_past > 0 - should not happen: https://github.com/ggml-org/llama.cpp/pull/13833#discussion_r2116181237");
                                }

                                // when the prompt prefix does not match, print the tokens around the mismatch
                                // this is useful for debugging prompt caching
                                if (slots_debug) {
                                    const int np0 = std::max<int>(n_past - slots_n_diff, 0);
                                    const int np1 = std::min<int>(n_past + slots_n_diff + 2, std::min(slot.prompt.tokens.size(), slot.task->tokens.size()));

                                    std::stringstream ss0;
                                    std::stringstream ss1;

                                    std::stringstream st0;
                                    std::stringstream st1;

                                    ss0 << "old: ... ";
                                    ss1 << "new: ... ";

                                    for (int i = np0; i < np1; i++) {
                                        if (i == n_past) {
                                            ss0 << " | ";
                                            ss1 << " | ";
                                        }

                                        {
                                            const auto token = slot.prompt.tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss0 << piece;
                                            st0 << std::setw(8) << token;
                                        }

                                        {
                                            const auto token = slot.task->tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss1 << piece;
                                            st1 << std::setw(8) << token;
                                        }
                                    }

                                    SLT_WRN(slot, "%s\n", ss0.str().c_str());
                                    SLT_WRN(slot, "%s\n", ss1.str().c_str());

                                    SLT_WRN(slot, "%s\n", st0.str().c_str());
                                    SLT_WRN(slot, "%s\n", st1.str().c_str());
                                }

                                if (pos_min >= pos_min_thold) {
                                    // search for a context checkpoint
                                    const auto it = std::find_if(
                                        slot.prompt.checkpoints.rbegin(),
                                        slot.prompt.checkpoints.rend(),
                                        [&](const auto & cur) {
                                            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                            SLT_TRC(slot, "checking checkpoint with [%d, %d] against %d...\n", cur.pos_min, cur.pos_max, pos_min_thold);
                                            // workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]
                                            if (cur.pos_max > pos_next) {
                                                return false;
                                            }
                                            return cur.pos_min < pos_min_thold || cur.pos_min == 0;
                                        }
                                    );

                                    bool do_reset = it == slot.prompt.checkpoints.rend();

                                    if (!do_reset) {
                                        // restore the context checkpoint
                                        if (!it->load_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) ||
                                            !it->load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
                                            if (it->id_task != -1) {
                                                GGML_ABORT("failed to restore context checkpoint\n");
                                            }
                                            // restored from a slot file, not guaranteed to load - fall back to full prompt re-processing
                                            SLT_WRN(slot, "%s", "failed to load context checkpoint restored from a slot file\n");
                                            do_reset = true;
                                        }
                                    }

                                    if (!do_reset) {
                                        // restore the draft's speculative state
                                        common_speculative_set_state(slot.spec, slot.id, it->data_spec);

                                        pos_next = std::min(pos_next, std::max(it->pos_min + 1, it->pos_max));
                                        n_past   = std::min(slot.prompt.tokens.size_up_to_pos(pos_next), (size_t) it->n_tokens);
                                        SLT_TRC(slot, "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_past = %d, size = %.3f MiB)\n", it->pos_min, it->pos_max, it->n_tokens, n_past, (float) it->size() / 1024 / 1024);
                                    }

                                    if (do_reset) {
                                        SLT_TRC(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see %s)\n",
                                                "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                                        pos_next = 0;
                                        n_past = 0;
                                    }
                                }
                            }

                            {
                                // erase any checkpoints with pos_max > pos_next
                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur = *it;
                                    if (cur.pos_max > pos_next) {
                                        SLT_TRC(slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_swa = %d, pos_next = %d, size = %.3f MiB)\n", cur.pos_min, cur.pos_max, cur.n_tokens, n_swa, pos_next, (float) cur.size() / 1024 / 1024);
                                        ckpt_async_settle(&*it);
                                        ckpt_recycle_push(it->data_tgt);
                                        it = slot.prompt.checkpoints.erase(it);
                                    } else {
                                        ++it;
                                    }
                                }
                            }
                        }

                        // [TAG_PROMPT_LOGITS]
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
                            SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, slot.task->n_tokens());
                            n_past--;
                            SLT_WRN(slot, "n_past was set to %d\n", n_past);
                        }

                        slot.stats.n_prompt_cached    = n_past;
                        slot.stats.n_prompt_processed = 0;

                        metrics.add_prompt_cached(n_past);

                        slot.prompt.tokens.keep_first(n_past);

                        // this is to signal the client that the request has started processing
                        if (slot.task->params.stream) {
                            if (slot.task->params.return_progress) {
                                // send initial 0% progress update if needed
                                send_partial_response(slot, {}, true);
                            } else {
                                // otherwise, for streaming without progress, signal HTTP to send the headers (i.e. 200 status)
                                send_partial_response(slot, {}, false, true);
                            }
                        }
                    } // end of SLOT_STATE_STARTED

                    if (!slot.can_split()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.size() + slot.task->n_tokens() > n_batch) {
                            return;
                        }
                    }

                    // note: the prompt timing is advanced in post_decode(), so it does not cover
                    //       the tokens added to the batch below
                    slot.print_timings_pp();

                    // truncate any tokens that are beyond n_past for this slot
                    const llama_pos p0 = slot.prompt.tokens.pos_next();

                    SLT_TRC(slot, "cached n_tokens = %d, memory_seq_rm [%d, end)\n", slot.prompt.n_tokens(), p0);

                    dft_mem_seq_rm(slot, p0, -1);

                    // shared prompt prefix: once it is processed, the children continue from it with their own prompt
                    bool wait_shared = false;
                    if (slot.task->n_tokens_shared > 0) {
                        const bool is_shared_done = slot.prompt.n_tokens() == slot.task->n_tokens_shared;
                        for (auto & other : slots) {
                            if (other.state != SLOT_STATE_WAIT_OTHER || other.task->id_parent != slot.task->id) {
                                continue;
                            }
                            if (is_shared_done) {
                                SLT_TRC(slot, " - copying shared prompt (%d tokens) to child %d\n", slot.prompt.n_tokens(), other.id);
                                slot.copy_prompt_to(other);
                                other.state = SLOT_STATE_STARTED;
                            } else {
                                wait_shared = true;
                            }
                        }
                    }

                    // If using an alora, there may be uncached tokens that come
                    // before the invocation sequence. When this happens, the
                    // tokens before the invocation sequence need to be
                    // processed without the adapter in a separate batch, then
                    // the adapter needs to be enabled for the remaining tokens.
                    if (lora_all_alora(slot.lora) && slot.alora_invocation_start - 1 > slot.prompt.n_tokens()) {
                        SLT_DBG(slot, "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                        const auto & enabled_loras = lora_get_enabled_ids(slot.lora);
                        GGML_ASSERT(enabled_loras.size() == 1);
                        alora_scale = slot.lora[enabled_loras[0]].scale;
                        slot.lora[enabled_loras[0]].scale = 0.0f;
                        alora_disabled_id = enabled_loras[0];
                    }

                    bool do_checkpoint = params_base.n_ctx_checkpoints > 0;

                    // make checkpoints only for completion tasks
                    do_checkpoint = do_checkpoint && slot.task->type == SERVER_TASK_TYPE_COMPLETION;

                    // a request that does not reuse its prompt can never restore a prompt checkpoint
                    if (do_checkpoint && !slot.task->params.cache_prompt) {
                        do_checkpoint = false;
                        ++checkpoint_skipped_nocache;
                    }

                    // make a checkpoint of the parts of the memory that cannot be rolled back.
                    // checkpoints are created only if:
                    // - the model does not support partial sequence removal
                    // - the model uses SWA (and we are not using `swa_full`)
                    // - the model supports partial sequence removal but only up to a fixed bound
                    do_checkpoint = do_checkpoint && (
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS ||
                            n_swa > 0);

                    // TODO: do the same for all models, then remove process_mtmd_chunk()
                    if (use_mixed_batch() && !slot.can_split() && input_tokens.has_mtmd) {
                        if (!add_prompt_mixed(slot)) {
                            return;
                        }
                    }

                    bool has_mtmd = false;

                    // check if we should process the mtmd chunk
                    while (true) {
                        auto cur_token_idx = slot.prompt.n_tokens();
                        if (
                            cur_token_idx >= slot.task->n_tokens() ||
                            input_tokens[cur_token_idx] != LLAMA_TOKEN_NULL // encountered a text token
                        ) {
                            break;
                        }

                        // process the mtmd chunk
                        // note: it submits its own decode, potentially be async
                        //       so the timing is queued and flushed on the next sync
                        metrics_pre_decode();

                        // encode on the worker thread, so we can still handle metrics tasks
                        size_t n_tokens_out = 0;
                        int32_t res = 0;
                        queue_tasks.yield_to_queue([&]() {
                            res = process_mtmd_chunk(slot, slot.mbatch, cur_token_idx, n_tokens_out);
                        });

                        if (res != 0) {
                            SLT_ERR(slot, "failed to process mtmd chunk, res = %d\n", res);
                            send_error(slot, "failed to process mtmd chunk", ERROR_TYPE_SERVER);
                            slot.release();
                            return; // the slot is done, skip it entirely
                        }

                        metrics_queue_prompt(n_tokens_out);
                        slot.stats.n_prompt_processed += n_tokens_out;
                        slot.stats.update_prompt_last();

                        // add the mtmd chunk to cache
                        {
                            const auto & chunk = input_tokens.find_chunk(cur_token_idx);
                            // the chunk is already in the KV cache at this point, so we don't need to keep its data around
                            slot.prompt.tokens.push_back_placeholder(chunk.get());
                        }

                        has_mtmd = true;
                    }

                    const auto & spans = slot.task->params.message_spans;
                    const auto last_user_pos = spans.last_user_message_pos();
                    const int system_boundary = system_prefix_length(*slot.task);
                    publish_system_prefix(slot, system_boundary);

                    const int32_t n_decision_first = slot.task->type == SERVER_TASK_TYPE_DECISION ? slot.task->decision.pos_first() : -1;

                    const int prompt_quantum = canonical_prefill_tokens(slot);
                    if (prompt_quantum > 0 && n_batch - batch.size() < prompt_quantum) {
                        // Flush existing work, then admit an entire canonical chunk.
                        // Capacity reservation guarantees room when the next car starts.
                        return;
                    }
                    const int prompt_chunk_start = slot.prompt.n_tokens();
                    // add prompt tokens for processing in the current batch
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.size() < n_batch &&
                            (prompt_quantum == 0 || slot.prompt.n_tokens() - prompt_chunk_start < prompt_quantum)) {
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }

                        // stop at the end of the shared prefix, the children are started from this state
                        if (wait_shared && slot.prompt.n_tokens() == slot.task->n_tokens_shared) {
                            break;
                        }

                        // the outputs of a decision are read from one batch, do not split them
                        if (slot.prompt.n_tokens() == n_decision_first && batch.size() + slot.task->n_tokens() - n_decision_first > n_batch) {
                            break;
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                            break;
                        }

                        // embedding requires all tokens in the batch to be output;
                        // MTP also wants logits at every prompt position so the
                        // streaming hook can mirror t_h_nextn into slot.ctx_dft.
                        add_ok &= batch.add(slot.id,
                            cur_tok,
                            /* pos       = */ slot.prompt.tokens.pos_next(),
                            /* output    = */ slot.need_embd(),
                            /* is_prompt = */ true);
                        if (!slot.task->decision.order.empty()) {
                            batch.set_decision_order(batch.size() - 1, slot.task->decision.order[slot.prompt.n_tokens()]);
                        }
                        slot.prompt.tokens.push_back(cur_tok);
                        if (system_boundary > 0 && slot.prompt.n_tokens() == system_boundary) { break; }

                        // break at the last user message, or at user messages at least min step past the last checkpoint
                        if (do_checkpoint && spans.is_user_start(slot.prompt.n_tokens())) {
                            const auto pos = slot.prompt.n_tokens();
                            const auto & checkpoints = slot.prompt.checkpoints;

                            if (pos == last_user_pos || checkpoints.empty() || pos > checkpoints.back().n_tokens + params_base.checkpoint_min_step) {
                                break;
                            }
                        }

                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        // create checkpoints that many tokens before the end of the prompt:
                        //  - 4 + n_ubatch
                        //  - 4
                        // ref: https://github.com/ggml-org/llama.cpp/pull/20288
                        if (do_checkpoint) {
                            const int checkpoint_offsets[] = {4 + n_ubatch, 4};

                            bool should_break = false;
                            for (int offset : checkpoint_offsets) {
                                // A fused car has a smaller admission budget,
                                // which must not move recurrent checkpoints.
                                const int checkpoint_batch = pipeline_on ? int(llama_n_batch(ctx_tgt)) : n_batch;
                                const int n_last = std::min(checkpoint_batch, offset);
                                if (slot.task->n_tokens() == slot.prompt.n_tokens() + n_last) {
                                    should_break = true;
                                    break;
                                }
                            }
                            if (should_break) {
                                break;
                            }
                        }
                    }

                    // the number of tokens added to the batch for the current slot
                    const auto n_tokens_cur = batch.size() - n_tokens_prev;

                    const auto n_tokens_start = slot.prompt.n_tokens() - n_tokens_cur;

                    const bool near_prompt_end = slot.task->n_tokens() < slot.prompt.n_tokens() + n_ubatch;
                    const int physical_batch = pipeline_on ? int(llama_n_batch(ctx_tgt)) : n_batch;
                    const bool tail_checkpoint_needed = do_checkpoint && (
                            server_checkpoint_tail_needed(
                                near_prompt_end, slot.task->n_tokens(), n_tokens_start, n_ubatch, physical_batch,
                                slot.task->id, slot.prompt.checkpoints));

                    const bool is_user_start = spans.is_user_start(n_tokens_start);
                    const bool is_last_user_message = n_tokens_start == last_user_pos;

                    // entire prompt has been processed
                    if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.size() > 0);

                        // extract the logits only for the last token
                        batch.set_output(batch.size() - 1, true);

                        slot.stats.n_gen = 0;
                        slot.i_batch     = batch.size() - 1;

                        slot.init_sampler();
                    }

                    // Small admission slices preserve just the first available tail
                    // snapshot and the explicit final-four-token point. Apply this
                    // to the final car too; user-message boundary policy stays below.
                    if (!is_user_start && !tail_checkpoint_needed) {
                        do_checkpoint = false;
                    }

                    const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                    const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);

                    // nothing to checkpoint yet
                    // TODO: is this check needed?
                    if (do_checkpoint && pos_min < 0) {
                        do_checkpoint = false;
                    }

                    // do not checkpoint after mtmd chunks
                    do_checkpoint = do_checkpoint && !has_mtmd;

                    // no need to create checkpoints that are too close together, unless it's the last user message
                    do_checkpoint = do_checkpoint && (
                            slot.prompt.checkpoints.empty() ||
                            is_last_user_message || near_prompt_end ||
                            n_tokens_start > slot.prompt.checkpoints.back().n_tokens + params_base.checkpoint_min_step);
                    SLT_DBG(slot, "main/do_checkpoint = %s, pos_min = %d, pos_max = %d\n", do_checkpoint ? "yes" : "no", pos_min, pos_max);

                    // note: we create the checkpoint before calling llama_decode(), so the current batch is not
                    //       yet processed and therefore it is not part of the checkpoint.
                    if (do_checkpoint) {
                        create_checkpoint(slot, n_tokens_cur, pos_min, pos_max);
                    }
                }

                if (!slot_batched) {
                    slot_batched = &slot;
                }
            });
        }
    }

    void pre_decode() {
        std::vector<server_slot *> generating, drafting;
        prepare_decode_generations(generating, drafting);
        if (!drafting.empty()) {
            queue_tasks.yield_to_queue([&]() { draft_speculative_batch(); });
        }
        finish_drafts(drafting);
        finish_pre_decode(generating);
    }

    // see https://github.com/ggml-org/llama.cpp/pull/29969
    // TODO @ngxson : maybe remove this once we use "mixed" batch everywhere
    bool use_mixed_batch() const {
        return !llama_get_memory(ctx_tgt) || !llama_get_causal_attn(ctx_tgt);
    }

    // add the rest of the prompt to the batch, the mtmd chunks are added as embeddings next to the text tokens
    // the caller makes sure that it fits in the batch
    // returns false on error, the slot is then released
    bool add_prompt_mixed(server_slot & slot) {
        const auto & input_tokens = slot.task->tokens;
        const auto n_tokens_prev = batch.size();

        while (slot.prompt.n_tokens() < slot.task->n_tokens()) {
            const auto cur_token_idx = slot.prompt.n_tokens();
            const llama_token cur_tok = input_tokens[cur_token_idx];

            if (cur_tok != LLAMA_TOKEN_NULL) {
                const bool add_ok = batch.add(slot.id,
                    cur_tok,
                    /* pos       = */ slot.prompt.tokens.pos_next(),
                    /* output    = */ slot.need_embd(),
                    /* is_prompt = */ true);
                GGML_ASSERT(add_ok);
                if (!slot.task->decision.order.empty()) {
                    batch.set_decision_order(batch.size() - 1, slot.task->decision.order[cur_token_idx]);
                }
                slot.prompt.tokens.push_back(cur_tok);
                continue;
            }

            const auto & chunk = input_tokens.find_chunk(cur_token_idx);

            float * embd = slot.mbatch ? mtmd_batch_get_output_embd(slot.mbatch.get(), chunk.get()) : nullptr;
            if (!embd) {
                // encode on the worker thread, so we can still handle metrics tasks
                int32_t res = 0;
                queue_tasks.yield_to_queue([&]() {
                    res = encode_mtmd_chunk(slot, slot.mbatch, cur_token_idx);
                });
                embd = res == 0 ? mtmd_batch_get_output_embd(slot.mbatch.get(), chunk.get()) : nullptr;
            }

            if (!embd || !add_mtmd_chunk(slot, chunk.get(), embd, batch)) {
                SLT_ERR(slot, "%s", "failed to process mtmd chunk\n");
                // the batch must not keep the entries of a released slot
                batch.truncate(n_tokens_prev);
                send_error(slot, "failed to process mtmd chunk", ERROR_TYPE_SERVER);
                slot.release();
                return false;
            }

            slot.prompt.tokens.push_back_placeholder(chunk.get());
        }

        return true;
    }

    // returns true = success ; false = retry with smaller batch size
    // throw std::runtime_error on fatal error
    bool decode(int32_t & n_batch, int32_t off) {
        SRV_DBG("n_batch (effective) = %d, off = %d\n", n_batch, off);

        metrics_pre_decode();

        if (batch.size() == 0) {
            SRV_WRN("%s", "no tokens to decode\n");

            if (++n_empty_consecutive > 3) {
                GGML_ABORT("fatal error - please provide logs and repro in %s\n", "https://github.com/ggml-org/llama.cpp/pull/20277");
            }

            return true; // nothing to decode
        } else {
            n_empty_consecutive = 0;
        }

        // TODO @ngxson : dft model may have different n_embd than the tgt model, so we check & reject if that's the case
        // this case is not currently used by any models, but may need to be supported in the future
        if (spec && batch.has_embd()) {
            if (llama_model_n_embd_inp(model_dft) != llama_model_n_embd_inp(model_tgt)) {
                SRV_ERR("%s", "unsupported batch.has_embd + spec case\n");
                throw std::runtime_error("unsupported batch.has_embd + spec case");
            }
        }

        bool has_output = false;
        for (int i = off; i < off + batch.view.size(); ++i) {
            has_output |= batch.tokens[i].output;
        }

        // yield to the queue, so we can still handle metrics tasks while decoding
        // note: the sync is done here too, so that the wait is also covered by the yield
        int ret = 0;
        queue_tasks.yield_to_queue([&]() {
            ret = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get());
            if (ret == 0 && has_output) {
                llama_synchronize(ctx_tgt);
                if (pipeline_on) {
                    // Prompt and fallback decodes also use private output slabs.
                    // Pooled embeddings keep the regular shared-buffer path.
                    llama_decode_ticket_select(ctx_tgt, llama_decode_ticket_last(ctx_tgt));
                }
            }
        });

        if (ret != 0) {
            {
                std::string err;

                if (n_batch == 1 && ret == 1) {
                    // TODO: try to terminate only the largest active slot/sequence and continue with the rest
                    //       need to remove the tokens from the current batch too
                    err = "Context size has been exceeded.";
                }

                if (ret == -1) {
                    err = "Invalid input batch.";
                }

                if (ret < -1) {
                    // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                    err = "Compute error.";
                }

                // TODO: handle ret == 2 (abort) when we start aborting

                if (!err.empty()) {
                    SRV_ERR("%s off = %d, n_batch = %d, ret = %d\n", err.c_str(), off, n_batch, ret);

                    for (auto & slot : slots) {
                        if (slot.is_processing()) {
                            send_error(slot, err);
                            slot.release();

                            // note: it's complicated to keep track of how much of the current batch has been
                            //       processed before the error occurred, so we simply clear the entire context
                            slot.prompt_clear();
                        }
                    }

                    // stop, do not retry with smaller batch size
                    throw std::runtime_error(err);
                }
            }

            // retry with half the batch size to try to find a free slot in the KV cache
            if (!try_clear_idle_slots()) {
                n_batch /= 2;
            }

            SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, off = %d, n_batch = %d, ret = %d\n", off, n_batch, ret);

            return false; // retry with the updated n_batch
        } else {
            // success, apply batch metrics
            metrics_post_decode(off, batch.view.size(), has_output);
        }

        // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
        //       for now, always re-evaluate for simplicity
        //       ref: https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4400925384
        if (spec) {
            bool ok = true;
            queue_tasks.yield_to_queue([&]() {
                ok = process_speculative_batch(batch.view);
            });

            if (!ok) {
                SRV_ERR("%s", "failed to process speculative batch\n");

                // TODO: handle error
                throw std::runtime_error("failed to process speculative batch");
            }
        }

        // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
        for (auto & slot : slots) {
            if (slot.state == SLOT_STATE_DONE_PROMPT && slot.task->is_parent()) {
                std::vector<server_slot *> children;
                for (auto & other : slots) {
                    if (other.state == SLOT_STATE_WAIT_OTHER && slot.task->id == other.task->id_parent) {
                        children.push_back(&other);
                    }
                }

                // all children slots should already launched by launch_slots_with_parent_task()
                // copy state to the child slots
                for (auto & child : children) {
                    SLT_TRC(slot, " - copying state to child %d\n", child->id);

                    GGML_ASSERT(child->state == SLOT_STATE_WAIT_OTHER);
                    // children with their own prompt are started at the end of the shared prefix
                    GGML_ASSERT(slot.task->n_tokens_shared == 0);

                    slot.copy_state_to(*child);
                    child->state = SLOT_STATE_DONE_PROMPT;
                }
            }
        }

        return true;
    }

    void post_decode(int32_t n_batch_tokens, int32_t off) {
        // for checking if a given batch index is inside the current sub-batch
        auto is_inside_view = [&](int32_t idx) {
            return idx >= off && idx < off + n_batch_tokens;
        };

        // TODO @ngxson : it's tricky to make sub-batch compatible with common_sampler_sample_and_accept_n,
        // so for now we will throw an error in this case: https://github.com/ggml-org/llama.cpp/issues/24840
        iterate(slots, [&](server_slot & slot) {
            for (auto & i : slot.spec_i_batch) {
                if (!is_inside_view(i)) {
                    throw std::runtime_error(string_format("speculative batch index %d is not inside the current sub-batch [%d, %d)", i, off, off + n_batch_tokens));
                }
            }
        });

        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        iterate(slots, [&](server_slot & slot) {
            if (pipeline_settling) {
                const auto & lane = pipeline_lanes[pipeline_filter];
                // A later chunk may already have changed the shared slot to DONE_PROMPT.
                // Only the car containing its final prompt token may sample that slot.
                if (lane.prefill_slot == slot.id) {
                    if (lane.prefill_output < 0) {
                        // A retired prompt-only car has no logits to sample,
                        // but streaming clients still need its progress event.
                        if (slot.task->params.stream && slot.task->params.return_progress) {
                            send_partial_response(slot, {}, true);
                        }
                        return;
                    }
                    slot.i_batch = lane.prefill_output;
                }
            }
            // optionally send prompt processing progress
            if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->params.stream && slot.task->params.return_progress) {
                    send_partial_response(slot, {}, true);
                }
            }

            if (!is_inside_view(slot.i_batch)) {
                // the required token not in this sub-batch, skip
                return;
            }

            if (slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->type == SERVER_TASK_TYPE_EMBEDDING) {
                    // prompt evaluated for embedding
                    send_embedding(slot, batch.view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                if (slot.task->type == SERVER_TASK_TYPE_RERANK) {
                    send_rerank(slot, batch.view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                if (slot.task->type == SERVER_TASK_TYPE_DECISION) {
                    send_decision(slot, batch.view, slot.i_batch - off);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                GGML_ASSERT(slot.task->need_sampling());

                // prompt evaluated for next-token prediction
                slot.state = SLOT_STATE_GENERATING;

                if (slot.can_speculate()) {
                    dft_run([sp = slot.spec, id = slot.id, tokens = slot.prompt.tokens.get_text_tokens()]() {
                        common_speculative_begin(sp, id, tokens);
                    });
                }
            } else if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            if (slot.can_speculate() && !slot.spec_draft.empty()) {
                return; // sample using speculative decoding
            }

            // shifted according to the current sub-batch
            const int tok_idx = slot.i_batch - off;

            llama_token id;
            {
                scoped_timer timer(t_sampl, n_sampl);
                id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, tok_idx);
            }

            slot.i_batch = -1;

            common_sampler_accept(slot.smpl.get(), id, true);

            // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
            const int64_t t_now = ggml_time_us();

            slot.stats.n_gen += 1;

            if (slot.stats.n_gen == 1) {
                slot.stats.update_prompt_last();
                slot.t_print_last = t_now;
                slot.n_gen_last = 0;
            }

            slot.stats.update_gen_last();

            completion_token_output result;
            result.tok          = id;
            result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
            result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

            if (slot.task->params.sampling.n_probs > 0) {
                populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, tok_idx);
            }

            if (!process_token(result, slot)) {
                // release slot because of stop condition
                dft_join(__LINE__);
                slot.print_timings();
                send_final_response(slot);
                slot.release();

                return;
            }

            slot.print_timings_tg();
        });

        // speculative decoding - main model sample and accept
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate() ||
                    slot.spec_draft.empty() || slot.spec_i_batch.empty()) {
                return;
            }

            // save the original draft size
            const size_t n_draft = slot.spec_draft.size();

            GGML_ASSERT(n_draft > 0);

            std::vector<completion_token_output> verified_probs;
            // verify and try to accept the draft
            {
                common_sampler_ptr smpl_save;
                if (server_speculative_sampler_backup_needed(
                        ctx_tgt_seq_rm_type, llama_n_rs_seq(ctx_tgt), n_draft)) {
                    server_wall_timer timer(speculative_sampler_backup_us, speculative_sampler_backups);
                    smpl_save.reset(common_sampler_clone(slot.smpl.get()));
                } else {
                    ++speculative_sampler_backup_elided;
                }

                GGML_ASSERT(slot.spec_i_batch.size() == n_draft + 1);
                GGML_ASSERT(slot.spec_draft_q.empty() || (slot.spec_draft_q.size() == slot.spec_draft.size()));
                const auto & synth_probs = common_speculative_get_synth_probs(spec.get());

                // drafters that fill no distribution fall back here
                const bool use_rejection = slot.use_spec_rejection() && !slot.spec_draft_q.empty();

                std::vector<llama_token> accepted;
                if (!synth_probs.empty()) {
                    // synthetic acceptance replaces verification entirely, so it comes first
                    accepted = server_sample_and_accept_synth(
                            slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft,
                            synth_probs, slot.spec_synth_rng, slot.spec_is_replay);
                } else if (slot.spec_is_replay && slot.use_spec_rejection()) {
                    accepted = server_accept_replay(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft);
                } else if (use_rejection) {
                    accepted = common_sampler_sample_and_accept_n_rejection(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft, slot.spec_draft_q);
                } else {
                    std::function<void(llama_token, int)> on_sample;
                    if (slot.task->params.sampling.n_probs > 0) {
                        on_sample = [&](llama_token token, int idx) {
                            completion_token_output result;
                            result.tok = token;
                            result.prob = 1.0f;
                            populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, idx);
                            verified_probs.push_back(std::move(result));
                        };
                    }
                    accepted = common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft, false, on_sample);
                }
                slot.spec_i_batch.clear();

                GGML_ASSERT(accepted.size() >= 1);

                const uint32_t n_rollback = slot.spec_draft.size() + 1 - accepted.size();

                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                    (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_rollback > llama_n_rs_seq(ctx_tgt));

                // check for partial draft acceptance
                if (n_rollback > 0) {
                    if (use_ckpt_tgt) {
                        if (trace > 0) {
                            SLT_INF(slot, "accepted %2zu/%2zu draft tokens (restore checkpoint)\n", accepted.size() - 1, slot.spec_draft.size());
                        }

                        // partial acceptance is not supported by the context -> truncate the draft and restore the state
                        slot.spec_is_replay = true;
                        slot.spec_draft = std::move(accepted);

                        const auto & ckpt = slot.spec_ckpt;

                        SLT_DBG(slot, "restoring speculative checkpoint (pos_min = %d, pos_max = %d, size = %zu)\n", ckpt.pos_min, ckpt.pos_max, ckpt.size());

                        dft_join(__LINE__);
                        GGML_ASSERT(ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));

                        if (slot.ctx_dft) {
                            GGML_ASSERT(ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                        }

                        slot.mem.seq_rm(slot.id, ckpt.pos_max + 1, -1);

                        slot.prompt.tokens.keep_first(ckpt.n_tokens);
                        GGML_ASSERT(smpl_save);
                        common_sampler_copy(smpl_save.get(), slot.smpl.get());

                        return;
                    }
                }

                if (trace > 0) {
                    SLT_INF(slot, "accepted %2zu/%2zu draft tokens\n", accepted.size() - 1, n_draft);
                }

                dft_run([sp = slot.spec, id = slot.id, n = (uint16_t) (accepted.size() - 1)]() { common_speculative_accept(sp, id, n); });

                slot.spec_draft = std::move(accepted);
            }

            const auto ids = std::move(slot.spec_draft);

            size_t n_accepted = ids.size() - 1;
            if (slot.spec_is_replay && n_accepted > 0) {
                n_accepted--;
            }
            slot.spec_is_replay = false;

            slot.stats.update_gen_last();

            // update how many tokens out of those tested were accepted
            slot.stats.n_draft_accepted += n_accepted;
            slot.stats.n_draft_verif_steps += 1;

            auto & n_accepted_per_pos = slot.n_accepted_per_pos;
            if (n_accepted_per_pos.empty()) {
                n_accepted_per_pos.resize(common_speculative_n_max(spec.get()), 0);
            }
            for (size_t i = 0; i < n_accepted && i < n_accepted_per_pos.size(); ++i) {
                n_accepted_per_pos[i]++;
            }

            // add accepted tokens to the prompt
            slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);
            slot.prompt.tokens.insert({ids.begin(), ids.end() - 1});

            slot.sampled = ids.back(); // last accepted token
            SLT_DBG(slot, "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", slot.sampled, ids.size(), n_draft);

            dft_mem_seq_rm(slot, slot.prompt.tokens.pos_next(), -1);

            for (size_t i = 0; i < ids.size(); ++i) {
                completion_token_output result;

                if (i < verified_probs.size()) { result = std::move(verified_probs[i]); }
                result.tok          = ids[i];
                result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                if (verified_probs.empty()) { result.prob = 1.0f; }

                slot.stats.n_gen += 1;

                if (!process_token(result, slot)) {
                    dft_join(__LINE__);
                    slot.print_timings();
                    send_final_response(slot);
                    slot.release();

                    return;
                }
            }

            slot.print_timings_tg();

            SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int) n_accepted, (int) n_draft, slot.prompt.n_tokens());
        });
    }

    // context size of a single slot, capped by --kv-unified-per-slot and by the training context of the model
    int n_ctx_slot() const {
        int res = llama_n_ctx_seq(ctx_tgt);

        if (params_base.kv_unified_per_slot > 0) {
            res = std::min(res, params_base.kv_unified_per_slot);
        }

        return std::min(res, llama_model_n_ctx_train(model_tgt));
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
        refresh_speculative_prompt_states();
    }

    //
    // metrics helpers
    //

    // call before submitting a decode, so that the queued prompt stats can be timed
    void metrics_pre_decode() {
        t_decode_start = ggml_time_us();
    }

    // the batch is submitted, but its compute may not be done yet
    void metrics_queue_prompt(uint64_t n_tokens) {
        if (n_tokens == 0) {
            return;
        }
        if (n_prompt_queued == 0) {
            t_prompt_start = t_decode_start;
        }
        n_prompt_queued += n_tokens;
    }

    // call only after the context is synchronized, otherwise the time is meaningless
    void metrics_flush_prompt() {
        if (n_prompt_queued == 0) {
            return;
        }
        metrics.add_prompt(n_prompt_queued, ggml_time_us() - t_prompt_start);
        n_prompt_queued = 0;
    }

    // has_output is computed by the caller, which also already synchronized the context if it is set
    void metrics_post_decode(int32_t off, int32_t n_tokens, bool has_output) {
        metrics.n_decode++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                metrics.n_busy_slots++;
            }
            metrics.n_tokens_max = std::max(metrics.n_tokens_max, (uint64_t) slot.prompt.n_tokens());
        }

        // apply enqueued prompt tokens stats
        // note: a slot can be released before we get here, which clears its stats
        //       the tokens were still computed, counted in the global metrics, not in slot
        uint64_t n_prompt_tokens = 0;

        for (int i = off; i < off + n_tokens; ++i) {
            const auto & t = batch.tokens[i];

            if (!t.is_prompt) {
                continue; // generated tokens are handled after sampling
            }

            n_prompt_tokens++;

            auto & slot = slots[t.id_slot];
            if (slot.stats.is_set()) {
                slot.stats.n_prompt_processed++;
            }
        }

        metrics.n_prompt_observed += n_prompt_tokens;
        metrics_queue_prompt(n_prompt_tokens);

        if (has_output) {
            // the context is already synchronized, so the timings are correct
            metrics_flush_prompt();
        }

        // advance the prompt timing of the slots that had tokens in this batch
        // note: a second pass, it must run after the sync to reflect the compute
        const int64_t t_now = ggml_time_us();
        for (int i = off; i < off + n_tokens; ++i) {
            const auto & t = batch.tokens[i];
            auto & slot = slots[t.id_slot];
            if (t.is_prompt && slot.stats.is_set()) {
                slot.stats.set_prompt_last(t_now);
            }
        }
    }

    // flush any queued prompt metrics if all slots are now idle
    void metrics_flush_idle() {
        if (n_prompt_queued == 0) {
            return;
        }

        llama_synchronize(ctx_tgt);
        metrics_flush_prompt();
    }

    void metrics_on_prediction(const server_slot & slot) {
        const uint64_t t_us    = slot.stats.t_gen_us();
        const uint64_t n       = slot.stats.n_gen;
        const uint64_t n_steps = slot.stats.n_gen_steps();

        metrics.predict       .add(n, n_steps, t_us);
        metrics.predict_bucket.add(n, n_steps, t_us);

        metrics.n_draft_tokens      += slot.stats.n_draft_tokens;
        metrics.n_draft_accepted    += slot.stats.n_draft_accepted;
        metrics.n_draft_verif_steps += slot.stats.n_draft_verif_steps;

        auto & dst = metrics.n_accepted_per_pos;
        const auto & src = slot.n_accepted_per_pos;

        if (dst.size() < src.size()) {
            dst.resize(src.size(), 0);
        }
        for (size_t i = 0; i < src.size(); i++) {
            dst[i] += src[i];
        }
    }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
    // All inference producers have stopped. Wake every current or late HTTP
    // reader with an explicit error before the HTTP pool is joined.
    impl->queue_results.terminate();
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx_tgt;
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, eos_id, true) : "";

    const char * ftype_name = llama_ftype_name(llama_model_ftype(impl->model_tgt));

    return server_context_meta {
        /* build_info              */ std::string(llama_build_info()),
        /* model_name              */ impl->model_name,
        /* model_aliases           */ impl->model_aliases,
        /* model_tags              */ impl->model_tags,
        /* model_path              */ impl->params_base.model.path,
        /* model_output_modalities */ server_model_output_modalities(common_get_decision_type(impl->model_tgt)),
        /* has_mtmd                */ impl->mctx != nullptr,
        /* has_inp_image           */ impl->chat_params.allow_image,
        /* has_inp_audio           */ impl->chat_params.allow_audio,
        /* has_inp_video           */ impl->chat_params.allow_video,
        /* json_ui_settings        */ impl->json_ui_settings,
        /* slot_n_ctx              */ impl->n_ctx_slot(),
        /* pooling_type            */ llama_pooling_type(impl->ctx_tgt),

        /* chat_params             */ impl->chat_params,
        /* chat_template_caps      */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str           */ bos_token_str,
        /* eos_token_str           */ eos_token_str,
        /* fim_pre_token           */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token           */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token           */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token           */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token           */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token           */ llama_vocab_fim_sep(impl->vocab),

        /* logit_bias_eog          */ impl->params_base.sampling.logit_bias_eog,

        /* model_vocab_type        */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens    */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train       */ llama_model_n_ctx_train(impl->model_tgt),
        /* model_n_embd_inp        */ llama_model_n_embd(impl->model_tgt),
        /* model_n_params          */ llama_model_n_params(impl->model_tgt),
        /* model_size              */ llama_model_size(impl->model_tgt),
        /* model_ftype             */ ftype_name,
    };
}

// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_res_spipe {
    server_response_reader rd;
    server_res_generator(server_queue & queue_tasks, server_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

void server_context::set_state_callback(server_state_callback_t callback) {
    impl->callback_state = std::move(callback);
}

//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type,
            const common_chat_session & chat_session,
            bool has_system_message) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;
    auto & params = this->params;

    res->set_req(&req); // will also set spipe if needed

    int32_t sse_ping_interval = params.sse_ping_interval;

    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        if (!params.path_prompts_log_dir.empty()) {
            const auto file_path = std::filesystem::path(params.path_prompts_log_dir) / string_format("%012" PRId64 ".txt", ggml_time_ms());
            std::ofstream f(file_path);
            if (f) {
                f << (prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());
            } else {
                SRV_ERR("failed to create %s\n", file_path.string().c_str());
            }
        }

        // process prompt
        std::vector<server_tokens> inputs;

        if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files, ctx_server.init_opt));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true, ctx_server.init_opt);
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_schema::eval_llama_cmpl_schema(
                    ctx_server.vocab,
                    params,
                    meta->logit_bias_eog,
                    data);

            task.apply_chat_session(chat_session);
            if (has_system_message) {
                task.params.message_spans.spans.insert(task.params.message_spans.spans.begin(),
                        { COMMON_CHAT_ROLE_SYSTEM, 0, 0 });
            }

            task.id_slot = json_value(data, "id_slot", -1);
            sse_ping_interval = task.params.sse_ping_interval;

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks), chat_session);
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                // if single request, return single object instead of array
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                // if multiple results in OAI format, we need to re-format them
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }
                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (first_result_json == nullptr) {
            res->data = ""; // simply send HTTP headers and status code
        } else if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->set_next([res_this = res.get(), res_type, sse_ping_interval](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            auto effective_should_stop = [&res_this]() {
                return res_this->should_stop();
            };

            try {
                if (effective_should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                bool timeout = false;
                int64_t start_time = ggml_time_ms();
                auto result = rd.next([&timeout, &start_time, sse_ping_interval, &effective_should_stop]() {
                    if (effective_should_stop()) {
                        return true; // should_stop condition met
                    } else if (sse_ping_interval > 0 && ggml_time_ms() - start_time > (int64_t)sse_ping_interval * 1000) {
                        timeout = true;
                        return true; // timeout
                    }
                    return false;
                });

                if (timeout) {
                    // some clients may time out (e.g. undici) will time out if no data is received for a while, so we need to send a ping to keep the connection alive
                    SRV_DBG("%s", "sending SSE ping\n");
                    output = ":\n\n";
                    return true;
                }

                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(effective_should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        });
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    init_routes();

    // note: this must be registered before load_model()
    //       so that on sleep phase, the callback is called before ctx is destroyed
    queue_tasks.on_sleeping_state([this](bool is_sleeping) {
        update_cached_responses(is_sleeping);
    });
}

static json get_res_model_info(const server_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return {
        {"id",       meta.model_name},
        {"aliases",  meta.model_aliases},
        {"tags",     meta.model_tags},
        {"object",   "model"},
        {"architecture", server_model_architecture_json(
            meta.has_inp_image,
            meta.has_inp_audio,
            meta.has_inp_video,
            meta.model_output_modalities)},
        {"created",  std::time(0)},
        {"owned_by", "llamacpp"},
        {"meta",     {
            {"vocab_type",  meta.model_vocab_type},
            {"n_vocab",     meta.model_vocab_n_tokens},
            {"n_ctx",       meta.slot_n_ctx},
            {"n_ctx_train", meta.model_n_ctx_train},
            {"n_embd",      meta.model_n_embd_inp},
            {"n_params",    meta.model_n_params},
            {"size",        meta.model_size},
            {"ftype",       meta.model_ftype},
        }},
    };
}

static json get_res_models(const server_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return json{
        {"models", json::array({
            {
                {"name",  meta.model_name},
                {"model", meta.model_name},
                {"modified_at", ""},
                {"size", ""},
                {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                {"type", "model"},
                {"description", ""},
                {"tags", json::array({""})},
                {"capabilities", meta.has_mtmd ? json::array({"completion","multimodal"}) : json::array({"completion"})},
                {"parameters", ""},
                {"details", {
                    {"parent_model", ""},
                    {"format", "gguf"},
                    {"family", ""},
                    {"families", json::array({""})},
                    {"parameter_size", ""},
                    {"quantization_level", ""}
                }}
            }
        })},
        {"object", "list"},
        {"data", json::array({
            get_res_model_info(meta),
        })}
    };
}

static json get_res_props(const server_context_meta & meta, const common_params & params, bool is_sleeping) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    task_params tparams;
    tparams.sampling = params.sampling;
    json default_generation_settings_for_props = json {
        { "params", tparams.to_json(true) },
        { "n_ctx",  meta.slot_n_ctx },
    };

    std::string tmpl_default = common_chat_templates_source(meta.chat_params.tmpls.get(), "");
    std::string tmpl_tools   = common_chat_templates_source(meta.chat_params.tmpls.get(), "tool_use");

    json props = {
        { "default_generation_settings", default_generation_settings_for_props },
        { "total_slots",                 params.n_parallel },
        { "model_alias",                 meta.model_name },
        { "model_ftype",                 meta.model_ftype },
        { "model_path",                  meta.model_path },
        { "modalities",                  json {
            {"vision", meta.has_inp_image},
            {"video",  meta.has_inp_video},
            {"audio",  meta.has_inp_audio},
        } },
        { "media_marker",                get_media_marker() },
        { "endpoint_slots",              params.endpoint_slots },
        { "endpoint_props",              params.endpoint_props },
        { "endpoint_metrics",            params.endpoint_metrics },
        { "ui",                          params.ui },
        { "ui_settings",                 meta.json_ui_settings },
        { "chat_template",               tmpl_default },
        { "chat_template_caps",          meta.chat_template_caps },
        { "bos_token",                   meta.bos_token_str },
        { "eos_token",                   meta.eos_token_str },
        { "build_info",                  meta.build_info },
        { "is_sleeping",                 is_sleeping },
        { "cors_proxy_enabled",          params.ui_mcp_proxy },
    };
    if (params.use_jinja) {
        if (!tmpl_tools.empty()) {
            props["chat_template_tool_use"] = tmpl_tools;
        }
    }

    return props;
}

json server_routes::get_model_info() const {
    return get_res_model_info(*meta);
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok({{"status", "ok"}});
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response(true);
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // render response using cached_metrics
        auto use_cached_metrics = [&]() {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->headers["Process-Start-Time-Unix"] = std::to_string(cached_metrics.t_start);
            server_task_result_metrics tmp;
            tmp.metrics = cached_metrics;
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = tmp.to_metrics();
            // the gauges are averaged over the window between two scrapes
            cached_metrics.reset_bucket();
            should_reset_buckets = true;
        };

        if (queue_tasks.is_sleeping()) {
            use_cached_metrics();

        } else {
            // request slots data using task queue
            {
                server_task task(SERVER_TASK_TYPE_METRICS);
                task.id = res->rd.get_new_id();
                // the gauges are averaged over the window between two scrapes
                task.metrics_reset_bucket = true;
                res->rd.post_task(std::move(task), true); // high-priority task
            }

            // a task posted right before sleeping is never processed, do not wait for it
            auto result = res->rd.next([&]{
                return req.should_stop() || queue_tasks.is_sleeping();
            });
            if (!result) {
                if (!req.should_stop()) {
                    use_cached_metrics();
                }
                return res;
            }

            if (result->is_error()) {
                res->error(result->to_json());
                return res;
            }

            auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
            GGML_ASSERT(res_task != nullptr);

            res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->metrics.t_start);
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = res_task->to_metrics();
        }

        return res;
    };

    // --dash. Neither handler posts a task: the page is a constant, the snapshot is a struct copy under a mutex
    // the main loop only ever try_locks. No create_response(): that would wait for a sleeping server to wake.
    this->get_dash = [](const server_http_req &) {
        auto res = std::make_unique<server_http_res>();
        res->content_type = "text/html; charset=utf-8";
        res->headers["Cache-Control"] = "no-store";
        res->data = paraflow_dash_html();
        return res;
    };
    this->get_dash_snap = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        res->headers["Cache-Control"] = "no-store";
        paraflow_dash_snap d;
        std::string config;
        const bool have = ctx_server.dash_read(d, config);
        const int64_t now = ggml_time_us();
        std::string o;
        o.reserve(8192);
        char b[512];
        snprintf(b, sizeof(b), "{\"ok\":%s,\"seq\":%" PRIu64 ",\"now_us\":%" PRId64 ",\"t_us\":%" PRId64 ",\"t_start_us\":%" PRId64 ",\"mem_t_us\":%" PRId64,
                 have ? "true" : "false", d.seq, now, d.t_us, d.t_start_us, d.mem_t_us);
        o += b;
        o += ",\"config\":"; o += config.empty() ? "null" : config;
        const auto u64 = [&](const char * k, uint64_t v) { snprintf(b, sizeof(b), ",\"%s\":%" PRIu64, k, v); o += b; };
        const auto i64 = [&](const char * k, int64_t v)  { snprintf(b, sizeof(b), ",\"%s\":%" PRId64, k, v); o += b; };
        u64("prompt", d.prompt); u64("predict", d.predict); u64("cached", d.cached); u64("decodes", d.decodes);
        u64("draft_tokens", d.draft_tokens); u64("draft_accepted", d.draft_accepted); u64("draft_steps", d.draft_steps);
        o += ",\"accepted_pos\":[";
        for (int i = 0; i < d.n_accepted_pos; ++i) { snprintf(b, sizeof(b), "%s%" PRIu64, i ? "," : "", d.accepted_pos[i]); o += b; }
        o += "]";
        u64("settled", d.settled); u64("cars_prefill", d.cars_prefill); u64("cars_decode", d.cars_decode); u64("cars_mixed", d.cars_mixed);
        u64("payload_prompt", d.payload_prompt); u64("payload_decode", d.payload_decode);
        i64("issue_us", d.issue_us); i64("wait_us", d.wait_us); i64("process_us", d.process_us); i64("drain_us", d.drain_us);
        u64("admission_us", d.admission_us); u64("admission_skipped", d.admission_skipped); u64("vip_pause_skipped", d.vip_pause_skipped);
        u64("compact_moves", d.compact_moves); u64("ckpt_holds", d.ckpt_holds); u64("deferrals", d.deferrals);
        u64("oom_isolations", d.oom_isolations); u64("oom_terminated", d.oom_terminated);
        u64("draft_degrade", d.draft_degrade); u64("draft_failures", d.draft_failures);
        u64("dft_joins", d.dft_joins); u64("dft_join_waits", d.dft_join_waits); i64("dft_join_wait_us", d.dft_join_wait_us);
        u64("sc_hits", d.sc_hits); u64("sc_hit_tokens", d.sc_hit_tokens); u64("sc_misses", d.sc_misses);
        u64("sc_evictions", d.sc_evictions); u64("sc_failures", d.sc_failures); u64("sc_bytes", d.sc_bytes);
        i64("n_slots", d.n_slots); i64("n_processing", d.n_processing); i64("n_prompt", d.n_prompt); i64("n_gen", d.n_gen);
        i64("n_paused", d.n_paused); i64("n_suspended", d.n_suspended); i64("n_vip", d.n_vip); i64("n_deferred", d.n_deferred);
        u64("kv_used", d.kv_used); u64("kv_cap", d.kv_cap); u64("kv_max", d.kv_max);
        // slots: [id, state, lane, stream, tokens, cells, vip, mtp, paused]
        o += ",\"slots\":[";
        for (int i = 0; i < d.n_rows; ++i) {
            const auto & r = d.rows[i];
            snprintf(b, sizeof(b), "%s[%d,%d,%d,%d,%d,%d,%d,%d,%d]", i ? "," : "", r.id, r.state, r.lane, r.stream, r.n_tokens, r.n_cells, r.vip, r.mtp, r.paused);
            o += b;
        }
        o += "],\"susp_bytes\":" + std::to_string(d.susp_bytes);
        // susp: [slot, tokens, vip, capacity, bytes]
        o += ",\"susp\":[";
        for (int i = 0; i < d.n_susp; ++i) {
            const auto & r = d.susp[i];
            snprintf(b, sizeof(b), "%s[%d,%d,%d,%d,%" PRIu64 "]", i ? "," : "", r.id, r.n_tokens, r.vip, r.capacity, r.bytes);
            o += b;
        }
        o += "]";
        // context-length histogram: cumulative completions per bucket
        o += ",\"ctx_hist\":[";
        for (int i = 0; i < 64; ++i) { o += (i ? "," : "") + std::to_string(d.ctx_hist[i]); }
        o += "],\"devices\":[";
        for (int i = 0; i < d.n_dev; ++i) {
            const auto & e = d.dev[i];
            snprintf(b, sizeof(b), "%s{\"name\":\"%s\",\"model\":%" PRIu64 ",\"context\":%" PRIu64 ",\"compute\":%" PRIu64 ",\"total\":%" PRIu64 "}",
                     i ? "," : "", e.name, e.model, e.context, e.compute, e.total);
            o += b;
        }
        o += "]";
        // ?h=<first bucket wanted>: throughput history; the page asks again for its last few buckets each poll
        const std::string h = req.get_param("h");
        if (!h.empty()) { ctx_server.dash_hist_json(std::strtoll(h.c_str(), nullptr, 10), o); }
        o += "}";
        res->data = std::move(o);
        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_SLOT_GET);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        auto * res_task = dynamic_cast<server_task_result_slots*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->to_json());
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_props);
        } else {
            res->ok(get_res_props(*meta, params, false));
        }
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true, ctx_server.init_opt);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT,
            session,
            server_has_system_message(body));
    };

    this->post_chat_completions_tok = [this](const server_http_req & req) {
        return handle_count_tokens(req, TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_control = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        const std::string cmpl_id = json_value(body, "id", std::string());
        const std::string action  = json_value(body, "action", std::string());
        if (cmpl_id.empty()) {
            res->error(format_error_response("missing completion id", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (action != "reasoning_end") {
            res->error(format_error_response("unknown control action", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_CONTROL);
            task.id              = rd.get_new_id();
            task.params.control_cmpl_id = cmpl_id;
            task.params.control_action  = action;
            rd.post_task(std::move(task));
        }

        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_responses_to_chatcmpl(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP,
            session,
            server_has_system_message(body));
    };

    this->post_responses_tok_oai = [this](const server_http_req & req) {
        return handle_count_tokens(req, TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_transcriptions_oai = [this](const server_http_req & req) {
        auto res = create_response();

        if (!meta->has_mtmd || !meta->chat_params.allow_audio) {
            res->error(format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::vector<raw_buffer> files;
        json body = convert_transcriptions_to_chatcmpl(
            json::parse(req.body),
            meta->chat_params.tmpls.get(),
            req.files,
            files);
        SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_ASR,
            session,
            server_has_system_message(body));
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        server_apply_request_policy(body, req.headers);
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC,
            session,
            server_has_system_message(body));
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        return handle_count_tokens(req, TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        common_chat_session session;   // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_models);
        } else {
            res->ok(get_res_models(*meta));
        }
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens").get<llama_tokens>();
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_rerank = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);

        // if true, use TEI API format, otherwise use Jina API format
        // Jina: https://jina.ai/reranker/
        // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
        bool is_tei_format = body.contains("texts");

        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                res->error(format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        } else {
            res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::vector<std::string> documents = json_value(body, "documents",
                                             json_value(body, "texts", std::vector<std::string>()));
        if (documents.empty()) {
            res->error(format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int top_n = json_value(body, "top_n", (int)documents.size());

        // create and queue the task
        json responses = json::array();
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            tasks.reserve(documents.size());
            for (size_t i = 0; i < documents.size(); i++) {
                auto tmp = format_prompt_rerank(ctx_server.model_tgt, ctx_server.vocab, ctx_server.mctx, query, documents[i], ctx_server.init_opt);
                server_task task = server_task(SERVER_TASK_TYPE_RERANK);
                task.id     = rd.get_new_id();
                task.tokens = std::move(tmp);
                tasks.push_back(std::move(task));
            }
            rd.post_tasks(std::move(tasks));
        }

        // wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);

        // collect results
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        }

        // write JSON response
        json root = format_response_rerank(
            body,
            meta->model_name,
            responses,
            is_tei_format,
            documents,
            top_n);

        res->ok(root);
        return res;
    };

    this->post_systemone = [this](const server_http_req & req) {
        auto res = create_response();
        const auto & decision = ctx_server.decision;
        if (decision.type == COMMON_DECISION_TYPE_NONE) {
            res->error(format_error_response("This model is not a decision model", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);
        const auto questions = decision.parse_questions(body);

        std::vector<raw_buffer> files;
        const json state = decision.parse_state(body, files);
        if (!files.empty() && (!decision.can_use_images() || !meta->has_inp_image)) {
            res->error(format_error_response("This server does not support image input for decisions. For a model that supports it, start it with `--mmproj`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // one task per variant of each question, or one task for all the questions
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            if (decision.is_joint()) {
                server_task task = server_task(SERVER_TASK_TYPE_DECISION);
                task.id = rd.get_new_id();
                decision.fill_task_joint(state, questions, files, ctx_server.mctx, ctx_server.init_opt, task);
                tasks.push_back(std::move(task));
            } else {
                for (const auto & question : questions) {
                    for (size_t variant = 0; variant < decision.n_variants(question); variant++) {
                        server_task task = server_task(SERVER_TASK_TYPE_DECISION);
                        task.id = rd.get_new_id();
                        decision.fill_task(state, questions, question, variant, files, ctx_server.mctx, ctx_server.init_opt, task);
                        tasks.push_back(std::move(task));
                    }
                }
            }
            if (decision.can_share_prompt()) {
                tasks = server_decision_group_tasks(std::move(tasks), params.n_parallel);
            }
            rd.post_tasks(std::move(tasks));
        }

        auto all_results = rd.wait_for_all(req.should_stop);

        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        }

        json answers = json::object();
        int32_t n_tokens = 0;
        size_t i_result = 0;
        size_t i_score  = 0;
        for (const auto & question : questions) {
            std::vector<std::vector<float>> scores;
            if (decision.is_joint()) {
                // one result with the scores of all the questions, in order
                auto * result = dynamic_cast<server_task_result_decision *>(all_results.results[0].get());
                GGML_ASSERT(result != nullptr && i_score + question.options.size() <= result->scores.size());
                scores.emplace_back(result->scores.begin() + i_score, result->scores.begin() + i_score + question.options.size());
                i_score += question.options.size();
                n_tokens = result->n_tokens;
            }
            for (size_t variant = 0; !decision.is_joint() && variant < decision.n_variants(question); variant++) {
                auto * result = dynamic_cast<server_task_result_decision *>(all_results.results[i_result++].get());
                GGML_ASSERT(result != nullptr);
                scores.push_back(result->scores);
                n_tokens += result->n_tokens;
            }
            answers[question.id] = decision.format_answer(question, scores, !files.empty());
        }

        res->ok(json{
            {"model",   meta->model_name},
            {"answers", answers},
            {"usage",   {
                {"input_tokens",  n_tokens},
                {"output_tokens", 0},
            }},
        });
        return res;
    };

    this->get_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_GET_LORA);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_get_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    // same shapes as tokenize_input_prompts(), plus OAI content: { "content": [ { "type": "text"|"image_url"|"input_audio"|"input_video", ... } ] }
    auto tokenize_entry = [&](const json & p) {
        if (p.is_object() && p.contains("content")) {
            return tokenize_oai_content_array(ctx_server.vocab, ctx_server.mctx, meta->chat_params, p.at("content"), true, true, ctx_server.init_opt);
        }
        return tokenize_input_subprompt(ctx_server.vocab, ctx_server.mctx, p, true, true, ctx_server.init_opt);
    };

    std::vector<server_tokens> tokenized_prompts;
    if (prompt.is_array() && !json_is_array_and_contains_numbers(prompt)) {
        for (const auto & p : prompt) {
            tokenized_prompts.push_back(tokenize_entry(p));
        }
    } else {
        tokenized_prompts.push_back(tokenize_entry(prompt));
    }
    if (tokenized_prompts.empty()) {
        res->error(format_error_response("\"input\" must not be empty", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = params.embd_normalize;
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize").get<int>();
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.res_type = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64)
        : json(responses);
    res->ok(root);
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_count_tokens(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    std::vector<raw_buffer> files;
    json body = json::parse(req.body);
    bool is_oai = false;

    switch (res_type) {
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            {
                is_oai = true;
            } break;
        case TASK_RESPONSE_TYPE_OAI_RESP:
            {
                is_oai = true;
                body = server_chat_convert_responses_to_chatcmpl(body);
            } break;
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            {
                body = server_chat_convert_anthropic_to_oai(body);
                server_apply_request_policy(body, req.headers);
            } break;
        default:
            res->error(format_error_response("invalid res_type", ERROR_TYPE_INVALID_REQUEST));
            return res;
    }

    common_chat_session session; // dummy, unused
    json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
    json prompt = body_parsed.at("prompt");
    // SRV_DBG("prompt = %s\n", prompt.dump().c_str());

    // TODO @ngxson : refactor this code block, move this to server-common and reuse it in other places
    size_t n_tokens;
    if (ctx_server.mctx != nullptr) {
        if (!prompt.is_string()) {
            throw std::runtime_error("for mtmd, input prompt must be a string.");
        }
        n_tokens = process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files, ctx_server.init_opt, true).size();
    } else {
        n_tokens = tokenize_mixed(ctx_server.vocab, prompt, true, true).size();
    }

    json response = {{"input_tokens", static_cast<int64_t>(n_tokens)}};
    if (is_oai) {
        response["object"] = "response.input_tokens";
    }
    res->ok(response);
    return res;
}

void server_routes::update_cached_responses(bool is_sleeping) {
    // caller is task_queue, so ctx_server can be accessed without holding locks
    std::unique_lock<std::mutex> lock(mutex_cache);

    if (is_sleeping) {
        cached_models  = get_res_models(*meta);
        cached_props   = get_res_props(*meta, params, true);
        cached_metrics = ctx_server.get_metrics();

        should_reset_buckets = false;

        SRV_DBG("%s\n", "cached responses updated");

    } else if (should_reset_buckets) {
        // a scrape during sleep already reported these buckets
        ctx_server.reset_metrics_bucket();

        should_reset_buckets = false;
    }
}
