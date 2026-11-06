#pragma once
#include "llama-model.h"
#include "llama-cparams.h"
#include <algorithm>
inline uint32_t llama_prefill_initial_reserve_sequences(
        const llama_model & model, const llama_cparams & params, uint32_t requested) {
    if (params.training || model.hparams.no_alloc ||
            (model.arch != LLM_ARCH_QWEN3NEXT && model.arch != LLM_ARCH_QWEN35 && model.arch != LLM_ARCH_QWEN35MOE)) {
        return requested;
    }
    return std::min<uint32_t>(requested, 1);
}

inline uint32_t llama_pipeline_verify_ubatch_limit(const llama_cparams & params, uint32_t lanes) {
    const uint64_t rows_per_seq = uint64_t(params.n_rs_seq) + 1;
    if (params.n_rs_seq == 0 || rows_per_seq >= 32) {
        return params.n_ubatch;
    }
    lanes = std::max<uint32_t>(lanes, 1);
    const uint64_t lane_seqs = (uint64_t(params.n_seq_max) + lanes - 1) / lanes;
    // Keep the existing small-batch allowance; grow for complete lane tails.
    const uint64_t rows = std::max<uint64_t>(32, lane_seqs * rows_per_seq);
    return uint32_t(std::min<uint64_t>(params.n_ubatch, rows));
}
