#pragma once
#include <cstdint>
#include <utility>
#include <vector>
// Every equal-sequence shape within a small decode microbatch. Reserving each
// legal shape covers both recurrent single-token and multi-token paths.
inline std::vector<std::pair<uint32_t, uint32_t>> llama_pipeline_reserve_shapes(
        uint32_t max_tokens, uint32_t max_sequences, bool tight) {
    if (!max_tokens || !max_sequences) return {};
    if (!tight || max_tokens > 32) return {{max_tokens, max_sequences}};
    std::vector<std::pair<uint32_t, uint32_t>> result;
    for (uint32_t sequences = 1; sequences <= max_sequences && sequences <= max_tokens; ++sequences) {
        for (uint32_t tokens_per_sequence = 1; tokens_per_sequence <= max_tokens / sequences; ++tokens_per_sequence) {
            result.emplace_back(tokens_per_sequence * sequences, sequences);
        }
    }
    return result;
}
