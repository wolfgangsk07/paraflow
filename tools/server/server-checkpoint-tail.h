#pragma once

#include <algorithm>
#include <cstdint>

// Prompt snapshots are taken BEFORE the car, at n_start. Admission slices
// must not turn the final physical ubatch into a checkpoint per small car.
// Keep its first available snapshot and the explicit final-four-token point.
// User-message boundaries retain their separate existing eligibility rules.
template <typename Checkpoints>
bool server_checkpoint_tail_needed(bool near_end, int64_t n_prompt, int64_t n_start,
        int64_t n_ubatch, int64_t physical_batch, int id_task, const Checkpoints & checkpoints) {
    if (!near_end) {
        return false;
    }
    const int64_t first = std::max<int64_t>(0, n_prompt - std::min(physical_batch, n_ubatch + 4));
    const int64_t final = n_prompt - std::min<int64_t>(physical_batch, 4);
    if (n_start == final) {
        return true;
    }
    for (const auto & checkpoint : checkpoints) {
        if (checkpoint.id_task == id_task && checkpoint.n_tokens >= first && checkpoint.n_tokens <= n_start) {
            return false;
        }
    }
    return true;
}
