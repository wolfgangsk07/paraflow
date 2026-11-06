#pragma once

#include <algorithm>
#include <cstddef>

namespace server_loop_guard {

constexpr size_t min_generated    = 512;
constexpr size_t check_stride     = 16;
constexpr size_t min_period       = 10;
constexpr size_t repeat_count     = 5;

struct match {
    size_t period  = 0;
    size_t repeats = 0;

    explicit operator bool() const {
        return period != 0;
    }
};

// Detect an exact repeated suffix in a bounded, chronologically ordered token
// history. Cheap boundary checks reject almost every candidate period before
// any full block comparison. Callers retain at most their slot context size
// in generated tokens. Scan every feasible period in that bounded history,
// including long repeated records; there is no separate 300-token blind spot.
template <typename TokenContainer>
match detect(size_t total_generated, const TokenContainer & recent) {
    if (total_generated < min_generated ||
            (total_generated - min_generated) % check_stride != 0) {
        return {};
    }

    const size_t largest_period = recent.size() / repeat_count;
    for (size_t period = min_period; period <= largest_period; ++period) {
        const size_t end = recent.size();

        bool boundary_match = true;
        for (size_t repeat = 1; repeat < repeat_count; ++repeat) {
            if (recent[end - 1] != recent[end - 1 - repeat * period]) {
                boundary_match = false;
                break;
            }
        }
        if (!boundary_match) {
            continue;
        }

        bool blocks_match = true;
        const size_t reference = end - period;
        for (size_t repeat = 1; repeat < repeat_count && blocks_match; ++repeat) {
            const size_t candidate = end - (repeat + 1) * period;
            for (size_t i = 0; i < period; ++i) {
                if (recent[reference + i] != recent[candidate + i]) {
                    blocks_match = false;
                    break;
                }
            }
        }
        if (blocks_match) {
            return { period, repeat_count };
        }
    }

    return {};
}

} // namespace server_loop_guard
