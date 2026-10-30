#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// Consecutive source rows can be copied together only while their destination
// rows are also consecutive. Keep the original order between separate runs.
template<typename F>
void llama_for_each_index_run(const std::vector<int32_t> & indices, F && apply) {
    for (size_t first = 0; first < indices.size();) {
        size_t end = first + 1;
        while (end < indices.size() && int64_t(indices[end]) == int64_t(indices[first]) + int64_t(end - first)) {
            ++end;
        }
        apply(first, indices[first], end - first);
        first = end;
    }
}
