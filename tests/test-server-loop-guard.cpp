#include "server-loop-guard.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cstdint>
#include <deque>
#include <vector>

static std::vector<int32_t> unique_pattern(size_t size, int32_t base) {
    std::vector<int32_t> result;
    result.reserve(size);
    for (size_t i = 0; i < size; ++i) {
        result.push_back(base + static_cast<int32_t>(i));
    }
    return result;
}

static void append_repeated(
        std::vector<int32_t> & output,
        const std::vector<int32_t> & pattern,
        size_t repeats) {
    for (size_t i = 0; i < repeats; ++i) {
        output.insert(output.end(), pattern.begin(), pattern.end());
    }
}

static std::deque<int32_t> recent_tokens(const std::vector<int32_t> & tokens, size_t context_size = 32768) {
    const size_t keep = std::min(tokens.size(), context_size);
    return { tokens.end() - keep, tokens.end() };
}

static size_t aligned_total(size_t repeated_size) {
    size_t total = std::max(server_loop_guard::min_generated, repeated_size);
    const size_t remainder =
        (total - server_loop_guard::min_generated) % server_loop_guard::check_stride;
    if (remainder != 0) {
        total += server_loop_guard::check_stride - remainder;
    }
    return total;
}

int main() {
    {
        auto tokens = unique_pattern(461, 1000);
        append_repeated(tokens, unique_pattern(10, 100), 5);
        assert(!server_loop_guard::detect(tokens.size(), recent_tokens(tokens)));
    }

    for (size_t period = server_loop_guard::min_period;
         period <= 300;
         ++period) {
        const size_t repeated_size = period * server_loop_guard::repeat_count;
        const size_t total = aligned_total(repeated_size);
        auto tokens = unique_pattern(total - repeated_size, 10000);
        append_repeated(tokens, unique_pattern(period, 100), server_loop_guard::repeat_count);
        const auto found = server_loop_guard::detect(tokens.size(), recent_tokens(tokens));
        assert(found.period == period);
        assert(found.repeats == server_loop_guard::repeat_count);
    }

    {
        auto tokens = unique_pattern(8000, 100000);
        assert(!server_loop_guard::detect(tokens.size(), recent_tokens(tokens)));
    }

    {
        auto tokens = unique_pattern(462, 1000);
        append_repeated(tokens, unique_pattern(10, 100), 5);
        const auto found = server_loop_guard::detect(tokens.size(), recent_tokens(tokens));
        assert(found.period == 10);
        assert(found.repeats == 5);
    }

    {
        auto tokens = unique_pattern(4, 1000);
        append_repeated(tokens, unique_pattern(300, 100), 5);
        const auto found = server_loop_guard::detect(tokens.size(), recent_tokens(tokens));
        assert(found.period == 300);
        assert(found.repeats == 5);
    }

    {
        auto tokens = unique_pattern(472, 1000);
        append_repeated(tokens, unique_pattern(10, 100), 4);
        assert(!server_loop_guard::detect(tokens.size(), recent_tokens(tokens)));
    }

    {
        auto tokens = unique_pattern(462, 1000);
        const auto pattern = unique_pattern(10, 100);
        append_repeated(tokens, pattern, 5);
        tokens[tokens.size() - 23] = -1;
        assert(!server_loop_guard::detect(tokens.size(), recent_tokens(tokens)));
    }

    {
        auto tokens = unique_pattern(463, 1000);
        append_repeated(tokens, unique_pattern(10, 100), 5);
        assert(!server_loop_guard::detect(tokens.size(), recent_tokens(tokens)));
    }

    {
        auto tokens = unique_pattern(478, 1000);
        append_repeated(tokens, unique_pattern(10, 100), 5);
        const auto found = server_loop_guard::detect(tokens.size(), recent_tokens(tokens));
        assert(found.period == 10);
    }

    {
        auto tokens = unique_pattern(462, 1000);
        append_repeated(tokens, std::vector<int32_t>{ 7 }, 50);
        const auto found = server_loop_guard::detect(tokens.size(), recent_tokens(tokens));
        assert(found.period == 10);
    }

    // Observed long loops, the 32K boundary, and the Agent's 64K context.
    for (size_t context : { size_t(32768), size_t(65536) }) {
        for (size_t period : { size_t(301), size_t(356), size_t(718), size_t(1024),
                               size_t(4096), context / server_loop_guard::repeat_count }) {
            const size_t size = period * server_loop_guard::repeat_count;
            const size_t total = aligned_total(size);
            auto tokens = unique_pattern(total - size, 100000);
            append_repeated(tokens, unique_pattern(period, 100), 5);
            const auto found = server_loop_guard::detect(total, recent_tokens(tokens, context));
            assert(found.period == period && found.repeats == 5);
            // One different token is not an exact five-block loop.
            tokens[tokens.size() - period - 1] = -1;
            assert(!server_loop_guard::detect(total, recent_tokens(tokens, context)));
        }
        assert(!server_loop_guard::detect(context, recent_tokens(unique_pattern(context, 100), context)));
    }

    // Four exact blocks must never be sufficient, even with repeated prefixes.
    for (size_t period : { size_t(356), size_t(718), size_t(4096) }) {
        const size_t total = aligned_total(period * 4);
        auto tokens = unique_pattern(total - period * 4, 100000);
        append_repeated(tokens, unique_pattern(period, 100), 4);
        assert(!server_loop_guard::detect(total, recent_tokens(tokens)));
    }

    // A bounded window cannot claim a loop whose fifth block has aged out.
    {
        std::vector<int32_t> tokens;
        append_repeated(tokens, unique_pattern(8192, 100), 5);
        assert(!server_loop_guard::detect(tokens.size(), recent_tokens(tokens, 32768)));
        assert(server_loop_guard::detect(tokens.size(), recent_tokens(tokens, 65536)).period == 8192);
    }

    // Actual streaming schedule, reset and continuation preserve long history.
    {
        const auto pattern = unique_pattern(718, 100);
        std::deque<int32_t> history;
        size_t first = 0;
        for (size_t i = 0; i < 4096; ++i) {
            history.push_back(pattern[i % pattern.size()]);
            const auto match = server_loop_guard::detect(i + 1, history);
            if (match) { first = i + 1; break; }
        }
        assert(first >= 5 * 718 && first < 5 * 718 + server_loop_guard::check_stride);
        const auto saved = history;
        assert(server_loop_guard::detect(first, saved).period == 718);
        history.clear();
        assert(!server_loop_guard::detect(0, history));
    }

    return 0;
}
