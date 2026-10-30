#pragma once

#include <algorithm>
#include <climits>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

// Partition sorted active sequence ids into balanced ordered lanes. A gap in
// ids, or a boundary through consecutive ids, requires another KV stream run.
// Try every position of the larger groups instead of fixing them at the front.
// Return a plan only when it strictly reduces runs; minimize moves on ties.
inline std::vector<int> pipeline_balanced_compact_plan(
        const std::vector<int> & ids, const std::vector<int> & current, int lanes) {
    if (lanes <= 0 || ids.size() != current.size()) {
        throw std::invalid_argument("invalid pipeline lane plan dimensions");
    }
    const int n = int(ids.size());
    int old_runs = 0, gaps = 0;
    std::vector<int> gap_prefix(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        if (ids[i] < 0 || (i && ids[i] <= ids[i - 1]) || current[i] < 0 || current[i] >= lanes) {
            throw std::invalid_argument("invalid pipeline lane plan membership");
        }
        const bool gap = i == 0 || ids[i] != ids[i - 1] + 1;
        gaps += gap;
        old_runs += gap || current[i] != current[i - 1];
        gap_prefix[i + 1] = gaps;
    }
    // The normal C15 and C3 layouts stop here, without allocating a DP table.
    if (n == 0 || old_runs <= std::max(gaps, std::min(lanes, n))) {
        return {};
    }
    const int small = n / lanes, large_count = n % lanes;
    using cost = std::pair<int, int>; // runs, moves
    const cost unreachable = {INT_MAX / 4, INT_MAX / 4};
    cost best = {old_runs, INT_MAX / 4};
    std::vector<int> result;
    for (int shift = 0; shift < lanes; ++shift) {
        std::vector<std::vector<cost>> dp(lanes + 1, std::vector<cost>(large_count + 1, unreachable));
        std::vector<std::vector<int>> parent(lanes + 1, std::vector<int>(large_count + 1, -1));
        dp[0][0] = {0, 0};
        for (int k = 0; k < lanes; ++k) {
            const int lane = (k + shift) % lanes;
            std::vector<int> moves(n + 1, 0);
            for (int i = 0; i < n; ++i) {
                moves[i + 1] = moves[i] + (current[i] != lane);
            }
            for (int used = 0; used <= large_count; ++used) {
                if (dp[k][used] == unreachable) { continue; }
                const int start = k * small + used;
                for (int extra = 0; extra <= 1; ++extra) {
                    const int next_used = used + extra;
                    if (next_used > large_count || large_count - next_used > lanes - k - 1) { continue; }
                    const int end = start + small + extra;
                    if (end > n) { continue; }
                    const int runs = end == start ? 0 : 1 + gap_prefix[end] - gap_prefix[start + 1];
                    const cost candidate = {dp[k][used].first + runs,
                                            dp[k][used].second + moves[end] - moves[start]};
                    if (candidate < dp[k + 1][next_used]) {
                        dp[k + 1][next_used] = candidate;
                        parent[k + 1][next_used] = extra;
                    }
                }
            }
        }
        if (dp[lanes][large_count].first >= old_runs || dp[lanes][large_count] >= best) { continue; }
        best = dp[lanes][large_count];
        result.assign(n, -1);
        int used = large_count;
        for (int k = lanes; k > 0; --k) {
            const int extra = parent[k][used];
            const int start = (k - 1) * small + used - extra;
            const int end = k * small + used;
            std::fill(result.begin() + start, result.begin() + end, (k - 1 + shift) % lanes);
            used -= extra;
        }
    }
    return result;
}

// Verification rows are the capacity unit: ordinary=1, MTP(n)=n+1.
// Keep the old homogeneous fast path in the caller. For heterogeneous work,
// search a bounded set of LPT seeds and improve by single moves and exchanges.
// Balance comes before KV stream runs, then preserve existing ownership on ties.
// This is a placement heuristic, not a claim that rows predict elapsed GPU time.
inline std::vector<int> pipeline_work_balanced_plan(
        const std::vector<int> & ids, const std::vector<int> & current,
        const std::vector<int> & weights, int lanes) {
    if (lanes <= 0 || ids.size() != current.size() || ids.size() != weights.size()) {
        throw std::invalid_argument("invalid weighted pipeline dimensions");
    }
    const int n = int(ids.size());
    for (int i = 0; i < n; ++i) {
        if (ids[i] < 0 || (i && ids[i] <= ids[i-1]) || weights[i] <= 0 ||
                current[i] < 0 || current[i] >= lanes) {
            throw std::invalid_argument("invalid weighted pipeline membership");
        }
    }
    if (!n) { return {}; }
    using score = std::tuple<int64_t, int64_t, int, int>;
    auto cost = [&](const std::vector<int> & plan) {
        std::vector<int64_t> load(lanes, 0);
        int runs = 0, moves = 0;
        for (int i = 0; i < n; ++i) {
            load[plan[i]] += weights[i];
            runs += i == 0 || ids[i] != ids[i-1]+1 || plan[i] != plan[i-1];
            moves += plan[i] != current[i];
        }
        int64_t square = 0;
        for (auto value : load) { square += value * value; }
        return score{*std::max_element(load.begin(), load.end()), square, runs, moves};
    };
    auto best = current;
    auto best_cost = cost(best);
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return weights[a] > weights[b]; });
    // The current layout is also a seed, so an established optimum never churns.
    for (int seed = -1; seed < lanes; ++seed) {
        auto plan = current;
        if (seed >= 0) {
            std::vector<int64_t> load(lanes, 0);
            for (int i : order) {
                int lane = (current[i] + seed) % lanes;
                for (int k = 0; k < lanes; ++k) {
                    if (load[k] < load[lane]) { lane = k; }
                }
                plan[i] = lane;
                load[lane] += weights[i];
            }
        }
        auto value = cost(plan);
        for (int pass = 0; pass < 4 * n * lanes; ++pass) {
            auto next = plan;
            auto next_cost = value;
            for (int i = 0; i < n; ++i) {
                const int previous = plan[i];
                for (int lane = 0; lane < lanes; ++lane) {
                    if (lane == previous) { continue; }
                    plan[i] = lane;
                    const auto candidate = cost(plan);
                    if (candidate < next_cost) { next = plan; next_cost = candidate; }
                }
                plan[i] = previous;
                for (int j = i + 1; j < n; ++j) {
                    if (plan[i] == plan[j]) { continue; }
                    std::swap(plan[i], plan[j]);
                    const auto candidate = cost(plan);
                    if (candidate < next_cost) { next = plan; next_cost = candidate; }
                    std::swap(plan[i], plan[j]);
                }
            }
            if (next_cost == value) { break; }
            plan = std::move(next);
            value = next_cost;
        }
        if (value < best_cost) { best = std::move(plan); best_cost = value; }
    }
    return best == current ? std::vector<int>{} : best;
}

// Ordered partitions can cut a consecutive run even when swapping whole actors
// between non-adjacent groups would preserve both load and that run. Reuse the
// bounded placement search with equal weights, only above the run lower bound.
inline std::vector<int> pipeline_balanced_stream_plan(
        const std::vector<int> & ids, const std::vector<int> & current, int lanes) {
    auto result = pipeline_balanced_compact_plan(ids, current, lanes);
    const auto & ordered = result.empty() ? current : result;
    int gaps = 0, runs = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
        const bool gap = !i || ids[i] != ids[i-1] + 1;
        gaps += gap;
        runs += gap || ordered[i] != ordered[i-1];
    }
    const int lower_bound = std::max(gaps, std::min(lanes, int(ids.size())));
    if (runs <= lower_bound) { return result; }
    auto best = ordered;
    const std::vector<int> weights(ids.size(), 1);
    bool improved = false;
    // Seed rotations depend on the layout. Finish all strictly improving
    // steps now, so a cached membership never hides another useful exchange.
    for (size_t attempt = 0; attempt < ids.size() && runs > lower_bound; ++attempt) {
        auto candidate = pipeline_work_balanced_plan(ids, best, weights, lanes);
        if (candidate.empty()) { break; }
        int candidate_runs = 0;
        for (size_t i = 0; i < ids.size(); ++i) {
            candidate_runs += !i || ids[i] != ids[i-1] + 1 || candidate[i] != candidate[i-1];
        }
        if (candidate_runs >= runs) { break; }
        runs = candidate_runs;
        best = std::move(candidate);
        improved = true;
    }
    return improved ? best : result;
}

// Exact physical split count for equal complete output tails with sequential
// non-unified KV and the complete-tail packing policy. Each contiguous id run
// admits at most floor(capacity / weight) actors into one physical graph.
inline std::pair<int, int> pipeline_complete_tail_graph_cost(
        const std::vector<int> & ids, const std::vector<int> & plan,
        int lanes, int weight, int capacity) {
    if (lanes <= 0 || ids.size() != plan.size() || weight <= 0 || capacity < weight) {
        throw std::invalid_argument("invalid complete-tail graph dimensions");
    }
    std::vector<int> graphs(lanes, 0), last(lanes, -2), run(lanes, 0);
    const int actors_per_graph = capacity / weight;
    for (size_t i = 0; i < ids.size(); ++i) {
        const int lane = plan[i];
        if (ids[i] < 0 || (i && ids[i] <= ids[i-1]) || lane < 0 || lane >= lanes) {
            throw std::invalid_argument("invalid complete-tail graph membership");
        }
        if (ids[i] != last[lane] + 1) { run[lane] = 0; }
        if (run[lane]++ % actors_per_graph == 0) { ++graphs[lane]; }
        last[lane] = ids[i];
    }
    return {*std::max_element(graphs.begin(), graphs.end()),
            std::accumulate(graphs.begin(), graphs.end(), 0)};
}

inline std::vector<int> pipeline_physical_compact_plan(
        const std::vector<int> & ids, const std::vector<int> & current,
        int lanes, int weight, int capacity) {
    const auto previous = pipeline_complete_tail_graph_cost(ids, current, lanes, weight, capacity);
    auto best_cost = previous;
    auto best = current;
    // Preserve actor balance. Compare both existing balanced candidates with
    // the current ownership; fewer id runs alone need not mean fewer graphs.
    for (size_t attempt = 0; attempt < ids.size(); ++attempt) {
        auto next = best;
        auto next_cost = best_cost;
        for (const auto & candidate : {pipeline_balanced_compact_plan(ids, best, lanes),
                                      pipeline_balanced_stream_plan(ids, best, lanes)}) {
            if (candidate.empty()) { continue; }
            const auto value = pipeline_complete_tail_graph_cost(ids, candidate, lanes, weight, capacity);
            if (value.first <= best_cost.first && value.second < best_cost.second && value < next_cost) {
                next = candidate;
                next_cost = value;
            }
        }
        if (next == best) { break; }
        best = std::move(next);
        best_cost = next_cost;
    }
    return best == current ? std::vector<int>{} : best;
}

// Lanes without highest-priority speculative work may have at most one target
// car ahead of those with it. The anchor lanes remain independently ready-driven.
inline bool pipeline_work_round_allowed(const std::vector<uint64_t> & issued,
                                       const std::vector<bool> & speculative, int lane) {
    if (issued.size() != speculative.size() || lane < 0 || lane >= int(issued.size())) {
        throw std::invalid_argument("invalid pipeline work round");
    }
    if (speculative[lane]) { return true; }
    for (size_t i = 0; i < issued.size(); ++i) {
        if (speculative[i] && issued[lane] > issued[i]) { return false; }
    }
    return true;
}


// Complete short verification tails are split into consecutive runs of equal
// length. Nominal row balance alone does not minimize physical target graphs.
inline std::pair<int, int> pipeline_mixed_tail_graph_cost(
        const std::vector<int> & ids, const std::vector<int> & plan,
        const std::vector<int> & weights, int lanes, int capacity) {
    if (lanes <= 0 || ids.size() != plan.size() || ids.size() != weights.size()) {
        throw std::invalid_argument("invalid mixed complete-tail dimensions");
    }
    std::vector<int> graphs(lanes, 0), last(lanes, -2), length(lanes, 0), run(lanes, 0);
    for (size_t i = 0; i < ids.size(); ++i) {
        const int lane = plan[i], weight = weights[i];
        if (ids[i] < 0 || (i && ids[i] <= ids[i-1]) || lane < 0 || lane >= lanes ||
                weight <= 0 || weight > capacity) {
            throw std::invalid_argument("invalid mixed complete-tail membership");
        }
        if (ids[i] != last[lane] + 1 || weight != length[lane]) { run[lane] = 0; }
        if (run[lane]++ % (capacity / weight) == 0) { ++graphs[lane]; }
        last[lane] = ids[i]; length[lane] = weight;
    }
    return {*std::max_element(graphs.begin(), graphs.end()),
            std::accumulate(graphs.begin(), graphs.end(), 0)};
}

inline std::vector<int> pipeline_mixed_physical_compact_step(
        const std::vector<int> & ids, const std::vector<int> & current,
        const std::vector<int> & weights, int lanes, int capacity) {
    const auto & baseline = current;
    if (ids.empty()) { return {}; }
    const auto baseline_graphs = pipeline_mixed_tail_graph_cost(ids, baseline, weights, lanes, capacity);
    auto rows = [&](const std::vector<int> & plan) {
        std::vector<int64_t> load(lanes, 0);
        for (size_t i = 0; i < ids.size(); ++i) { load[plan[i]] += weights[i]; }
        return load;
    };
    const auto baseline_rows = rows(baseline);
    // Draft work is tied to each lane's device and worker. Fewer target
    // graphs must not concentrate draft rows or draft actors on fewer lanes.
    // Require every prefix of each descending load vector to stay bounded by
    // the accepted weighted plan, preserving both peak and distributed work.
    auto draft_load = [&](const std::vector<int> & plan, bool actors) {
        std::vector<int64_t> load(lanes, 0);
        for (size_t i = 0; i < ids.size(); ++i) {
            if (weights[i] > 1) { load[plan[i]] += actors ? 1 : weights[i] - 1; }
        }
        std::sort(load.begin(), load.end(), std::greater<int64_t>());
        return load;
    };
    const auto baseline_draft_rows = draft_load(baseline, false);
    const auto baseline_draft_actors = draft_load(baseline, true);
    auto draft_bounded = [&](const std::vector<int> & plan) {
        for (bool actors : {false, true}) {
            const auto candidate = draft_load(plan, actors);
            const auto & reference = actors ? baseline_draft_actors : baseline_draft_rows;
            int64_t candidate_sum = 0, reference_sum = 0;
            for (int lane = 0; lane < lanes; ++lane) {
                candidate_sum += candidate[lane]; reference_sum += reference[lane];
                if (candidate_sum > reference_sum) { return false; }
            }
        }
        return true;
    };
    const int64_t row_limit = std::max<int64_t>(capacity, *std::max_element(baseline_rows.begin(), baseline_rows.end()));
    const int occupied_required = std::min(lanes, int(ids.size()));
    using score = std::tuple<int, int, int64_t, int64_t, int, int>;
    const score invalid{INT_MAX, INT_MAX, INT64_MAX, INT64_MAX, INT_MAX, INT_MAX};
    auto cost = [&](const std::vector<int> & plan) {
        const auto graph = pipeline_mixed_tail_graph_cost(ids, plan, weights, lanes, capacity);
        const auto load = rows(plan);
        const auto maximum = *std::max_element(load.begin(), load.end());
        if (graph.first > baseline_graphs.first || graph.second > baseline_graphs.second ||
                maximum > row_limit || !draft_bounded(plan) ||
                std::count_if(load.begin(), load.end(), [](int64_t x) { return x > 0; }) < occupied_required) {
            return invalid;
        }
        int64_t square = 0;
        int runs = 0, moves = 0;
        for (auto x : load) { square += x*x; }
        for (size_t i = 0; i < ids.size(); ++i) {
            runs += !i || ids[i] != ids[i-1]+1 || plan[i] != plan[i-1];
            moves += plan[i] != current[i];
        }
        return score{graph.first, graph.second, maximum, square, runs, moves};
    };
    auto best = baseline;
    auto best_cost = cost(best);
    // Whole physical groups provide a seed that row-balancing cannot reach by
    // a single move: e.g. [MTP,ordinary] / [MTP] / [MTP] needs the ordinary
    // actor relocated while two adjacent MTP actors are combined elsewhere.
    std::vector<std::pair<size_t,size_t>> groups;
    for (size_t i = 0; i < ids.size();) {
        size_t end = i+1;
        while (end < ids.size() && ids[end] == ids[end-1]+1 && weights[end] == weights[i] &&
                end-i < size_t(capacity/weights[i])) { ++end; }
        groups.push_back({i,end}); i = end;
    }
    std::stable_sort(groups.begin(),groups.end(),[&](auto a,auto b) {
        return (a.second-a.first)*weights[a.first] > (b.second-b.first)*weights[b.first];
    });
    std::vector<int> packed(ids.size(),0), group_count(lanes,0);
    std::vector<int64_t> packed_rows(lanes,0);
    for (auto group : groups) {
        int chosen = 0;
        auto value = [&](int lane) {
            int moves = 0;
            for (size_t i = group.first; i < group.second; ++i) { moves += current[i] != lane; }
            return std::make_tuple(group_count[lane],packed_rows[lane],moves,lane);
        };
        for (int lane = 1; lane < lanes; ++lane) { if (value(lane) < value(chosen)) { chosen = lane; } }
        for (size_t i = group.first; i < group.second; ++i) { packed[i] = chosen; }
        ++group_count[chosen]; packed_rows[chosen] += (group.second-group.first)*weights[group.first];
    }
    // The caller supplies the accepted weighted layout. Search only for a
    // physical improvement; do not re-run the opposing row optimizer here.
    for (const auto & seed : {current, packed}) {
        auto plan = seed;
        auto value = cost(plan);
        if (value == invalid) { continue; }
        for (size_t pass = 0; pass < 4 * ids.size() * size_t(lanes); ++pass) {
            auto next = plan;
            auto next_cost = value;
            for (size_t i = 0; i < ids.size(); ++i) {
                const int old = plan[i];
                for (int lane = 0; lane < lanes; ++lane) {
                    if (lane == old) { continue; }
                    plan[i] = lane;
                    const auto candidate = cost(plan);
                    if (candidate < next_cost) { next = plan; next_cost = candidate; }
                }
                plan[i] = old;
                for (size_t j = i+1; j < ids.size(); ++j) {
                    if (plan[i] == plan[j]) { continue; }
                    std::swap(plan[i], plan[j]);
                    const auto candidate = cost(plan);
                    if (candidate < next_cost) { next = plan; next_cost = candidate; }
                    std::swap(plan[i], plan[j]);
                }
            }
            if (next_cost == value) { break; }
            plan = std::move(next); value = next_cost;
        }
        if (value < best_cost) { best = std::move(plan); best_cost = value; }
    }
    // Relax row balance only for a strict reduction of physical graph work;
    // never increase the worst lane, concentrate draft work, or empty useful lanes.
    const auto graph = pipeline_mixed_tail_graph_cost(ids, best, weights, lanes, capacity);
    if (graph.first > baseline_graphs.first || graph.second >= baseline_graphs.second) { return {}; }
    return best == current ? std::vector<int>{} : best;
}

inline std::vector<int> pipeline_mixed_physical_compact_plan(
        const std::vector<int> & ids, const std::vector<int> & current,
        const std::vector<int> & weights, int lanes, int capacity, bool * converged = nullptr) {
    if (converged) { *converged = false; }
    auto result = current;
    for (size_t pass = 0; pass <= ids.size(); ++pass) {
        auto next = pipeline_mixed_physical_compact_step(ids, result, weights, lanes, capacity);
        if (next.empty()) {
            if (converged) { *converged = true; }
            return result == current ? std::vector<int>{} : result;
        }
        result = std::move(next);
    }
    // Each accepted step strictly decreases total graphs; this bound cannot
    // be exhausted. Preserve the accepted weighted input if a future change
    // violates that invariant.
    return {};
}
