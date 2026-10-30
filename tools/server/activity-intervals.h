#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

namespace gpu_activity {
struct interval { uint64_t start, end; };
using intervals = std::vector<interval>;
inline intervals unite(intervals rows) {
    std::sort(rows.begin(), rows.end(), [](const interval & a, const interval & b) {
        return a.start < b.start || (a.start == b.start && a.end < b.end);
    });
    intervals result;
    for (const auto & r : rows) {
        if (r.end <= r.start) { continue; }
        if (result.empty() || result.back().end < r.start) { result.push_back(r); }
        else { result.back().end = std::max(result.back().end, r.end); }
    }
    return result;
}
inline uint64_t length(const intervals & rows) {
    uint64_t result = 0;
    for (const auto & r : rows) { result += r.end - r.start; }
    return result;
}
inline uint64_t overlap(const intervals & rows, uint64_t start, uint64_t end) {
    uint64_t result = 0;
    for (const auto & r : rows) {
        if (r.start >= end) { break; }
        if (r.end > start) { result += std::min(r.end, end) - std::max(r.start, start); }
    }
    return result;
}
inline intervals intersect(const intervals & a, const intervals & b) {
    intervals result; size_t i=0,j=0;
    while(i<a.size() && j<b.size()) {
        const auto lo=std::max(a[i].start,b[j].start),hi=std::min(a[i].end,b[j].end);
        if(hi>lo)result.push_back({lo,hi});
        if(a[i].end<b[j].end)++i;else ++j;
    }
    return result;
}
inline intervals subtract(const intervals & a,const intervals & b) {
    intervals result;
    for(const auto & r:a) {
        uint64_t pos=r.start;
        for(const auto & x:b) {
            if(x.end<=pos)continue;
            if(x.start>=r.end)break;
            if(x.start>pos)result.push_back({pos,std::min(x.start,r.end)});
            pos=std::max(pos,x.end);
            if(pos>=r.end)break;
        }
        if(pos<r.end)result.push_back({pos,r.end});
    }
    return result;
}
}
