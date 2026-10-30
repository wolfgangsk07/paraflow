#pragma once

#include <algorithm>
#include <cctype>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// Read-only, opaque correlation. It never changes prompts, sampling or scheduling.
inline bool server_valid_observation_id(const std::string & value) {
    return value.size() == 32 && std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

inline std::string server_observation_id(const std::map<std::string, std::string> & headers) {
    std::string result;
    bool found = false;
    for (const auto & item : headers) {
        auto name = item.first;
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::tolower(c); });
        if (name != "x-llama-request-id") { continue; }
        if (found || !server_valid_observation_id(item.second)) { return {}; }
        found = true;
        result = item.second;
    }
    return result;
}

struct server_request_observation {
    std::string request_id;
    int task_id;
    // none = running; vip/capacity = paused. Storage location is not exposed.
    std::string reason;
};

inline std::string server_observation_metrics(const std::vector<server_request_observation> & rows) {
    std::ostringstream out;
    out << "# HELP llamacpp:request_states_supported Opaque instantaneous request observations\n"
        << "# TYPE llamacpp:request_states_supported gauge\n"
        << "llamacpp:request_states_supported 1\n"
        << "# HELP llamacpp:request_state Instantaneous state of an identified active request\n"
        << "# TYPE llamacpp:request_state gauge\n";
    for (const auto & row : rows) {
        if (!server_valid_observation_id(row.request_id) || row.task_id < 0 ||
                (row.reason != "none" && row.reason != "vip" && row.reason != "capacity")) { continue; }
        out << "llamacpp:request_state{request_id=\"" << row.request_id
            << "\",task_id=\"" << row.task_id
            << "\",state=\"" << (row.reason == "none" ? "running" : "paused")
            << "\",reason=\"" << row.reason << "\"} 1\n";
    }
    return out.str();
}
