#pragma once

#include "json.h"
#include <algorithm>
#include <cctype>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>

// Transport policy for clients whose SDK cannot add OpenAI extension fields.
// Applied to each converted Anthropic request, never to server defaults.
inline void server_apply_request_policy(
        common_json & body, const std::map<std::string, std::string> & headers) {
    std::map<std::string, std::string> policy;
    for (const auto & item : headers) {
        auto name = item.first;
        std::transform(name.begin(), name.end(), name.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (name != "x-llama-priority" && name != "x-llama-enable-thinking" &&
                name != "x-llama-mtp" && name != "x-llama-mtp-n-max") {
            continue;
        }
        if (!policy.emplace(name, item.second).second) {
            throw std::invalid_argument("Duplicate inference policy header");
        }
    }
    auto priority = policy.find("x-llama-priority");
    if (priority != policy.end()) {
        if (priority->second != "0" && priority->second != "1") {
            throw std::invalid_argument("X-LLAMA-Priority must be 0 or 1");
        }
        body["priority"] = priority->second == "1" ? 1 : 0;
    }
    auto thinking = policy.find("x-llama-enable-thinking");
    if (thinking != policy.end()) {
        if (thinking->second != "true" && thinking->second != "false") {
            throw std::invalid_argument("X-LLAMA-Enable-Thinking must be true or false");
        }
        body["chat_template_kwargs"]["enable_thinking"] = thinking->second == "true";
        if (thinking->second == "false") {
            body.erase("thinking_budget_tokens");
        }
    }
    auto mtp = policy.find("x-llama-mtp");
    if (mtp != policy.end()) {
        if (mtp->second != "true" && mtp->second != "false") {
            throw std::invalid_argument("X-LLAMA-MTP must be true or false");
        }
        body["mtp"] = mtp->second == "true";
    }
    auto draft_max = policy.find("x-llama-mtp-n-max");
    if (draft_max != policy.end()) {
        const auto & value = draft_max->second;
        if (value.empty() || value.size() > 10) {
            throw std::invalid_argument("X-LLAMA-MTP-N-Max must be a non-negative integer");
        }
        int parsed = 0;
        for (unsigned char c : value) {
            if (c < '0' || c > '9' || parsed > (std::numeric_limits<int>::max() - (c - '0')) / 10) {
                throw std::invalid_argument("X-LLAMA-MTP-N-Max is invalid or too large");
            }
            parsed = parsed * 10 + (c - '0');
        }
        // The request schema also enforces the configured draft-length ceiling.
        body["speculative.n_max"] = parsed;
    }
}
