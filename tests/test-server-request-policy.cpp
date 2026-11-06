#ifdef NDEBUG
#undef NDEBUG
#endif
#include "server-request-policy.h"
#include <cassert>

int main() {
    using json = common_json;
    json original = {{"messages", json::array()},
        {"chat_template_kwargs", {{"enable_thinking", true}, {"other", "kept"}}}};
    auto request = original;
    server_apply_request_policy(request, {});
    assert(request == original);
    server_apply_request_policy(request, {{"X-LLAMA-Priority", "1"},
        {"X-LLAMA-Enable-Thinking", "false"}, {"Authorization", "not-copied"}});
    assert(request["priority"] == 1);
    assert(request["chat_template_kwargs"]["enable_thinking"] == false);
    assert(request["chat_template_kwargs"]["other"] == "kept");
    assert(!request.contains("Authorization"));
    assert(original["chat_template_kwargs"]["enable_thinking"] == true);
    auto next = original;
    server_apply_request_policy(next, {});
    assert(next == original); // No cross-request or main-pipeline setting leak.
    request["thinking_budget_tokens"] = 10000;
    server_apply_request_policy(request, {{"x-llama-enable-thinking", "false"}});
    assert(!request.contains("thinking_budget_tokens"));
    for (bool vip : {false, true}) {
        for (bool mtp : {false, true}) {
            for (int n_max : {0, 1, 3, 5}) {
                auto current = original;
                server_apply_request_policy(current, {{"X-LLAMA-Priority", vip ? "1" : "0"},
                    {"x-llama-mtp", mtp ? "true" : "false"},
                    {"X-LLAMA-MTP-N-Max", std::to_string(n_max)},
                    {"X-LLAMA-Enable-Thinking", "true"}});
                assert(current["priority"] == (vip ? 1 : 0));
                assert(current["mtp"] == mtp);
                assert(current["speculative.n_max"] == n_max);
                assert(current["chat_template_kwargs"]["enable_thinking"] == true);
                auto unrelated = original;
                server_apply_request_policy(unrelated, {});
                assert(unrelated == original);
            }
        }
    }
    for (const char * invalid : {"", "-1", "+1", " 3", "3 ", "1.5", "true", "0x3", "2147483648", "99999999999999999999"}) {
        bool rejected = false;
        try { server_apply_request_policy(request, {{"X-LLAMA-MTP-N-Max", invalid}}); }
        catch (const std::invalid_argument &) { rejected = true; }
        assert(rejected);
    }
    for (const auto & headers : {
        std::map<std::string, std::string>{{"X-LLAMA-Priority", "-1"}},
        std::map<std::string, std::string>{{"X-LLAMA-Priority", "true"}},
        std::map<std::string, std::string>{{"X-LLAMA-Enable-Thinking", "0"}},
        std::map<std::string, std::string>{{"X-LLAMA-Priority", "1"}, {"x-llama-priority", "0"}},
        std::map<std::string, std::string>{{"X-LLAMA-MTP", "1"}},
        std::map<std::string, std::string>{{"X-LLAMA-MTP", "true"}, {"x-llama-mtp", "false"}},
        std::map<std::string, std::string>{{"X-LLAMA-MTP-N-Max", "1"}, {"x-llama-mtp-n-max", "2"}}}) {
        bool rejected = false;
        try { server_apply_request_policy(request, headers); }
        catch (const std::invalid_argument &) { rejected = true; }
        assert(rejected);
    }
    return 0;
}
