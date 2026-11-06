#include "server-common.h"

#include <cstdlib>

static void require(bool condition) {
    if (!condition) {
        std::abort();
    }
}

int main() {
    const llama_tokens raw = { 10, 20, 30 };

    // Loading a projector makes has_mtmd true even for a text-only request.
    // The old continuation guard called get_tokens() here and aborted.
    server_tokens projector_text(raw, true);
    require(projector_text.has_mtmd && !projector_text.has_media());
    const uint64_t original = projector_text.continuation_fingerprint();

    server_tokens cloned = projector_text.clone();
    require(!cloned.has_media());
    require(cloned.continuation_fingerprint() == original);

    cloned.push_back(40);
    require(cloned.continuation_fingerprint() != original);

    server_tokens text_only(raw, false);
    require(!text_only.has_media());
    require(text_only.continuation_fingerprint() != original);

    return 0;
}
