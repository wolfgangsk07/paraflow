#pragma once

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

// Private implementation, included after the scheduler definition. Snapshots own
// host bytes until a completion event; GPU uploads use the consumer's own stream.
// Default scope remains single-GPU/single-copy. Explicit pipeline callers may
// opt in with one owned arena per backend/copy and gate it at every submission.
struct ggml_backend_sched_input_staging {
    struct arena {
        ggml_backend_buffer_t buffer = nullptr;
        ggml_backend_event_t event = nullptr;
        size_t capacity = 0;
        bool pending = false;
    };
    struct upload {
        ggml_tensor * source;
        int backend;
        size_t offset;
        size_t bytes;
    };
    ggml_backend_sched_t owner;
    std::vector<arena> arenas;
    size_t limit;
    std::atomic<uint64_t> tensors{0}, bytes{0}, fences{0}, wait_ns{0}, fallbacks{0}, allocated{0};

    explicit ggml_backend_sched_input_staging(ggml_backend_sched_t sched)
        : owner(sched), arenas(sched->n_backends * sched->n_copies), limit(16 * 1024 * 1024) {
    }
    ggml_backend_sched_input_staging(const ggml_backend_sched_input_staging &) = delete;
    arena & get(int backend, int copy) {
        GGML_ASSERT(copy >= 0 && copy < owner->n_copies);
        return arenas[backend * owner->n_copies + copy];
    }
    void wait(int backend, int copy) {
        auto & a = get(backend, copy);
        if (!a.pending) return;
        const auto begin = std::chrono::steady_clock::now();
        if (a.event) ggml_backend_event_synchronize(a.event);
        else ggml_backend_synchronize(owner->backends[backend]);
        wait_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - begin).count(), std::memory_order_relaxed);
        fences.fetch_add(1, std::memory_order_relaxed);
        a.pending = false;
    }
    ~ggml_backend_sched_input_staging() {
        for (int b = 0; b < owner->n_backends; ++b) {
            for (int copy = 0; copy < owner->n_copies; ++copy) {
                wait(b, copy);
                auto & a = get(b, copy);
                if (a.event) ggml_backend_event_free(a.event);
                if (a.buffer) ggml_backend_buffer_free(a.buffer);
            }
        }
    }

    struct call {
        ggml_backend_sched_input_staging * state;
        std::unordered_map<ggml_tensor *, upload> uploads;
        std::vector<bool> usable, queued;
        int copy_slot;
        explicit call(ggml_backend_sched_t sched)
            : state(sched->input_staging), copy_slot(sched->cur_copy) {
            if (!state) return;
            usable.assign(sched->n_backends, false);
            queued.assign(sched->n_backends, false);
            if (sched->callback_eval) return;
            if (sched->callback_copy) {
                for (int si = 0; si < sched->n_splits; ++si) {
                    const auto & split = sched->splits[si];
                    for (int ii = 0; ii < split.n_inputs; ++ii) {
                        const auto * t = split.inputs[ii];
                        if (t->buffer && ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                            ggml_backend_buffer_is_host(t->buffer)) return;
                    }
                }
            }
            std::vector<size_t> required(sched->n_backends, 0);
            std::vector<bool> fits(sched->n_backends, true);
            for (int s = 0; s < sched->n_splits; ++s) {
                auto & split = sched->splits[s];
                const int b = split.backend_id;
                auto backend = sched->backends[b];
                auto dev = ggml_backend_get_device(backend);
                if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU ||
                    !backend->iface.set_tensor_async || !ggml_backend_dev_host_buffer_type(dev)) continue;
                for (int i = 0; i < split.n_inputs; ++i) {
                    auto * source = split.inputs[i];
                    auto buffer = source->view_src ? source->view_src->buffer : source->buffer;
                    if (!(source->flags & GGML_TENSOR_FLAG_INPUT) || !buffer ||
                        !ggml_backend_buffer_is_host(buffer) || !ggml_is_contiguous(source)) continue;
                    auto * destination = tensor_copy(source, b, sched->cur_copy);
                    if (!destination || !ggml_is_contiguous(destination) || uploads.count(destination)) continue;
                    const size_t size = ggml_nbytes(source);
                    if (!size) continue;
                    const size_t offset = GGML_PAD(required[b], 64);
                    if (size > state->limit || offset > state->limit - size) {
                        fits[b] = false;
                        continue;
                    }
                    GGML_ASSERT(size == ggml_nbytes(destination));
                    uploads.emplace(destination, upload{source, b, offset, size});
                    required[b] = offset + size;
                }
            }
            for (int b = 0; b < sched->n_backends; ++b) {
                if (!fits[b] || !required[b]) continue;
                auto & a = state->get(b, copy_slot);
                // The previous call may still be reading this host snapshot.
                state->wait(b, copy_slot);
                if (a.capacity < required[b]) {
                    size_t capacity = std::min<size_t>(state->limit, std::max<size_t>(65536, a.capacity));
                    while (capacity < required[b]) capacity = std::min(state->limit, capacity * 2);
                    auto type = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(sched->backends[b]));
                    auto buffer = ggml_backend_buft_alloc_buffer(type, capacity);
                    if (!buffer || !ggml_backend_buffer_get_base(buffer)) {
                        if (buffer) ggml_backend_buffer_free(buffer);
                        continue; // The original synchronous implementation remains available.
                    }
                    if (a.buffer) ggml_backend_buffer_free(a.buffer);
                    state->allocated.fetch_add(capacity - a.capacity, std::memory_order_relaxed);
                    a.buffer = buffer;
                    a.capacity = capacity;
                    if (!a.event) a.event = ggml_backend_event_new(ggml_backend_get_device(sched->backends[b]));
                }
                usable[b] = true;
            }
            for (const auto & entry : uploads) {
                const auto & u = entry.second;
                if (!usable[u.backend]) continue;
                auto * base = static_cast<char *>(ggml_backend_buffer_get_base(state->get(u.backend, copy_slot).buffer));
                std::memcpy(base + u.offset, u.source->data, u.bytes);
            }
        }
        bool copy(ggml_tensor * destination) {
            if (!state) return false;
            const auto it = uploads.find(destination);
            if (it == uploads.end() || !usable[it->second.backend]) return false;
            const auto & u = it->second;
            const auto & a = state->get(u.backend, copy_slot);
            const auto * base = static_cast<const char *>(ggml_backend_buffer_get_base(a.buffer));
            // FIFO ordering also protects destinations reused by later graph splits.
            ggml_backend_tensor_set_async(state->owner->backends[u.backend], destination, base + u.offset, 0, u.bytes);
            queued[u.backend] = true;
            state->tensors.fetch_add(1, std::memory_order_relaxed);
            state->bytes.fetch_add(u.bytes, std::memory_order_relaxed);
            return true;
        }
        void fallback() {
            if (state) state->fallbacks.fetch_add(1, std::memory_order_relaxed);
        }
        ~call() {
            if (!state || queued.empty()) return;
            for (int b = 0; b < state->owner->n_backends; ++b) {
                if (!queued[b]) continue;
                auto & a = state->get(b, copy_slot);
                if (a.event) ggml_backend_event_record(a.event, state->owner->backends[b]);
                a.pending = true;
            }
        }
    };
};
