#include "server-task.h"

#include <condition_variable>
#include <deque>
#include <exception>
#include <future>
#include <limits>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

// Exercise otherwise unreachable allocator boundaries without a production hook.
#define private public
#include "server-queue.h"
#undef private

#include <cstdio>
#include <stdexcept>

static void require(bool condition, const char * message) {
    if (!condition) { throw std::runtime_error(message); }
}

static void test_id_boundaries() {
    server_queue queue;
    for (const server_task_id boundary : {INT64_C(2147483647), INT64_C(4294967295)}) {
        queue.id = boundary - 1;
        for (int i = 0; i < 4; ++i) {
            require(queue.get_new_id() == boundary - 1 + i, "ID narrowed or wrapped at 32 bits");
        }
    }
    queue.id = std::numeric_limits<server_task_id>::max() - 1;
    std::vector<server_task> batch;
    batch.emplace_back(SERVER_TASK_TYPE_METRICS);
    batch.emplace_back(SERVER_TASK_TYPE_METRICS);
    bool threw = false;
    try { queue.post(std::move(batch)); } catch (const std::overflow_error &) { threw = true; }
    require(threw && queue.queue_tasks.empty(), "Exhausted batch must not partially enqueue");
    require(queue.get_new_id() == std::numeric_limits<server_task_id>::max() - 1, "Last usable ID lost");
    threw = false;
    try { queue.get_new_id(); } catch (const std::overflow_error &) { threw = true; }
    require(threw && queue.id == std::numeric_limits<server_task_id>::max(), "Allocator overflowed");
}

static void test_response_and_cancel() {
    const server_task_id low = 7, high = (INT64_C(1) << 32) + low;
    server_response responses;
    responses.add_waiting_task_ids({low, high});
    for (auto id : {high, low}) {
        auto result = std::make_unique<server_task_result_error>();
        result->id = id;
        responses.send(std::move(result));
    }
    require(responses.recv(low)->id == low, "High ID response leaked to low ID request");
    require(responses.recv(high)->id == high, "High ID response lost");

    server_queue queue;
    for (auto id : {low, high}) {
        server_task task(SERVER_TASK_TYPE_COMPLETION); task.id = id;
        queue.post(std::move(task));
        server_task deferred(SERVER_TASK_TYPE_COMPLETION); deferred.id = id;
        queue.defer(std::move(deferred));
    }
    server_task cancel(SERVER_TASK_TYPE_CANCEL);
    cancel.id = high + 1; cancel.id_target = high;
    queue.post(std::move(cancel), true);
    require(queue.queue_tasks.size() == 2 && queue.queue_tasks.back().id == low, "Cancel matched truncated ID");
    require(queue.queue_tasks_deferred.size() == 1 && queue.queue_tasks_deferred.front().id == low, "Deferred cancel matched truncated ID");
    queue.request_update();
    server_task invalid_cancel(SERVER_TASK_TYPE_CANCEL);
    invalid_cancel.id = high + 3; invalid_cancel.id_target = -2;
    queue.post(std::move(invalid_cancel));
    require(std::count_if(queue.queue_tasks.begin(), queue.queue_tasks.end(), [](const server_task & task) {
        return task.type == SERVER_TASK_TYPE_NEXT_RESPONSE;
    }) == 1, "Invalid cancellation removed an internal wakeup");
    server_task parent(SERVER_TASK_TYPE_COMPLETION); parent.id = high;
    parent.add_child(high, high + 2);
    require(parent.child_tasks[0].id_parent == high, "Parent ID narrowed");
    std::vector<server_task> parents;
    parents.push_back(std::move(parent));
    require(server_task::get_list_id(parents).count(high + 2) == 1, "Child ID narrowed");
}

static void test_scheduler_wakeups() {
    server_queue queue;
    queue.id = INT64_C(4294967295);
    for (int i = 0; i < 100000; ++i) { queue.request_update(); }
    require(queue.queue_tasks.size() == 1 && queue.queue_tasks.front().id == -2 &&
            queue.id == INT64_C(4294967295), "Internal wakes consumed request IDs or failed to coalesce");
    std::atomic<int> tasks{0};
    int updates = 0;
    queue.on_new_task([&](server_task && task, bool yielding) {
        if (yielding && task.type == SERVER_TASK_TYPE_NEXT_RESPONSE) { return false; }
        if (task.type == SERVER_TASK_TYPE_METRICS) { ++tasks; }
        return true;
    });
    queue.on_update_slots([&]() {
        ++updates;
        if (updates == 2) {
            queue.request_update();
            server_task task(SERVER_TASK_TYPE_METRICS); task.id = queue.get_new_id();
            queue.post(std::move(task));
            queue.yield_to_queue([&]() {
                const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(2);
                while (!tasks.load() && std::chrono::steady_clock::now() < end) { std::this_thread::yield(); }
            });
        }
        if (updates < 10000) { queue.request_update(); queue.request_update(); }
        else { queue.terminate(); }
    });
    queue.start_loop();
    require(updates == 10000 && tasks == 1, "Internal wake or yielding dropped external work");
    require(queue.id == INT64_C(4294967296), "Scheduler consumed external request IDs");
}

static void test_idle_wakeup() {
    server_queue queue;
    std::promise<void> idle;
    int updates = 0;
    queue.on_new_task([](server_task &&, bool) { return true; });
    queue.on_update_slots([&]() {
        if (++updates == 1) { idle.set_value(); }
        else { queue.terminate(); }
    });
    std::thread thread([&]() { queue.start_loop(); });
    idle.get_future().wait();
    queue.request_update();
    thread.join();
    require(updates == 2, "An idle loop lost its internal wakeup");
}

int main() {
    try {
        test_id_boundaries();
        test_response_and_cancel();
        test_scheduler_wakeups();
        test_idle_wakeup();
        std::puts("queue ID, response isolation, cancellation, yield and wakeup checks passed");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
