#pragma once
#include <cstdint>
#include <string>

// Optional, process-local CUDA activity tracing. No inference/control decisions.
namespace gpu_activity {
void initialize();
void shutdown();
bool enabled();
void state(uint32_t active, uint32_t prefill, uint32_t mtp, uint32_t vip);
std::string status();
}
