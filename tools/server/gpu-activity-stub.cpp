#include "gpu-activity.h"
namespace gpu_activity {
void initialize() {}
void shutdown() {}
bool enabled() { return false; }
void state(uint32_t,uint32_t,uint32_t,uint32_t) {}
std::string status() { return "{\"configured\":false,\"enabled\":false,\"supported\":false}"; }
}
