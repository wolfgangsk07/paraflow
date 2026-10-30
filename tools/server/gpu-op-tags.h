#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
namespace gpu_activity {
constexpr std::size_t operation_count = 10;
// Fixed categories only: no kernel symbol, model data or request content is saved.
inline uint32_t operation_group(const char * name) {
    if (!name) return 0;
    auto has = [name](const char * value) { return std::strstr(name, value) != nullptr; };
    if (has("gated_delta") || has("gdn_") || has("ssm_") || has("recurrent") || has("solve_tri")) return 5;
    if (has("flash_attn") || has("fattn")) return 4;
    if (has("mul_mat_vec") || has("mmvq") || has("mmvf")) return 1;
    if (has("mul_mat_q") || has("mmq") || has("mul_mat_id")) return 2;
    if (has("gemm") || has("gemv") || has("cutlass")) return 3;
    if (has("quantize") || has("dequantize")) return 9;
    if (has("norm") || has("soft_max") || has("softmax")) return 6;
    if (has("copy") || has("cpy") || has("get_rows") || has("set_rows") || has("gather") || has("scatter") || has("cont_")) return 7;
    if (has("add_") || has("mul_") || has("scale") || has("rope") || has("silu") || has("gelu") || has("glu_") || has("elementwise")) return 8;
    return 0;
}
}
