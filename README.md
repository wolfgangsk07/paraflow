# ParaFlow · 并流

**让多张 GPU 接力，让多路推理并行。**

ParaFlow is an experimental multi-GPU inference engine built on llama.cpp. It explores pipelined execution, request scheduling, dynamic KV memory management, and native multi-token prediction (MTP).

This repository is a **private pre-release preparation snapshot**, based on the V72b component stack. It is not a public release or a claim of production readiness across models and hardware.

## What is being prepared

- Pipeline-parallel execution across multiple GPUs.
- Conventional and MTP request scheduling.
- An asynchronous draft executor and batched draft processing.
- Conditional avoidance of unnecessary pipeline drains during request admission.
- Dynamic KV management and draft-memory failure handling.

Features and experimental switches have hardware- and workload-specific constraints. A separate draft GPU is an optional tested configuration, not a universal hardware requirement or a guarantee of optimal allocation.

## Source and build status

The source was assembled from a frozen earlier engine snapshot and the tested V72b runtime dependency chain. `PARAFLOW-SOURCE.json` records the file hashes and component provenance. A clean CUDA build and release QA remain required before release; historical tests of incrementally linked binaries do not establish reproducibility of this assembled tree.

A local GCC CPU-only build compiled the source and libraries, but failed at final executable linking because the server requires `ggml_backend_cuda_host_buffer_type`. CPU-only support is therefore not established. The CUDA build has not yet been reproduced from this snapshot.

The directory and executable names inherited from llama.cpp remain unchanged for now. Branding does not imply compatibility changes.

A preliminary source-build configuration is:

```sh
cmake -S . -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_APP=OFF \
  -DGGML_SCHED_MAX_COPIES=8
cmake --build build --target llama-server -j
```

Architecture `86` is an example for Ampere; do not use it blindly for other GPUs. This command is a preparation recipe, not a verified release installation procedure.

## Evidence and limitations

V72b has shown a mixed-workload throughput improvement in controlled comparisons on a six-GPU configuration, including a dedicated draft GPU. This does not establish the same gain on five GPUs, three GPUs, or different models. No universal speedup multiplier is advertised here.

Release validation must cover automatic slot allocation, request churn, cache reuse, priorities, memory pressure, and output correctness. Some historical QA uses investigated near-tie token differences rather than exact token identity. Separate V100 experiments found unresolved cross-configuration output differences; those experiments are not release validation.

Model weights, credentials, clinical data, private prompts, raw outputs, machine access scripts, and production service configuration are not part of this repository.

## Upstream and license

Based on [llama.cpp](https://github.com/ggml-org/llama.cpp). The upstream MIT license and third-party license notices are retained in `LICENSE` and `licenses/`. See `README.upstream.md` for the upstream documentation snapshot. Exact upstream revision attribution is still being audited.

## 中文说明

“并流”强调逐级接力与多路并行。当前阶段先完成源码归档、构建重现与发布验证，再决定公开发布。现有测试的硬件、模型、并发、MTP深度、缓存和统计窗口必须一并披露，不将短窗口峰值当作长期平台，也不将增加硬件后的成绩称作同硬件提升。
