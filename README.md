# ParaFlow · 并流

让多张 GPU 接力，让多路推理并行。

ParaFlow is a multi-GPU inference engine built on llama.cpp. It adds pipeline request scheduling, native MTP scheduling, dynamic KV management, and recurrent-state memory optimizations for PCIe multi-GPU systems.

This repository is a **private pre-release source snapshot**. Synchronizing it does not publish a release or change repository visibility.

## Complete, unrestricted core

The pipeline scheduler, recurrent-state implementation, and all build dependencies are included as source. Commercial tiers, GPU UUID entitlement checks, activation codes, and the binary-only core delivery boundary have been removed. Directories historically named `paraflow-private` are ordinary source modules compiled by CMake; they do not require a separate proprietary kernel.

Model weights and their licenses remain separate. Credentials, clinical data, private prompts, raw benchmark outputs, and machine access configuration are excluded.

## Build

The validated baseline is source snapshot `89fcb8089e3f`. Current source also includes the later server-parameter documentation update. `PARAFLOW-SOURCE.json` maps every canonical source file to its repository path and SHA-256.

CPU-only build:

```sh
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build-cpu --target llama-server test-server-loop-guard -j4
ctest --test-dir build-cpu --output-on-failure -R '^test-server-loop-guard$'
```

V100 CUDA build (SM70):

```sh
cmake -S . -B build-sm70 \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 -DGGML_CUDA_GRAPHS=ON \
  -DGGML_NATIVE=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build-sm70 --target llama-server test-llama-context -j4
```

Use a CUDA toolkit and host compiler compatible with each other. RTX 3080 Ti uses architecture `86` instead of `70`. Choose build parallelism according to available RAM. Source validation included a clean CPU build and a full SM70 build; this synchronization does not claim a new CUDA rebuild or a new performance measurement.

## ParaFlow options

The [server documentation](tools/server/README.md) describes the differences from upstream, including:

- `--pipe-lanes`, `--pipe-fuse`, `--pipe-prefill-chunk`, `--pipe-prefill-lanes`, and `--pipe-pf-inflight`.
- `--kv-dynamic`, `--kv-dynamic-seq-max`, `--kv-spill`, and `--system-cache-mib`.
- `--vip` and request-level `priority`.
- `--loop-guard` / `--no-loop-guard`; loop protection defaults **off**.

Request-level MTP is controlled independently by `mtp` and `speculative.n_max`. The model must support the selected MTP path. Shared KV capacity, per-sequence context, concurrency, and memory overhead must be budgeted together.

Keep CUDA graphs enabled for the tested configurations. Use NVIDIA persistence mode and a resident NVML or `nvidia-smi dmon` sampler when benchmarking. Repeatedly starting `nvidia-smi` can materially perturb MTP timing on tested V100 environments.

## Validation boundaries

The unrestricted-core acceptance covered CPU checks, a full SM70 source build, 27 scoped full-model correctness/function cases across three trained models, and exact-output RAM paging/restoration checks. This is acceptance with documented qualifications, not a claim that every numerical test passes.

A tiny Q4 AR/MTP equality test retains a documented near-tie split also present in the accepted reference. A Huatuo C3/MTP5 sanity point had median 172.6 tokens/s versus historical 180.8 (-4.54%); this was within the declared 5% sanity bound and does not establish statistical equivalence. Earlier numerical, output-branch, and short-window limitations remain applicable. Benchmarks are model-, configuration-, and hardware-specific.

The current real eight-card V100 16GB scaling run is incomplete and includes thermal/memory failures. It is not advertised as a completed performance curve here.

Upstream workflow templates are retained under [contrib/upstream-github-workflows](contrib/upstream-github-workflows) rather than activated under `.github/workflows`. No release, scheduled publication, or package deployment workflow is enabled by this source synchronization.

## Upstream, license, and community

Based on [llama.cpp](https://github.com/ggml-org/llama.cpp). The upstream MIT license, authorship, and third-party notices are retained in [LICENSE](LICENSE), [AUTHORS](AUTHORS), and [licenses](licenses/). See [README.upstream.md](README.upstream.md) for upstream documentation.

Community discussion: [LINUX DO](https://linux.do/).
