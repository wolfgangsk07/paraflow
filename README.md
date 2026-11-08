# ParaFlow

**English** | [简体中文](README.zh-CN.md)

**Multi-GPU pipeline inference for llama.cpp.**

ParaFlow turns several GPUs into a decode pipeline that stays busy: requests are split into lanes and staggered across GPUs. Upstream llama.cpp PP runs one batch at a time; ParaFlow fills the idle stages with the next lane.

ParaFlow is built to eliminate that GPU waste. It keeps every GPU busy on ordinary PCIe 2.0/3.0 servers, with no NVLink and no tensor-parallel rewrite; the remaining GPU capacity is utilized at over 98% efficiency (for example, 4-card C32 single-stream throughput is on par with upstream llama.cpp at C8).

- **Runs on slow interconnects**: scale on ordinary PCIe 2.0/3.0 multi-GPU servers; even multi-machine over LAN loses less than 7% efficiency.
- **Small cards count**: 2×V100 16 GB at C8 reaches 225.7 t/s, above one V100 32 GB at 147.0 t/s.
- **High-concurrency throughput**: 6×3080 Ti C60 1,454 t/s; 8×V100 C80 1,101 t/s, about 4.9× upstream llama.cpp.
- **Source-complete**: built on llama.cpp; model format and quantization are unchanged, fully compatible with the existing ecosystem.


## Multi-GPU acceleration: keep every stage fed

Layer split gives you the stages. It does not automatically keep them busy.

With upstream llama.cpp, one batch moves through the GPUs at a time. While it samples, waits for KV, or moves between prompt chunks, the other stages have nothing to do. ParaFlow keeps multiple lanes in the pipeline:

![Lane scheduling](docs/lane-schedule.en.png)

The GPU count does not change. Stage 1 can process lane B while stage 0 prepares lane C, and a finished lane can be sampled. ParaFlow combines pipeline lanes, asynchronous decode tickets, dynamic KV, and MTP so the wait between stages is filled by the next lane.

## Performance

### Decode

![ParaFlow scaling](docs/performance.en.png)

*Total decode throughput vs concurrency. Qwen3.8-27B Q8, MTP3, 256 output tokens; official llama.cpp and ParaFlow measured at the same concurrency.*

- **6× RTX 3080 Ti 12 GB**: C60 = **1,454 t/s**. The curve is still rising at C60; further concurrency was blocked by the context budget, not by a throughput plateau.
- **8× V100 16 GB**: C80 = **1,101 t/s** vs upstream llama.cpp 225 t/s, **4.9×**. The point is the mean of repeated C80 runs.
- **2× V100 16 GB vs 1× V100 32 GB**, C8: **225.7 vs 147.0 t/s** (+53.5%). Per-GPU work match: 225.7 vs 2×112.1 = 224.2 t/s, **≈100%**.
- At C1 the two engines are close. The acceleration comes from concurrency, not from making a single request faster.

These tests were mainly run on small-memory cards. The same model and scheduler remain valid on large-memory cards, which can hold more context and get more out of the GPU cores.

### Context

- 6× RTX 3080 Ti total KV pool: **432K tokens** measured.
- 8× V100 total KV pool: **1088K tokens** measured (1088Ki / 34 slots, 32K per-sequence cap; first and last slot outputs matched).

## Multi-node over LAN

RPC can spread a model across machines. Each node on the LAN can hold one or more GPUs, and all GPUs take part in ParaFlow's multi-GPU acceleration. Qwen3.8-27B Q8 MTP3 results:

| Concurrency | Local | LAN | LAN / local | RPC pipeline off |
|---:|---:|---:|---:|---:|
| C8 | 255.2 | 252.8 | 99% | 156.8 |
| C16 | 378.4 | 362.4 | 96% | 208.8 |
| C24 | 476.0 | 443.0 | 93% | 248.4 |

### Quick start: 4 GPUs, 3 nodes

Concrete topology: node A has two GPUs, node B has one, and node C has one GPU and acts as the coordinator.

```text
node A (2 GPUs) --RPC--node B (1 GPU)  --RPC---> node C (1 GPU, runs llama-server)
                        |
clients ----------------+
```

The **coordinator** is the machine running `llama-server`: it accepts client requests, handles tokenization, sampling and pipeline scheduling, and can contribute its own GPU as one pipeline stage. Nodes A and B only run `ggml-rpc-server` and expose their GPUs to the coordinator.

Build every node from the same source with CUDA and RPC enabled:

```sh
cmake -S . -B build -DGGML_CUDA=ON -DGGML_RPC=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build --target ggml-rpc-server llama-server -j8
```

Node A (two GPUs, one RPC server process):

```sh
CUDA_VISIBLE_DEVICES=0,1 ./build/bin/ggml-rpc-server -H 0.0.0.0 -p 50052 -c
```

Node B (one GPU):

```sh
CUDA_VISIBLE_DEVICES=0 ./build/bin/ggml-rpc-server -H 0.0.0.0 -p 50052 -c
```

Node C (coordinator, one local GPU; start A and B first, then C):

```sh
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-server -m Qwen3.8-27B.Q8_0.gguf \
  -ngl 99 -sm layer -ts 1,1,1,1 \
  --rpc 10.0.0.11:50052,10.0.0.12:50052 \
  -c 8192 -np 8 -fa on -ctk q8_0 -ctv q8_0 \
  --no-kv-unified --pipe-lanes 4 --pipe-fuse \
  --pipe-prefill-chunk 64 --pipe-prefill-lanes 1 \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-ngl 99 \
  --host 0.0.0.0 --port 8080
```

`--rpc` lists only the two **RPC servers** (A and B); A's one server exposes two GPUs, so the total device count is four:

```text
A's 2 + B's 1 + C's local 1 = 4
-ts 1,1,1,1
--pipe-lanes 4
```

If coordinator C contributes no GPU and only schedules, clear its `CUDA_VISIBLE_DEVICES` and change `-ts` and `--pipe-lanes` to 3. The RPC path uses static KV; do not add `--kv-dynamic`. The RPC server has no authentication; keep it on a trusted LAN and never expose it publicly.


## Quick start

Tested on V100 (SM70) and RTX 3080 Ti (SM86); Linux and CUDA are required. Build for RTX 3080 Ti (SM86; use `70` for V100):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DGGML_CUDA_GRAPHS=ON -DGGML_NATIVE=OFF \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_UI=OFF
cmake --build build --target llama-server -j8
```

Launch a 4-GPU layer-split server. Adjust paths, tensor split, context, and slots for your hardware:

```sh
./build/bin/llama-server \
  -m /models/Qwen3.8-27B.Q8_0.gguf \
  -sm layer -ts 1,1,1,1 -ngl 99 \
  -c 131072 -np 16 \
  -fa on -ctk q8_0 -ctv q8_0 \
  --no-kv-unified \
  --pipe-lanes 4 \
  --kv-dynamic --kv-dynamic-seq-max 8192 \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --host 0.0.0.0 --port 8080
```

Call it:

```sh
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8-27B.Q8_0.gguf",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.2,
    "max_tokens": 256,
    "mtp": true,
    "speculative.n_max": 3
  }'
```

## The scaling recipe

For multi-GPU acceleration, these are the controls that matter:

- `--pipe-lanes N`: set it to the GPU count — one lane per GPU. Start there, then adjust if KV and scheduling pressure require it.
- `-np N`: number of server slots. Raise it with concurrency; the throughput curve is measured over slots.
- `-c N`: with dynamic KV, this is the **shared** token budget, not the per-slot context.
- `--kv-dynamic-seq-max N`: per-sequence cap; required with dynamic KV.
- `--no-kv-unified`: required for pipeline mode.
- `--spec-draft-n-max N`: MTP depth. More depth can raise accepted tokens per verification step, but costs KV and verify rows.

Dynamic KV also requires CUDA KV offload and Flash Attention; pipeline mode requires `--no-kv-unified`.

Start with the default ubatch. If a configuration runs out of memory, lower `-np` or add cards; tune `-ub` only after measuring.

## Other options

These do not change the multi-GPU scaling model; they are policy knobs for production serving.

| Option | What it does |
|---|---|
| `--vip` + `priority: 1` | Pause ordinary slots in place while a VIP request runs. |
| `--kv-spill` | Move a frozen request to host RAM when the KV pool is full. Requires unified or dynamic KV. |
| `--system-cache-mib N` | Cache shared system-prefix states in host RAM. Requires non-unified KV. |
| `--loop-guard` | Stop long exact-repetition loops; off by default. |

Full list and constraints: [parameter documentation](tools/server/README.md).

## Limits

- Linux + CUDA for the pipeline and dynamic-KV paths.
- Native MTP requires a model with NextN/MTP heads.
- Multi-node RPC is a preview: the tested hybrid LAN topology is experimental, not a supported production deployment.
- Model weights and their licenses are separate.

## Friend link

- [LINUX DO](https://linux.do/)

## License and community

ParaFlow is based on [llama.cpp](https://github.com/ggml-org/llama.cpp). The upstream MIT license, authorship, and third-party notices are retained in [LICENSE](LICENSE), [AUTHORS](AUTHORS), and [licenses](licenses/). Upstream documentation is preserved in [README.upstream.md](README.upstream.md).

The engine core is fully included as source and built by CMake; it does not require a separate proprietary kernel.

Contact: dongyiwan1988@qq.com  
Community: [LINUX DO](https://linux.do/).
