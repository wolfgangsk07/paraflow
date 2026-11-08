# ParaFlow · 并流

[English](README.md) | **简体中文**

**基于 llama.cpp 的多 GPU 流水线推理引擎。**

ParaFlow 把多张 GPU 组织成一条持续满载的并发推理流水线：请求分成多条 lane，在不同 GPU 上错峰推进；原版 llama.cpp 的 PP 方案一次只跑一个批次，空出来的 stage 由下一条 lane 补上。

ParaFlow 正是为解决这一GPU浪费而设计的，普通 PCIe 2.0/3.0 服务器也能把多卡吃满，不需要 NVLink，也不需要把模型改成 TP，剩余GPU利用效率达98%以上（例如4卡C32下单流速度与原版 llama.cpp C8 相当）。

- **低速互联就能跑**：普通 PCIe 2.0/3.0 多卡机就能扩展，甚至基于局域网的多机互联也只损失7%不到的效率。
- **小卡也能加速**：2×V100 16G 在 C8 达到 225.7 t/s，高于单张 V100 32G 的 147.0 t/s。
- **高并发吞吐**：6×3080Ti C60 1,454 t/s；8×V100 C80 1,101 t/s，约为原版 llama.cpp 的 4.9 倍。
- **源码完整**：基于 llama.cpp，模型格式和量化方式不变，完全兼容现有生态。

## 多卡加速：让每个 stage 都有活干

Layer split 只是把层分到各张卡上，不会自动让这些 stage 保持忙碌。

上游 llama.cpp 一次只推进一个批次。这个批次在采样、等待 KV、或者两段 prompt 之间时，其他 stage 是空着的。ParaFlow 让多条 lane 同时在流水线里跑：

![lane 调度](docs/lane-schedule.zh.png)

卡数没有变。区别是：stage 1 处理 lane B 时，stage 0 已经在准备 lane C；某条 lane 的 logits 就绪后可以先采样。ParaFlow 把 pipeline lane、异步 decode ticket、动态 KV 和 MTP 组合起来，让 stage 之间的等待被下一条 lane 填上。

## 性能

### Decode

![ParaFlow 扩展曲线](docs/performance.zh.png)

*总 decode 吞吐对并发数。Qwen3.8-27B Q8，MTP3，输出 256 token；原版 llama.cpp 和 ParaFlow 在同并发下测量。*

- **6× RTX 3080 Ti 12 GB**：C60 = **1,454 t/s**。曲线到 C60 仍在上升；进一步加并发受上下文总量限制，没法继续测。
- **8× V100 16 GB**：C80 = **1,101 t/s**，原版 llama.cpp 225 t/s，**4.9×**。该点是多次 C80 运行的均值。
- **2× V100 16 GB 对 1× V100 32 GB**，C8：**225.7 vs 147.0 t/s**（+53.5%）。每 GPU 同工作量：225.7 vs 2×112.1 = 224.2 t/s，**≈100%**。
- C1 时两个引擎差不多。加速来自并发，不是把单条请求变快。

这些测试主要在小显存卡上完成。同样的模型和调度在大显存卡上依然成立；大卡能容纳更多的上下文，进一步榨干多核算力。


### 上下文

- 6× RTX 3080 Ti 总 KV 池：实测 **432K tokens**。
- 8× V100 总 KV 池：实测 **1088K tokens**（1088Ki / 34 slots，单序列上限 32K；首尾 slot 输出一致）。

## 局域网多机

RPC 路径可以把模型跨机展开。局域网内每个节点可以容纳 1 张或者多张卡，所有卡共同启用 ParaFlow 的多卡加速；Qwen3.8-27B Q8 MTP3 的结果：

| 并发 | 本机 | 局域网 | 局域网 / 本机 | RPC 流水化关闭 |
|---:|---:|---:|---:|---:|
| C8 | 255.2 | 252.8 | 99% | 156.8 |
| C16 | 378.4 | 362.4 | 96% | 208.8 |
| C24 | 476.0 | 443.0 | 93% | 248.4 |

### 快速开始

所有节点用同一份源码构建，开启 CUDA 和 RPC：

```sh
cmake -S . -B build -DGGML_CUDA=ON -DGGML_RPC=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build --target ggml-rpc-server llama-server -j8
```

每个 GPU 启动一个 RPC server（下面以节点 A 的两张卡为例；节点 B 同理，端口不要冲突）：

```sh
CUDA_VISIBLE_DEVICES=0 ./build/bin/ggml-rpc-server -H 0.0.0.0 -p 50052 -c
CUDA_VISIBLE_DEVICES=1 ./build/bin/ggml-rpc-server -H 0.0.0.0 -p 50053 -c
```

在协调机上把所有 RPC 端点按顺序传给 `llama-server`：

```sh
CUDA_VISIBLE_DEVICES='' ./build/bin/llama-server -m Qwen3.8-27B.Q8_0.gguf \
  -ngl 99 -sm layer -ts 1,1,1 \
  --rpc 10.0.0.11:50052,10.0.0.11:50053,10.0.0.12:50052 \
  -c 8192 -np 8 -fa on -ctk q8_0 -ctv q8_0 \
  --no-kv-unified --pipe-lanes 3 --pipe-fuse \
  --pipe-prefill-chunk 64 --pipe-prefill-lanes 1 \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-ngl 99 \
  --host 0.0.0.0 --port 8080
```

`-ts` 的个数要和 `--rpc` 暴露的设备数（加本机 GPU）一致；`--pipe-lanes` 用总设备数。上面的例子让协调机只做调度，所以清空了 `CUDA_VISIBLE_DEVICES`；如果协调机也出卡，把它加入可见设备，并在 `-ts` 和 `--rpc` 里补上对应项。RPC 路径目前用静态 KV，不要加 `--kv-dynamic`。RPC server 没有认证，只放在可信局域网，不要暴露公网。

## 快速开始

已在 V100（SM70）和 RTX 3080 Ti（SM86）上验证；要求 Linux + CUDA。RTX 3080 Ti（SM86；V100 用 `70`）构建：

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DGGML_CUDA_GRAPHS=ON -DGGML_NATIVE=OFF \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_UI=OFF
cmake --build build --target llama-server -j8
```

启动 4 卡 layer split 服务。路径、tensor split、上下文和 slot 数按实际硬件调整：

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

调用：

```sh
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8-27B.Q8_0.gguf",
    "messages": [{"role": "user", "content": "你好"}],
    "temperature": 0.2,
    "max_tokens": 256,
    "mtp": true,
    "speculative.n_max": 3
  }'
```

## 多卡加速怎么调

真正影响多卡加速的参数就这几个：

- `--pipe-lanes N`：直接设为 GPU 数，一卡一条 lane；先跑满，再根据 KV 和调度压力调整。
- `-np N`：服务 slot 数。提高并发时一起加；吞吐曲线是按 slot 数测的。
- `-c N`：开启动态 KV 后，这是**共享** token 预算，不是单 slot 上下文。
- `--kv-dynamic-seq-max N`：单 sequence 上限；开启动态 KV 时必填。
- `--no-kv-unified`：流水线模式必须。
- `--spec-draft-n-max N`：MTP 深度。更深可以提高每次验证的产出，但更占 KV 和验证行数。

动态 KV 还要求 CUDA KV offload 和 Flash Attention；流水线模式要求 `--no-kv-unified`。

先用默认 ubatch；如果显存不够，降低 `-np` 或加卡，`-ub` 等测量后再调。

## 其他参数

这些不改变多卡扩展模型，属于生产策略：

| 参数 | 作用 |
|---|---|
| `--vip` + `priority: 1` | VIP 请求运行时原地暂停普通 slot。 |
| `--kv-spill` | KV 池满时把冻结请求换到主机内存。要求 unified 或 dynamic KV。 |
| `--system-cache-mib N` | 用主机内存缓存共享 system 前缀状态。要求 non-unified KV。 |
| `--loop-guard` | 遇到长段完全重复输出时停止；默认关闭。 |

完整参数和约束见[参数文档](tools/server/README.md)。

## 边界

- 流水线和动态 KV 路径依赖 Linux + CUDA。
- 原生 MTP 要求模型带 NextN/MTP 头。
- 多机 RPC 仍是预览：目前只验证了混合局域网拓扑，暂不作为生产部署支持。
- 模型权重及其许可证独立。

## 友链

- [LINUX DO](https://linux.do/)

## 许可与社区

ParaFlow 基于 [llama.cpp](https://github.com/ggml-org/llama.cpp)。上游 MIT 许可证、作者信息和第三方声明保留在 [LICENSE](LICENSE)、[AUTHORS](AUTHORS) 和 [licenses](licenses/) 中；上游文档见 [README.upstream.md](README.upstream.md)。

引擎核心以源码形式完整包含在仓库中，由 CMake 直接构建，不需要单独的闭源内核。

联系方式：dongyiwan1988@qq.com  
社区交流：[LINUX DO](https://linux.do/)。
