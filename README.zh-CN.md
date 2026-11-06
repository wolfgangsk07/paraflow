# ParaFlow · 并流

[English](README.md) | **简体中文**

让多张 GPU 接力，让多路推理并行。

ParaFlow 是基于 llama.cpp 开发的多 GPU 推理引擎，面向 PCIe 多卡平台，提供流水线请求调度、原生 MTP 调度、动态 KV 管理和循环状态显存优化。

当前仓库是**私有预发布源码快照**。同步源码不代表正式发布，也不会改变仓库可见性。

## 完整、无商业限制的核心

流水线调度器、循环状态实现及相关核心模块均提供完整源码。商业分档、GPU UUID 授权绑定、启动授权码以及仅提供核心二进制的交付限制均已移除。沿用历史名称的 `paraflow-private` 目录是由 CMake 直接编译的普通源码模块，不需要另行获取闭源内核。

模型权重及其许可证独立于推理引擎。仓库不包含凭据、临床数据、私有提示词、原始测试输出或机器接入配置。

## 构建

已验收的基线为源码快照 `89fcb8089e3f`。当前源码还包含后续补充的服务器参数文档。`PARAFLOW-SOURCE.json` 记录每个主源码文件对应的仓库路径和 SHA-256。

仅 CPU 构建：

```sh
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build-cpu --target llama-server test-server-loop-guard -j4
ctest --test-dir build-cpu --output-on-failure -R '^test-server-loop-guard$'
```

V100 CUDA 构建（SM70）：

```sh
cmake -S . -B build-sm70 \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 -DGGML_CUDA_GRAPHS=ON \
  -DGGML_NATIVE=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build-sm70 --target llama-server test-llama-context -j4
```

请使用彼此兼容的 CUDA 工具链与主机编译器。RTX 3080 Ti 对应的架构值是 `86`，应替换示例中的 `70`。构建并行度需要结合可用内存选择。源码验收已包含干净的 CPU 构建和完整的 SM70 构建；此次源码同步没有重新执行 CUDA 构建或性能测试。

## ParaFlow 参数

[服务器文档](tools/server/README.md) 说明了与上游不同的参数，包括：

- `--pipe-lanes`、`--pipe-fuse`、`--pipe-prefill-chunk`、`--pipe-prefill-lanes` 和 `--pipe-pf-inflight`。
- `--kv-dynamic`、`--kv-dynamic-seq-max`、`--kv-spill` 和 `--system-cache-mib`。
- `--vip` 与请求级 `priority`。
- `--loop-guard` / `--no-loop-guard`；循环输出保护默认**关闭**。

请求级 MTP 通过 `mtp` 和 `speculative.n_max` 独立控制。模型必须支持所选的 MTP 路径。共享 KV 总池、单会话上下文、并发数以及其他显存开销需要一并规划。

已测试的配置保持 CUDA 图开启。进行性能测试时，请开启 NVIDIA 持久模式，并使用常驻 NVML 或 `nvidia-smi dmon` 采样。在已测试的 V100 环境中，反复启动新的 `nvidia-smi` 进程会明显干扰 MTP 计时。

## 验收边界

移除商业限制后的核心验收包括 CPU 检查、完整 SM70 源码构建、三个完整模型上的27项限定范围正确性／功能用例，以及输出完全一致的 RAM 换出与恢复检查。这是保留已知边界的验收，不代表所有数值测试均已通过。

小型 Q4 模型的 AR／MTP 严格一致性测试仍保留一项已知分歧：候选 token 概率非常接近，已接受的参考版本也存在同样现象。华佗 C3／MTP5 单点检查的吞吐中位数为172.6 tokens/s，历史值为180.8，差异为 -4.54%；该结果处于事先设定的5%检查边界内，但不能据此认定统计等效。此前记录的数值、输出分叉和短窗口限制仍然适用。性能数据均依赖具体模型、配置和硬件。

当前真实八卡 V100 16GB 扩展性测试尚未完成，其中包含热降频和内存相关失败；这里不将它作为已经完成的性能曲线宣传。

上游工作流模板保留在 [contrib/upstream-github-workflows](contrib/upstream-github-workflows)，未放在会自动生效的 `.github/workflows` 中。此次源码同步不会启用正式发布、定时发布或软件包部署工作流。

## 上游、许可与社区

基于 [llama.cpp](https://github.com/ggml-org/llama.cpp)。上游 MIT 许可证、作者信息和第三方许可声明均保留在 [LICENSE](LICENSE)、[AUTHORS](AUTHORS) 和 [licenses](licenses/) 中。上游文档快照见 [README.upstream.md](README.upstream.md)。

社区交流：[LINUX DO](https://linux.do/)。
