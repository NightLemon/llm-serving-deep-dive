# 工具链

## 推理框架

| 工具 | 链接 | 说明 |
|------|------|------|
| vLLM | https://github.com/vllm-project/vllm | 本仓库主要参考的推理引擎 |
| SGLang | https://github.com/sgl-project/sglang | RadixAttention, 高效调度 |
| TensorRT-LLM | https://github.com/NVIDIA/TensorRT-LLM | NVIDIA 优化推理引擎 |
| TGI | https://github.com/huggingface/text-generation-inference | HuggingFace 推理服务 |
| llama.cpp | https://github.com/ggerganov/llama.cpp | CPU/边缘设备推理 |

> 版本变化很快。跑实验前先看 [版本基线与 Freshness Gate](version-baseline.md)，并用 `vllm serve --help` / 官方 latest docs 复核 CLI 参数。

## 本仓库脚本

| 工具 | 用途 |
|------|------|
| [`scripts/kv_cache_calculator.py`](../scripts/kv_cache_calculator.py) | 零依赖 KV Cache 容量计算器，用于 Ch01/Ch13 的手算验收、TP 分片估算和 VLM 流量 mix 粗算 |
| [`scripts/kv_transfer_calculator.py`](../scripts/kv_transfer_calculator.py) | Ch05 分离架构 KV 传输大小和网络耗时估算，支持 full KV 与单 shard 两种口径 |
| [`scripts/speculative_decoding_simulator.py`](../scripts/speculative_decoding_simulator.py) | Ch07 投机解码 rejection sampling 模拟，验证 $E=(1-\alpha^{\gamma+1})/(1-\alpha)$ |
| [`scripts/batching_throughput_estimator.py`](../scripts/batching_throughput_estimator.py) | Ch08 用输出长度分布估算 static batching token 浪费和 continuous batching 理想收益 |
| [`scripts/tp_comm_estimator.py`](../scripts/tp_comm_estimator.py) | Ch09 decode 阶段 TP 通信、KV 读取和权重读取的粗略拆解 |
| [`scripts/freshness_check.py`](../scripts/freshness_check.py) | 可选联网检查 PyPI 当前版本，并与版本基线对照；不放进默认 CI，避免网络波动影响部署 |
| [`scripts/smoke_tests.py`](../scripts/smoke_tests.py) | 一键跑所有脚本的代表性样例，CI 也会执行 |

示例：

```bash
python scripts/smoke_tests.py
```

## 性能分析工具

| 工具 | 用途 |
|------|------|
| NVIDIA Nsight Systems (`nsys`) | GPU 时间线分析 |
| NVIDIA Nsight Compute (`ncu`) | Kernel 级性能分析 |
| PyTorch Profiler | Python 层性能分析 |
| vLLM `--collect-detailed-traces` | vLLM 内置 profiling |
| `vllm bench serve` | vLLM 官方 benchmark 工具 |

## 监控

| 工具 | 用途 |
|------|------|
| Prometheus | 指标采集 |
| Grafana | 可视化 Dashboard |
| vLLM Prometheus exporter | vLLM 内置指标导出 |

## Attention Kernel

| 库 | 链接 | 说明 |
|------|------|------|
| FlashAttention | https://github.com/Dao-AILab/flash-attention | IO-aware exact attention |
| FlashInfer | https://github.com/flashinfer-ai/flashinfer | 灵活的 attention kernel 库 |
| FlashMLA | https://github.com/deepseek-ai/FlashMLA | MLA 专用 kernel |

## KV Cache 管理

| 工具 | 链接 | 说明 |
|------|------|------|
| LMCache | https://github.com/LMCache/LMCache | 独立 KV Cache 管理库 |
| Mooncake | https://github.com/kvcache-ai/Mooncake | 月之暗面 KV Transfer 方案 |
| NIXL | https://github.com/ai-dynamo/nixl | NVIDIA KV Transfer 库 |
