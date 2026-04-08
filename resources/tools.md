# 工具链

## 推理框架

| 工具 | 链接 | 说明 |
|------|------|------|
| vLLM | https://github.com/vllm-project/vllm | 本仓库主要参考的推理引擎 |
| SGLang | https://github.com/sgl-project/sglang | RadixAttention, 高效调度 |
| TensorRT-LLM | https://github.com/NVIDIA/TensorRT-LLM | NVIDIA 优化推理引擎 |
| TGI | https://github.com/huggingface/text-generation-inference | HuggingFace 推理服务 |
| llama.cpp | https://github.com/ggerganov/llama.cpp | CPU/边缘设备推理 |

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
| Mooncake | - | 月之暗面 KV Transfer 方案 |
| NIXL | - | NVIDIA KV Transfer 库 |
