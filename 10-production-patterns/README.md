# Ch10: 生产环境实践

> 前置知识：Ch06 KV Cache 卸载、Ch08 调度与批处理、Ch09 分布式推理

## 🎯 学习目标

- 掌握 LLM 推理服务从"能跑"到"能上线"之间的关键工程实践
- 理解 cache-aware 路由和负载均衡的实现
- 能够配置 Prometheus + Grafana 监控推理服务
- 掌握性能 profiling 方法论（memory-bound vs compute-bound 诊断）
- 理解推理服务的成本优化策略

## 📑 内容大纲

### 1. Cache-Aware 路由与负载均衡（01-routing.md）

**为什么需要 cache-aware 路由？**
- 传统 round-robin 路由：每个请求随机分配到一个 replica
- 问题：相同前缀的请求分散到不同 replica → cache hit rate 低
- Cache-aware 路由：基于 prompt 前缀 hash 路由到同一 replica
- OpenAI 的做法：前 256 tokens hash + `prompt_cache_key` 路由

**路由策略：**
- Content-hash 路由：hash(prompt_prefix) → replica
- Consistent hashing：replica 增减时最小化 cache 失效
- 热度感知：将高频 prompt 的 replica 副本数增加

**实现参考：**
- Nginx / Envoy 中的自定义 hash 路由
- vLLM multi-instance 部署中的路由方案

### 2. 监控与可观测性（02-monitoring.md）

**vLLM 内置指标（Prometheus）：**
- `vllm:num_requests_running` — 正在处理的请求数
- `vllm:num_requests_waiting` — 等待队列长度
- `vllm:gpu_cache_usage_perc` — GPU KV Cache 使用率
- `vllm:cpu_cache_usage_perc` — CPU KV Cache 使用率
- `vllm:avg_prompt_throughput_toks_per_s` — prefill 吞吐量
- `vllm:avg_generation_throughput_toks_per_s` — decode 吞吐量
- `vllm:e2e_request_latency_seconds` — 端到端延迟
- `vllm:time_to_first_token_seconds` — TTFT
- `vllm:time_per_output_token_seconds` — TBT

**Grafana Dashboard 设计：**
- 吞吐量面板（input tokens/s, output tokens/s）
- 延迟面板（P50, P90, P99 TTFT 和 TBT）
- KV Cache 利用率面板
- 队列深度面板

**告警规则建议：**
- KV Cache 使用率 > 95%（可能触发 preemption）
- 等待队列 > N（可能需要扩容）
- P99 TTFT > SLA（可能需要增加 prefill 资源）

### 3. 性能 Profiling（03-profiling.md）

**诊断方法论：**
1. **识别瓶颈类型：**
   - Compute-bound：GPU SM 利用率高，显存带宽有余 → prefill 阶段
   - Memory-bound：显存带宽接近上限，GPU SM 空闲 → decode 阶段
   - Communication-bound：GPU 在等待 NCCL AllReduce → TP 通信

2. **工具：**
   - `nsys`（NVIDIA Nsight Systems）：全局时间线分析
   - `ncu`（NVIDIA Nsight Compute）：kernel 级分析
   - vLLM 内置 profiling：`--collect-detailed-traces`
   - PyTorch Profiler：自定义区间标记

3. **Roofline Model 应用：**
   - 计算 Arithmetic Intensity = FLOPs / Bytes
   - Prefill：AI 高 → compute-bound
   - Decode：AI 低 → memory-bound

**常见性能问题及对策：**
| 问题 | 表现 | 对策 |
|------|------|------|
| 长 prompt 阻塞 decode | TBT 突然飙高 | 启用 chunked prefill |
| KV Cache 不足 | 频繁 preemption | 增加 GPU 内存利用率或量化 KV |
| TP 通信瓶颈 | GPU 利用率低 | 检查 NVLink、减小 TP 度 |
| 小 batch | GPU 利用率低 | 增大 max_num_seqs |

### 4. 成本优化（04-cost-optimization.md）

**TCO 分析框架：**
- 硬件成本：GPU 租赁 / 采购
- 能耗成本：功率 × 时间
- 人力成本：运维管理

**优化策略：**
- **Prompt Caching**：减少重复计算，cache read 费用远低于重新 prefill
- **量化**：FP8 推理 → 同等显存服务更多请求
- **Speculative Decoding**：减少 decode 步数 → 提高吞吐量
- **Batch 优化**：提高 GPU 利用率 → 每 GPU 处理更多请求
- **按需扩缩容**：非高峰时段缩减 GPU 数量

**不同场景的硬件选择：**
| 场景 | 推荐硬件 | 理由 |
|------|---------|------|
| 低延迟、高并发 | H100/H200 | 高带宽 + 大显存 |
| 高吞吐、成本敏感 | A100 80GB | 性价比 |
| 超长上下文 | H200 (141GB) | 显存容量 |
| 小模型、高并发 | L40S | 成本最低 |

### 5. 故障恢复与高可用（05-high-availability.md）

**故障场景：**
- GPU 故障：单卡 hang，整机宕机
- 网络故障：TP 通信中断
- OOM：KV Cache 分配失败

**应对策略：**
- Health check + 自动重启
- 请求重试与幂等性
- Graceful degradation：降级到更小的 batch size
- Sleep mode：空闲时释放 GPU 资源，按需唤醒

**vLLM 相关功能：**
- Sleep mode：`vllm/features/sleep_mode.md`
- Pause/Resume：热暂停和恢复
- KV load failure recovery

### 6. 模型更新与灰度发布（06-model-updates.md）

**在不停服的情况下更新模型：**
- 蓝绿部署：新模型部署在新实例，切换路由
- 金丝雀发布：部分流量切到新模型
- A/B 测试：不同用户组使用不同模型

**KV Cache 兼容性问题：**
- 模型更新后，旧的 KV Cache 是否还有效？
- 通常不兼容 → 需要清除缓存或等待过期

## 📁 文件清单

- [x] `01-routing.md` — Cache-Aware 路由
- [x] `02-monitoring.md` — 监控与可观测性
- [x] `03-profiling.md` — 性能 Profiling
- [x] `04-cost-optimization.md` — 成本优化
- [x] `05-high-availability.md` — 故障恢复与高可用
- [x] `06-model-updates.md` — 模型更新与灰度发布
- [x] `exercises.md` — 动手练习（搭建监控 dashboard）
