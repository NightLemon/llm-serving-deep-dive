# 决策指南：何时使用 Prefill-Decode 分离架构

> 分离架构不是银弹。本节提供量化的决策框架，帮助你判断何时该用、何时不该用。

## 1. 核心决策公式

分离架构的净收益可以用以下公式估算：

$$
\text{Net Benefit} = \underbrace{G_{utilization} + G_{latency} + G_{scaling}}_{\text{分离收益}} - \underbrace{C_{transfer} + C_{ops} + C_{infra}}_{\text{分离成本}}
$$

其中：
- $G_{utilization}$：GPU 利用率提升带来的成本节省
- $G_{latency}$：TTFT/TPOT 改善带来的 SLO 达成率提升
- $G_{scaling}$：独立扩缩容带来的弹性收益
- $C_{transfer}$：KV Cache 传输的延迟和带宽开销
- $C_{ops}$：运维复杂度增加（监控、故障排查、部署流程）
- $C_{infra}$：额外的基础设施成本（高速网络、metadata service 等）

### 1.1 简化判断：Prompt/Output 比率

一个快速的经验规则——计算 **Prompt-Output Ratio (POR)**：

$$
\text{POR} = \frac{\text{avg\_prompt\_length}}{\text{avg\_output\_length}}
$$

| POR 范围 | 建议 | 典型场景 |
|---------|------|---------|
| POR > 10 | **强烈推荐**分离 | RAG、文档分析、长上下文摘要 |
| 3 < POR < 10 | **推荐**分离（需评估传输开销） | 代码生成、翻译 |
| 1 < POR < 3 | **可选**（取决于并发和 SLO 要求） | 通用对话 |
| POR < 1 | **不推荐**分离 | 创意写作、故事生成 |

### 1.2 量化估算：传输开销 vs 分离收益

**传输开销（TTFT 增加量）估算：**

$$
T_{transfer} = \frac{\text{KV\_Cache\_Size}}{\text{Network\_Bandwidth}} = \frac{2 \times L \times n_{kv} \times d_h \times S \times b_{dtype}}{BW_{net}}
$$

其中：
- $L$ = 模型层数
- $n_{kv}$ = KV head 数
- $d_h$ = head dimension
- $S$ = prompt 长度（tokens）
- $b_{dtype}$ = 每元素字节数（FP16=2, FP8=1）
- $BW_{net}$ = 有效网络带宽

**示例计算（Llama-3-70B, FP16, prompt=4096 tokens）：**

```
KV Cache Size = 2 × 80 × 8 × 128 × 4096 × 2 = 1.28 GB

同机 NVLink (H100):     1.28 GB / 450 GB/s = 2.8 ms   ← 几乎可忽略
跨机 IB NDR (400 Gbps): 1.28 GB / 46 GB/s  = 27.8 ms  ← 可接受
跨机 IB HDR (200 Gbps): 1.28 GB / 23 GB/s  = 55.7 ms  ← 需要评估
跨机 RoCE (100 Gbps):   1.28 GB / 11 GB/s  = 116.4 ms ← 可能过高
```

**分离收益（吞吐提升）估算：**

以混合部署的 GPU 利用率为 baseline，分离后的利用率提升取决于工作负载特征：

```python
def estimate_benefit(prompt_len, output_len, batch_size, gpu_flops, gpu_bw):
    """估算分离架构的相对收益"""
    
    # Prefill 阶段 GPU 利用率
    prefill_ai = prompt_len  # Arithmetic Intensity ∝ prompt_len
    prefill_util = min(1.0, prefill_ai * gpu_bw / gpu_flops)
    
    # Decode 阶段 GPU 利用率
    decode_ai = batch_size  # Arithmetic Intensity ∝ batch_size
    decode_util = min(1.0, decode_ai * gpu_bw / gpu_flops)
    
    # 混合部署的加权利用率
    prefill_time_ratio = prompt_len / (prompt_len + output_len)
    mixed_util = prefill_util * prefill_time_ratio + decode_util * (1 - prefill_time_ratio)
    
    # 分离部署的利用率（两边分别优化）
    disagg_prefill_util = prefill_util * 1.2  # 独立 batch 优化
    disagg_decode_util = decode_util * 1.3     # 不被 prefill 打断
    disagg_util = (disagg_prefill_util + disagg_decode_util) / 2
    
    improvement = (disagg_util - mixed_util) / mixed_util
    return improvement
```

## 2. 适合分离架构的场景

### 2.1 场景一：长 Prompt + 短生成（RAG、文档分析）

```
典型参数:
  prompt_length: 4000-32000 tokens
  output_length: 100-500 tokens
  POR: 8-320

为什么适合:
  ✅ Prefill 占用大量计算时间，分离后可独立优化 batch size
  ✅ KV Cache 大但只需传输一次
  ✅ Decode 阶段短，不会长期占用 decode 节点
  ✅ 传输开销相对于 prefill 时间占比小
```

**实际案例：企业知识库 RAG**

```
工作负载: 平均 prompt 8K tokens, 平均 output 200 tokens
部署规模: 32 × A100-80GB

混合部署 (32 GPU):
  - 吞吐: 50 req/s
  - P99 TTFT: 1200 ms
  - P99 TPOT: 65 ms
  - GPU 利用率: 38%

分离部署 (8P + 24D):
  - 吞吐: 85 req/s (+70%)
  - P99 TTFT: 680 ms (-43%)
  - P99 TPOT: 28 ms (-57%)
  - GPU 利用率: 62%
```

### 2.2 场景二：高并发、严格 SLO

```
典型参数:
  并发请求数: 500+
  TTFT SLO: < 500 ms
  TPOT SLO: < 50 ms

为什么适合:
  ✅ 高并发下混合部署的 TPOT 抖动严重
  ✅ 分离后 decode 节点 TPOT 稳定
  ✅ 可以独立扩 decode 节点来满足并发需求
  ✅ SLO 达成率显著提高
```

### 2.3 场景三：混合工作负载

```
典型参数:
  请求类型 A: 长 prompt (8K) + 短 output (100) → 60% 流量
  请求类型 B: 短 prompt (200) + 长 output (2000) → 40% 流量

为什么适合:
  ✅ 两类请求对资源的需求截然不同
  ✅ 分离后可以按需分配 prefill 和 decode 资源
  ✅ 混合部署下两类请求互相干扰
```

### 2.4 场景四：多模型共享 Decode 池

```
架构:
  Prefill Pool A: Model-7B   (轻量级请求)
  Prefill Pool B: Model-70B  (复杂推理请求)
  Decode Pool:    共享        (两个模型的 decode 请求)

为什么适合:
  ✅ Decode 资源可以在多个模型间共享
  ✅ 减少 decode GPU 的总需求
  ✅ 提高整体资源利用率
```

## 3. 不适合分离架构的场景

### 3.1 场景一：短 Prompt + 长生成

```
典型参数:
  prompt_length: 50-200 tokens
  output_length: 2000-8000 tokens
  POR: 0.01-0.1

为什么不适合:
  ✗ Prefill 开销极小（<10ms），不需要分离
  ✗ KV Cache 传输开销 > Prefill 计算时间
  ✗ Decode 阶段长，KV Cache 在 decode 端持续增长
  ✗ 分离带来的额外 TTFT 延迟不可接受

示例:
  prompt=100 tokens, Llama-3-70B:
    Prefill 时间: ~5 ms
    KV Transfer (IB NDR): ~0.7 ms
    传输开销占比: 14% → 不太值得
    
  但如果 prompt=100, output=4000:
    Decode 时间 ≈ 4000 × 30ms = 120s
    传输节省的 TPOT 改善: 可忽略（decode 太长了）
```

### 3.2 场景二：单机部署

```
为什么不适合:
  ✗ 同机 GPU 间传输虽快，但分离后每类节点的 GPU 数减少
  ✗ 单机 GPU 数量有限（通常 4-8），分离后每侧只有 2-4 GPU
  ✗ Chunked Prefill 在单机场景下通常是更好的选择
  ✗ 系统复杂度增加但资源池太小，弹性收益有限

例外:
  如果单机有 8 GPU 且工作负载 POR > 10，
  可以考虑 2P+6D 的配置。但需要仔细测试。
```

### 3.3 场景三：低并发

```
为什么不适合:
  ✗ 低并发下 GPU 利用率本来就低，分离不会有大的改善
  ✗ Decode 节点大部分时间空闲
  ✗ 分离的运维成本相对于收益过高
  ✗ 简单的 continuous batching 已经足够

阈值: 如果 QPS < 5，通常不需要分离。
```

### 3.4 场景四：模型较小

```
为什么不适合:
  ✗ 小模型的 prefill 速度很快，不是瓶颈
  ✗ KV Cache 小，传输开销也小，但分离的系统开销固定
  ✗ 小模型可以用更简单的方式（如多实例+负载均衡）解决扩展问题

阈值: 模型参数量 < 13B 时，通常不需要分离。
```

## 4. 决策流程图

```
                    开始评估
                      │
                      ▼
              ┌───────────────┐
              │ 模型 > 13B ?  │
              └──────┬────────┘
                     │
            No ──────┼────── Yes
            │        │        │
            ▼        │        ▼
        不推荐       │  ┌──────────────┐
        分离         │  │  POR > 3 ?   │
                     │  └──────┬───────┘
                     │         │
                     │  No ────┼──── Yes
                     │  │      │      │
                     │  ▼      │      ▼
                     │ ┌──────────┐  ┌──────────────┐
                     │ │QPS > 50? │  │  有高速互联?  │
                     │ └──┬───────┘  │ (IB/NVLink)  │
                     │    │          └──────┬───────┘
                     │ No─┤─Yes            │
                     │ │  │  │      No ────┼──── Yes
                     │ ▼  │  ▼      │      │      │
                     │不  │ 视SLO   ▼      │      ▼
                     │推  │ 要求  评估传输   │   推荐分离
                     │荐  │ 决定  开销是否   │
                     │    │      可接受     │
                     │    │                │
                     │    ▼                │
                     │  ┌──────────┐       │
                     │  │严格 SLO?│       │
                     │  └──┬──────┘       │
                     │  No─┤─Yes          │
                     │  │  │  │           │
                     │  ▼  │  ▼           │
                     │ 不  │ 推荐         │
                     │ 推  │ 分离         │
                     │ 荐  │              │
```

## 5. Benchmark 数据参考

### 5.1 不同工作负载下的分离 vs 混合对比

以下数据基于 Llama-3-70B，8×H100 SXM 集群，使用 NIXL over NVLink 传输：

| 工作负载 | prompt/output | 混合吞吐 | 分离吞吐 | 提升 | 分离值得？ |
|---------|--------------|---------|---------|------|-----------|
| RAG 短回答 | 8K/100 | 42 req/s | 78 req/s | +86% | **是** |
| 文档摘要 | 16K/500 | 18 req/s | 35 req/s | +94% | **是** |
| 代码生成 | 2K/1K | 30 req/s | 39 req/s | +30% | 视情况 |
| 通用对话 | 500/500 | 55 req/s | 58 req/s | +5% | **否** |
| 创意写作 | 100/4K | 12 req/s | 11 req/s | -8% | **否** |

### 5.2 不同网络条件下的 TTFT 影响

| 网络 | KV Transfer 延迟 (4K prompt, 70B) | 额外 TTFT | 可接受？ |
|------|----------------------------------|-----------|---------|
| NVLink H100 | 2.8 ms | +2.8 ms | 完全可接受 |
| IB NDR 400G | 27.8 ms | +27.8 ms | 可接受 |
| IB HDR 200G | 55.7 ms | +55.7 ms | 需评估 |
| RoCE 100G | 116.4 ms | +116.4 ms | 长 prompt 时勉强 |
| TCP 25G | 465.5 ms | +465.5 ms | 不推荐 |

### 5.3 不同 Prefill:Decode 比例的影响

以 RAG 场景（8K prompt, 200 output）为例，8×H100：

| P:D 比例 | Prefill 吞吐 | Decode 吞吐 | 总吞吐 | P99 TTFT | P99 TPOT |
|---------|-------------|-------------|--------|---------|---------|
| 1:7 | 瓶颈 | 过剩 | 45 req/s | 890 ms | 22 ms |
| 2:6 | 均衡 | 充足 | **72 req/s** | 520 ms | 28 ms |
| 3:5 | 充足 | 均衡 | **78 req/s** | 380 ms | 35 ms |
| 4:4 | 过剩 | 瓶颈 | 65 req/s | 280 ms | 52 ms |
| 5:3 | 过剩 | 严重瓶颈 | 42 req/s | 220 ms | 85 ms |

最优比例通常在 **2:6 到 3:5** 之间（取决于具体工作负载）。

## 6. 配比优化方法

### 6.1 基于利用率的配比公式

$$
\frac{N_P}{N_D} = \frac{T_{prefill}^{avg} \times \lambda}{T_{decode\_total}^{avg} \times \lambda} = \frac{T_{prefill}^{avg}}{L_{output}^{avg} \times T_{decode\_per\_token}^{avg}}
$$

**简化估算（假设充分 batching）：**

```python
def estimate_pd_ratio(
    avg_prompt_len: int,
    avg_output_len: int,
    prefill_speed: float,    # tokens/s per GPU (prefill throughput)
    decode_speed: float,     # tokens/s per GPU (decode throughput)
) -> float:
    """估算 Prefill:Decode GPU 比例"""
    
    # 单个请求的 prefill 时间（在 prefill GPU 上）
    t_prefill = avg_prompt_len / prefill_speed
    
    # 单个请求的 decode 时间（在 decode GPU 上）
    t_decode = avg_output_len / decode_speed
    
    # 比例 = prefill 资源需求 / decode 资源需求
    ratio = t_prefill / t_decode
    
    return ratio

# 示例: RAG 场景
ratio = estimate_pd_ratio(
    avg_prompt_len=8000,
    avg_output_len=200,
    prefill_speed=40000,   # H100 单卡 prefill ~40K tokens/s
    decode_speed=100,      # H100 单卡 decode ~100 tokens/s (per sequence)
)
# ratio ≈ 0.1, 即 1 Prefill GPU 对应 ~10 Decode GPU
# 实际中考虑 batching 效应，比例约为 1:4 到 1:6
```

### 6.2 动态调整策略

生产环境中，工作负载会随时间变化。推荐使用动态调整：

```python
class DynamicRatioAdjuster:
    """根据实时指标动态调整 P:D 比例"""
    
    def __init__(self, total_gpus, min_prefill=1, min_decode=1):
        self.total_gpus = total_gpus
        self.min_prefill = min_prefill
        self.min_decode = min_decode
    
    def adjust(self, metrics: dict) -> tuple[int, int]:
        prefill_queue = metrics["prefill_queue_depth"]
        decode_queue = metrics["decode_queue_depth"]
        prefill_util = metrics["prefill_gpu_util"]
        decode_util = metrics["decode_gpu_util"]
        
        # 如果 prefill 队列积压，增加 prefill GPU
        if prefill_queue > 10 and prefill_util > 0.8:
            return self._shift_to_prefill()
        
        # 如果 decode 队列积压，增加 decode GPU
        if decode_queue > 50 and decode_util > 0.8:
            return self._shift_to_decode()
        
        # 保持当前比例
        return self.current_n_prefill, self.current_n_decode
```

## 7. 成本效益分析

### 7.1 TCO (Total Cost of Ownership) 对比

以月度成本为例（AWS 上 8×H100 实例，约 $25/h per GPU）：

| 方案 | GPU 数 | 月成本 | 达到目标吞吐 | 成本效率 |
|------|-------|--------|------------|---------|
| 混合部署 | 16 GPU | $288,000 | 60 req/s | $4,800/req/s |
| 分离部署 | 12 GPU | $216,000 | 78 req/s | $2,769/req/s |
| **节省** | **-4 GPU** | **-$72,000** | **+30%** | **-42%** |

分离部署通过更高的利用率，用更少的 GPU 达到了更高的吞吐，每月节省 **$72,000**（25%）。

### 7.2 额外基础设施成本

| 项目 | 月成本 (估) | 说明 |
|------|-----------|------|
| 高速网络 (IB) | $0（通常已有）| GPU 集群标配 |
| Metadata Service | $200-500 | 轻量级服务 |
| Monitoring | $500-1000 | Prometheus + Grafana |
| 运维人力 | $2,000-5,000 | 额外的系统复杂度 |
| **总额外成本** | **$2,700-6,500** | |

即使算上额外成本，分离部署在大规模场景下仍然显著更划算。

## 8. 生产部署 Checklist

在决定使用分离架构之前，逐项检查：

```
前置条件:
□ 模型参数量 > 13B
□ 平均 POR > 3 或 QPS > 50
□ GPU 间有高速互联（NVLink 或 IB NDR/HDR）
□ 团队有分布式系统运维经验

技术准备:
□ 选定 KV Transfer 方案（NIXL / P2P NCCL / Mooncake）
□ 测试 KV Transfer 带宽和延迟
□ 确定初始 P:D 比例
□ 配置健康检查和故障恢复机制

性能验证:
□ 在目标工作负载下 benchmark 分离 vs 混合
□ 确认 TTFT 和 TPOT 满足 SLO
□ 测试峰值负载下的稳定性
□ 验证动态扩缩容是否正常工作

监控告警:
□ KV Transfer 延迟监控
□ Prefill/Decode 队列深度告警
□ GPU 利用率监控
□ 端到端延迟 SLO 告警
```

## 9. 小结

| 维度 | 适合分离 | 不适合分离 |
|------|---------|-----------|
| Prompt 长度 | 长 (>2K tokens) | 短 (<500 tokens) |
| Output 长度 | 短到中等 | 很长 (>4K tokens) |
| POR | > 3 | < 1 |
| 并发 | 高 (QPS > 50) | 低 (QPS < 5) |
| 模型规模 | 大 (>13B) | 小 (<13B) |
| 网络 | NVLink / IB | TCP only |
| SLO 要求 | 严格 | 宽松 |
| 运维能力 | 有分布式经验 | 简单部署优先 |

> **下一节**：[exercises.md](exercises.md) — 动手练习，亲自搭建和测试分离架构。
