# Continuous Batching 深入

> 从 static batching 到 iteration-level scheduling——理解现代 LLM 推理服务的吞吐量跃迁

## 1. 背景：为什么 batching 对推理至关重要？

LLM 推理的核心瓶颈在于 **memory-bandwidth bound**（decode 阶段）和 **compute bound**（prefill 阶段）。单个请求的 decode 阶段几乎无法充分利用 GPU 的算力——每步只对 1 个 token 做矩阵向量乘法，计算量远低于 GPU 的峰值 FLOPS。

Batching 的核心收益来自：将多个请求的矩阵向量乘法合并为矩阵矩阵乘法，从而提高算术强度（Arithmetic Intensity），将 memory-bandwidth bound 操作推向 compute bound。

```
单请求 decode: Q(1, d) × K^T(d, s) → 算术强度 ≈ 2s / (d + s) ≈ 2 (当 s << d)
Batch=B decode: Q(B, d) × K^T(d, s) → 算术强度 ≈ 2Bs / (Bd + s) → 随 B 增长
```

然而，如何高效地组织 batch 并非显而易见。传统方法（static batching）存在严重的资源浪费问题。

## 2. Static Batching 的问题

### 2.1 工作方式

Static Batching 的工作方式类似于传统的 batch processing：

1. 收集一组请求（batch_size = B）
2. 对整个 batch 执行所有 decode steps，直到 **所有** 请求都完成
3. 一次性返回所有结果
4. 开始处理下一个 batch

```
时间 →
Req1: [prefill][decode][decode][decode][decode][done][pad][pad][pad][pad]
Req2: [prefill][decode][decode][decode][decode][decode][decode][done][pad][pad]
Req3: [prefill][decode][decode][decode][decode][decode][decode][decode][decode][done]
      ↑ batch 开始                                                          ↑ batch 结束
```

### 2.2 Padding 浪费的精确量化

假设一个 batch 中有 B 个请求，第 i 个请求的输出长度为 $L_i$。设 $L_{max} = \max(L_i)$。

**实际计算的 token 数：**
$$T_{actual} = \sum_{i=1}^{B} L_i$$

**由于 padding 浪费的 token 数：**
$$T_{wasted} = \sum_{i=1}^{B} (L_{max} - L_i) = B \cdot L_{max} - \sum_{i=1}^{B} L_i$$

**GPU 利用率（单看 decode 阶段）：**
$$\eta_{static} = \frac{\sum_{i=1}^{B} L_i}{B \cdot L_{max}} = \frac{\bar{L}}{L_{max}}$$

当输出长度分布不均匀时，浪费极为严重。例如：

| 场景 | 输出长度分布 | GPU 利用率 |
|------|-------------|-----------|
| 代码补全 | L ~ Uniform(10, 200) | ~52% |
| 聊天对话 | L ~ Exponential(μ=100) | ~25-35% |
| 文档摘要 | L ~ Normal(500, 100) | ~70% |
| 混合工作负载 | L ~ Bimodal(50, 500) | ~15-25% |

**关键洞察**：聊天场景下指数分布的长尾特性意味着少数长回复会"拖住"整个 batch，导致大量 GPU 算力浪费在 padding 上。

### 2.3 排队延迟问题

除了 padding 浪费，static batching 还有 **排队延迟** 问题：

- 新请求到达时，如果当前 batch 还没执行完，必须等待
- 等待时间 = 当前 batch 剩余时间 = $(L_{max} - L_{current\_step}) \times t_{step}$
- 在高负载下，排队延迟可能达到数秒甚至数十秒

## 3. Iteration-Level Scheduling：Orca 的核心贡献

### 3.1 论文核心思想

[Orca (OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/yu) 提出了 **iteration-level scheduling**，其核心思想极为简洁：

> **在每个 decode iteration（而非整个请求的生命周期）进行调度决策。**

这意味着：
- 每个 decode step 结束后，检查是否有请求已完成（生成了 EOS token 或达到 max_length）
- 已完成的请求立即移出 batch，释放其占用的 GPU 显存
- 新到达的请求可以立即加入当前 batch
- batch 的组成在每个 iteration 都可能不同

```
时间 →
Req1: [prefill][decode][decode][decode][done]
Req2: [prefill][decode][decode][decode][decode][decode][done]
Req3:          [prefill][decode][decode][decode][decode][decode][decode][done]
Req4:                   [prefill][decode][decode][decode][done]
Req5:                                   [prefill][decode][decode][decode][done]
      ↑ 在任意 iteration 边界，请求可以进出 batch
```

### 3.2 Continuous Batching 的吞吐量优势

**Static Batching 的吞吐量：**
$$Throughput_{static} = \frac{B}{L_{max} \cdot t_{step} + t_{collect}}$$

其中 $t_{collect}$ 是收集满一个 batch 的等待时间。

**Continuous Batching 的吞吐量：**

在稳态下（系统满载），每个 iteration 都有 B 个请求在并行执行。每个 iteration 可能完成若干请求并立即补入新请求。

$$Throughput_{continuous} = \frac{B}{\bar{L} \cdot t_{step}}$$

**吞吐量提升比：**
$$\frac{Throughput_{continuous}}{Throughput_{static}} \approx \frac{L_{max}}{\bar{L}}$$

当输出长度方差较大时（例如 $L_{max} / \bar{L} = 3$），continuous batching 可以带来约 3 倍的吞吐量提升。

**实际数据参考（A100-80GB, Llama-2-13B）：**

| 调度方式 | 吞吐量 (tokens/s) | 提升比 |
|---------|-------------------|--------|
| Static Batching (B=8) | ~1,200 | 1x |
| Continuous Batching (B=8) | ~2,800 | 2.3x |
| Continuous Batching (B=32) | ~4,500 | 3.75x |

注意：continuous batching 还允许使用更大的 batch size，因为显存可以更高效地利用（已完成请求释放的显存可以立即给新请求使用）。

### 3.3 实现的关键挑战

Orca 论文指出实现 iteration-level scheduling 的几个关键挑战：

**1. Selective Batching**

不同请求可能处于不同阶段（prefill vs decode），需要将它们分开处理，或者特殊处理混合 batch。Orca 采用的方式是每次 iteration 中对 prefill 和 decode 请求分别执行 attention：

```python
# 伪代码：Orca 的 selective batching
def iteration(batch):
    # 非 attention 层可以直接 batch
    hidden = self.embed(batch.tokens)
    for layer in self.layers:
        hidden = layer.mlp(layer.norm(hidden))  # 直接 batch
        # attention 需要分别处理
        hidden = selective_attention(
            hidden, 
            batch.prefill_mask, 
            batch.decode_mask,
            batch.kv_caches
        )
    return self.lm_head(hidden)
```

**2. 动态显存管理**

每个 iteration 的 batch 组成可能变化，KV Cache 的分配和释放需要在 iteration 粒度高效管理。这正是 PagedAttention（Ch04）解决的问题。

**3. 请求状态跟踪**

调度器需要维护每个请求的当前状态：已处理的 token 数、剩余的 prompt tokens、是否已完成等。

## 4. Micro-Batching 策略

### 4.1 Prefill 与 Decode 的混合执行

在一个 iteration 中，batch 可能同时包含：
- 新到达的请求需要做 prefill（处理完整 prompt）
- 已经在 running 的请求需要做 decode（生成下一个 token）

这两类请求的计算特性截然不同：

| 特性 | Prefill | Decode |
|------|---------|--------|
| 计算量 | O(n^2) attention + O(n) FFN | O(n) attention + O(1) FFN |
| 主要瓶颈 | Compute bound | Memory-bandwidth bound |
| 每请求 token 数 | 数百~数千 | 1 |
| 耗时 | 数十~数百 ms | ~10 ms |

**方案 1：Prefill 优先**

```
Iteration 1: [Req1 prefill] [Req2 prefill]     ← 只做 prefill
Iteration 2: [Req1 decode] [Req2 decode]        ← 只做 decode
Iteration 3: [Req1 decode] [Req2 decode] [Req3 prefill]  ← 混合
```

优点：prefill 的 compute-bound 特性可以充分利用 GPU 算力。
缺点：在只做 prefill 的 iteration 中，正在 decode 的请求被阻塞，TBT（Time Between Tokens）上升。

**方案 2：混合执行（vLLM 默认）**

在同一个 iteration 中同时处理 prefill 和 decode 请求。GPU kernel 需要处理不同长度的序列。

优点：decode 请求不被阻塞，TBT 稳定。
缺点：prefill 的大量 token 和 decode 的单 token 混合时，GPU 利用效率可能不如纯 prefill 高效。

### 4.2 Piggybacking 技术

Piggybacking 是一种优化技巧：将 prefill 请求的**最后一个 token**作为 decode token 处理。

**核心思想：**

当一个请求完成 prefill 后，它需要生成第一个 output token。传统做法是在 prefill iteration 结束后，将这个请求加入 decode 列表，在下一个 iteration 开始 decode。

Piggybacking 将 prefill 的最后一个 token 的处理方式改为 decode 模式，这样 prefill 完成时就已经生成了第一个 output token，省去一个 iteration 的延迟。

```
Without piggybacking:
  Iteration 1: prefill [t1, t2, ..., tn]  → 生成 KV Cache
  Iteration 2: decode [tn]                → 生成第一个 output token

With piggybacking:
  Iteration 1: prefill [t1, t2, ..., tn-1] + decode [tn] → 同时生成 KV Cache + 第一个 output token
```

**收益：**
- TTFT（Time To First Token）减少一个 decode step 的时间
- 在 TTFT 敏感的场景（交互式聊天）中非常有价值
- vLLM 和大多数现代 serving 框架默认启用此优化

## 5. 与其他 Batching 策略的对比

### 5.1 Dynamic Batching（Triton Inference Server）

Dynamic Batching 介于 static 和 continuous 之间：

- 在短时间窗口内收集请求
- 组成一个 batch 一次性执行
- 但 batch 内仍然是 static 执行方式

```
时间窗口      batch 执行
[t0, t0+Δt] → [batch1 执行至所有完成]
[t1, t1+Δt] → [batch2 执行至所有完成]
```

### 5.2 Cellular Batching（S3）

[S3 (2023)](https://arxiv.org/abs/2306.06000) 提出 **Split-Fuse** 策略：

- **Split**：将长 prefill 请求拆分成多个 chunk
- **Fuse**：将 prefill chunk 和 decode 请求融合到同一个 batch

这实际上是 chunked prefill 的前身思想。核心目标是让每个 iteration 的计算量尽可能均匀，避免一个超长 prefill 垄断整个 iteration。

### 5.3 策略对比总结

| 策略 | 调度粒度 | Padding 浪费 | 新请求延迟 | 实现复杂度 |
|------|---------|-------------|-----------|-----------|
| Static | Request | 高 | 高 | 低 |
| Dynamic | 时间窗口 | 中 | 中 | 低 |
| Continuous | Iteration | 无 | 低 | 中 |
| Continuous + Chunked Prefill | Iteration + Chunk | 无 | 极低 | 高 |

## 6. 实际吞吐量对比数据

以下数据基于公开 benchmark 和论文结果（A100-80GB, Llama-2-13B, ShareGPT 数据集）：

### 6.1 不同并发下的吞吐量

| 并发请求数 | Static (req/s) | Continuous (req/s) | 提升倍数 |
|-----------|---------------|-------------------|---------|
| 4 | 2.1 | 3.8 | 1.81x |
| 8 | 3.5 | 7.2 | 2.06x |
| 16 | 4.2 | 12.5 | 2.98x |
| 32 | 4.8 | 18.3 | 3.81x |
| 64 | 5.0 | 22.1 | 4.42x |

随着并发增加，continuous batching 的优势愈发明显——因为它可以高效利用已完成请求释放的 GPU 显存来容纳更多并发请求。

### 6.2 不同输出长度分布下的吞吐量对比

| 输出长度分布 | Static 利用率 | Continuous 利用率 | 提升 |
|------------|-------------|-----------------|------|
| 固定 128 tokens | 100% | 100% | 1.0x |
| Uniform(64, 192) | 67% | ~100% | 1.5x |
| Exponential(μ=128) | 37% | ~100% | 2.7x |
| ShareGPT 真实分布 | 28% | ~100% | 3.6x |

当所有请求输出长度相同时，continuous batching 没有额外优势。真实场景下输出长度方差越大，continuous batching 的收益越大。

## 7. 实现要点总结

实现一个高效的 continuous batching 系统需要以下组件的协同：

```
┌─────────────────────────────────────────────────┐
│                   Scheduler                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Waiting   │  │ Running  │  │ Finished │       │
│  │ Queue     │→ │ Queue    │→ │ Queue    │       │
│  └──────────┘  └──────────┘  └──────────┘       │
│       ↑              ↕                           │
│       │     ┌─────────────────┐                  │
│       │     │ KV Cache Manager│                  │
│       │     │ (Block Alloc)   │                  │
│       │     └─────────────────┘                  │
│       │              ↓                           │
│  ┌─────────────────────────────────────┐        │
│  │     SchedulerOutput                  │        │
│  │  - scheduled_requests               │        │
│  │  - num_batched_tokens               │        │
│  │  - blocks_to_swap / blocks_to_copy  │        │
│  └─────────────────────────────────────┘        │
└─────────────────────────────────────────────────┘
                       ↓
              ┌─────────────────┐
              │   Model Runner   │
              │  (GPU Execution) │
              └─────────────────┘
```

**关键设计决策：**

1. **何时执行调度**：每个 iteration 开始前（同步）或上一个 iteration 执行中（异步，见 Ch08-05 DBO）
2. **如何处理显存不足**：preemption（抢占低优先级请求，见 Ch04）
3. **Prefill 和 Decode 的混合策略**：是否允许混合执行，chunk 大小如何选择（见 Ch08-03）
4. **Budget 控制**：每个 iteration 最多处理多少 tokens、多少请求

## 8. 参考资料

- [Orca: A Distributed Serving System for Transformer-Based Generative Models (OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/yu)
- [S3: Increasing GPU Utilization during Generative Inference for Higher Throughput (2023)](https://arxiv.org/abs/2306.06000)
- [Efficient Memory Management for LLM Serving with PagedAttention (SOSP 2023)](https://arxiv.org/abs/2309.06180)
- [vLLM 官方文档](https://docs.vllm.ai/en/latest/)
- [Anyscale: How Continuous Batching Enables 23x Throughput in LLM Inference While Reducing p50 Latency](https://www.anyscale.com/blog/continuous-batching-llm-inference)
