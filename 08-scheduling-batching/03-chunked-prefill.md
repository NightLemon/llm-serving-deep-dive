# Chunked Prefill

> 将长 prompt 的 prefill 切成小块，与 decode 请求交错执行——解决 "长 prefill 阻塞 decode" 的核心优化

## 1. 问题：长 Prompt Prefill 阻塞 Decode

### 1.1 Prefill 与 Decode 的耗时不对称

在 LLM 推理中，prefill 和 decode 阶段的耗时差异巨大：

```
Prefill 1024 tokens (Llama-2-70B, A100):  ~120 ms
Prefill 4096 tokens (Llama-2-70B, A100):  ~450 ms
Prefill 8192 tokens (Llama-2-70B, A100):  ~900 ms

Decode 1 token (Llama-2-70B, A100):       ~15 ms
```

当一个 8K token 的长 prompt 到达时，如果调度器决定在这个 iteration 执行它的 prefill，那么：

- 这个 iteration 将耗时 ~900ms
- 在这 900ms 内，所有正在 decode 的请求都被阻塞
- 这些 decode 请求的 TBT（Time Between Tokens）从正常的 ~15ms 跃升到 ~900ms
- 用户感知到明显的输出"卡顿"

### 1.2 量化影响

假设系统中有 30 个正在 decode 的请求，一个新的 4096 token prompt 到达：

```
Without chunked prefill:
  正常 TBT: ~20ms (30 个 decode 请求的 iteration)
  Prefill iteration: ~450ms (4096 tokens prefill + 30 decode)
  TBT 峰值: 450ms → 22.5x 正常值

  用户体验: 所有 30 个在线用户感受到约 0.4 秒的卡顿
```

这在交互式应用（聊天机器人、代码助手）中是不可接受的。

### 1.3 问题的根本原因

Prefill 是 **compute-bound** 操作，其计算量与 prompt 长度的平方（Self-Attention）和线性（FFN）成正比：

$$T_{prefill}(n) \approx \alpha \cdot n^2 + \beta \cdot n$$

其中 $\alpha$ 和 $\beta$ 取决于模型大小和硬件。

当 $n$ 很大时，一个 iteration 的执行时间被 prefill 支配，decode 请求虽然计算量极小（每个只有 1 token），但必须等待 prefill 完成才能得到结果。

## 2. 解决方案：Chunked Prefill

### 2.1 核心思想

将一个长 prompt 的 prefill 分成多个固定大小的 **chunk**，每个 iteration 只处理一个 chunk。在处理 prefill chunk 的同时，decode 请求照常执行。

```
Without chunked prefill (prompt = 4096 tokens):
Iteration 1: [────── prefill 4096 tokens ──────] [decode x30]   ← 耗时 ~450ms
Iteration 2:                                     [decode x30]   ← 耗时 ~20ms
Iteration 3:                                     [decode x30]   ← 耗时 ~20ms

With chunked prefill (chunk_size = 512):
Iteration 1: [prefill 512] [decode x30]  ← 耗时 ~70ms
Iteration 2: [prefill 512] [decode x30]  ← 耗时 ~70ms
Iteration 3: [prefill 512] [decode x30]  ← 耗时 ~70ms
...
Iteration 8: [prefill 512] [decode x30]  ← 耗时 ~70ms  (8 个 iteration 完成 prefill)
```

**效果：**
- TBT 从 450ms 降低到 ~70ms（仍高于纯 decode 的 20ms，但大幅改善）
- 代价：TTFT 从 450ms 增加到 8 * 70ms = 560ms（prefill 分摊到多个 iteration）

### 2.2 工作流程

```
新请求到达（prompt = 2048 tokens, chunk_size = 512）

Iteration 1:
  调度器分配 chunk: tokens[0:512]
  与 running 的 decode 请求一起执行
  请求状态: partial_prefill, processed_tokens = 512

Iteration 2:
  调度器分配 chunk: tokens[512:1024]
  请求状态: partial_prefill, processed_tokens = 1024

Iteration 3:
  调度器分配 chunk: tokens[1024:1536]
  请求状态: partial_prefill, processed_tokens = 1536

Iteration 4:
  调度器分配 chunk: tokens[1536:2048]
  请求状态: prefill_complete → 转为 decode
  生成第一个 output token（piggybacking）

Iteration 5+:
  正常 decode（每步 1 token）
```

## 3. vLLM 配置与实现

### 3.1 配置参数

```bash
# vLLM v1 默认启用 chunked prefill
# 以下参数控制其行为:

vllm serve model_name \
    --enable-chunked-prefill          # v1 中默认 True
    --max-num-batched-tokens 2048     # 每个 iteration 最大 token 总数
    --max-num-seqs 256                # 每个 iteration 最大请求数
```

**`--max-num-batched-tokens` 的双重作用：**

在启用 chunked prefill 时，`max_num_batched_tokens` 同时控制：
1. 每个 iteration 的总 token 预算（包括 prefill chunks + decode tokens）
2. 间接控制了 prefill chunk 的最大大小

```python
# 调度器中的逻辑（简化）
def _get_num_prefill_tokens(self, req, budget):
    remaining_prompt = req.num_prompt_tokens - req.num_processed_tokens
    # chunk 不能超过剩余预算
    max_chunk = budget.remaining_token_budget()
    return min(remaining_prompt, max_chunk)
```

### 3.2 调度器如何跟踪 Partial Prefill

在 vLLM v1 的 `Request` 对象中，以下字段跟踪 partial prefill 状态：

```python
class Request:
    # Prompt 相关
    prompt_token_ids: list[int]       # 完整的 prompt tokens
    num_prompt_tokens: int            # prompt 总长度

    # 已处理状态
    num_computed_tokens: int          # 已经计算过的 token 数
    # 对于 partial prefill:
    #   0 < num_computed_tokens < num_prompt_tokens

    @property
    def num_tokens_to_schedule(self) -> int:
        """本轮需要调度的 token 数"""
        if self.num_computed_tokens < self.num_prompt_tokens:
            # 还在 prefill 阶段
            remaining = self.num_prompt_tokens - self.num_computed_tokens
            return remaining  # 调度器会进一步限制为 chunk_size
        else:
            # decode 阶段
            return 1

    @property
    def is_prefill(self) -> bool:
        return self.num_computed_tokens < self.num_prompt_tokens
```

### 3.3 调度器中的 Chunked Prefill 逻辑

```python
# scheduler.py 中处理 waiting 请求的逻辑（简化版）
def _schedule_waiting(self, budget: SchedulingBudget):
    scheduled = []

    for req in self.waiting_queue:
        # 计算这个请求还需要 prefill 多少 tokens
        num_remaining = req.num_prompt_tokens - req.num_computed_tokens

        if self.enable_chunked_prefill:
            # 关键：chunk 大小取决于剩余预算
            num_tokens = min(num_remaining, budget.remaining_token_budget())

            # 如果预算不足以处理任何 token，停止
            if num_tokens == 0:
                break

            # 即使不能一次完成整个 prefill，也可以处理部分
        else:
            # 非 chunked 模式：必须一次处理完整个 prompt
            num_tokens = num_remaining
            if not budget.can_schedule(num_tokens, num_new_seqs=1):
                break  # 预算不足，放弃

        # 尝试分配 KV Cache
        status = self.kv_cache_manager.allocate_slots(req, num_tokens)
        if status != AllocationStatus.OK:
            break

        # 更新预算和请求状态
        budget.consume(num_tokens, num_new_seqs=1)
        req.num_computed_tokens += num_tokens

        if req.num_computed_tokens >= req.num_prompt_tokens:
            # Prefill 完成，移入 running
            self.waiting_queue.remove(req.request_id)
            self.running_requests[req.request_id] = req
        # 否则保留在 waiting queue，下轮继续

        scheduled.append((req, num_tokens))

    return scheduled
```

## 4. Chunk Size 对性能指标的影响

### 4.1 TTFT（Time To First Token）

Chunk size 直接影响 TTFT：

$$TTFT = \lceil \frac{n_{prompt}}{chunk\_size} \rceil \times t_{iteration}$$

其中 $t_{iteration}$ 取决于 chunk 中的 token 数和并发 decode 请求数。

```
Prompt = 4096 tokens, 30 个 decode 请求并发

chunk_size=4096 (无 chunked prefill):
  TTFT = 1 × 450ms = 450ms

chunk_size=2048:
  TTFT = 2 × 230ms = 460ms

chunk_size=1024:
  TTFT = 4 × 130ms = 520ms

chunk_size=512:
  TTFT = 8 × 80ms = 640ms

chunk_size=256:
  TTFT = 16 × 55ms = 880ms
```

**观察**：chunk 越小，TTFT 越高——因为每个 chunk 都有固定的开销（调度、kernel launch 等），且小 chunk 的 GPU 利用率较低（compute intensity 不够高）。

### 4.2 TBT（Time Between Tokens）

Chunk size 同样影响 TBT：

```
无 chunked prefill:
  正常 TBT: ~20ms
  Prefill 时 TBT 峰值: ~450ms
  TBT P99: ~450ms

chunk_size=512:
  正常 TBT: ~20ms
  Prefill chunk 时 TBT: ~70ms
  TBT P99: ~70ms

chunk_size=256:
  正常 TBT: ~20ms
  Prefill chunk 时 TBT: ~50ms
  TBT P99: ~50ms
```

**关键权衡**：TBT 和 TTFT 之间存在反向关系：
- Chunk 越小 → TBT 越好（decode 请求被阻塞时间更短）
- Chunk 越小 → TTFT 越差（prefill 分散到更多 iteration，总耗时增加）

### 4.3 吞吐量影响

Chunk size 对总吞吐量的影响较为微妙：

```
chunk_size 太小的问题：
  1. GPU 计算利用率低（小矩阵乘法 → memory-bandwidth bound）
  2. 调度开销占比增大（每个 iteration 都要做调度决策）
  3. Kernel launch overhead 相对增大
  → 总吞吐量下降

chunk_size 太大的问题：
  1. 退化为无 chunked prefill → TBT 飙升
  2. 无法高效利用 decode 请求释放的空闲 budget
  → TBT 不稳定

最优区间：
  通常在 512 ~ 2048 tokens 之间
  取决于：模型大小、GPU 类型、并发量、prompt 长度分布
```

## 5. Sarathi-Serve：系统化的 Chunked Prefill

### 5.1 论文贡献

[Sarathi-Serve (2024)](https://arxiv.org/abs/2403.02310) 是第一个系统化分析和优化 chunked prefill 的工作。其核心贡献：

**1. Stall-free Batching**

Sarathi-Serve 提出了"零停顿批处理"的概念：通过精心选择 chunk size，保证每个 iteration 的执行时间**几乎恒定**，从而消除 decode 请求的延迟波动。

```
目标: 让每个 iteration 的计算量 ≈ 恒定值 C

C = chunk_size × cost_per_prefill_token + num_decode_reqs × cost_per_decode_token

→ chunk_size = (C - num_decode_reqs × cost_per_decode_token) / cost_per_prefill_token
```

**2. 均匀 Chunk 的高效性**

论文证明了：当所有 iteration 的计算量相近时：
- GPU 利用率最高（避免了闲置和过载的交替）
- TBT 方差最小
- 总吞吐量接近理论最优

**3. Pipe-like 执行**

在多 GPU 的 pipeline parallelism 场景下，均匀 chunk 还可以减少 pipeline bubble：

```
Without uniform chunks (pipeline parallelism):
GPU0: [big_prefill]       [small_decode] [big_prefill]      ...
GPU1:            [big_prefill]       [small_decode]    ...
                 ↑ bubble            ↑ bubble

With uniform chunks:
GPU0: [chunk+decode] [chunk+decode] [chunk+decode] ...
GPU1:        [chunk+decode] [chunk+decode] [chunk+decode] ...
             ↑ minimal bubble
```

### 5.2 Sarathi-Serve 的调度算法

```python
# Sarathi-Serve 调度算法伪代码
def schedule_sarathi(
    waiting_queue,
    running_queue,
    target_batch_tokens,  # 目标每轮 token 数
):
    scheduled_tokens = 0

    # 1. 先安排所有 decode 请求（每个 1 token）
    decode_reqs = list(running_queue)
    scheduled_tokens += len(decode_reqs)

    # 2. 用剩余预算安排 prefill chunks
    remaining = target_batch_tokens - scheduled_tokens
    prefill_schedule = []

    for req in waiting_queue:
        if remaining <= 0:
            break
        chunk_size = min(
            req.remaining_prompt_tokens,
            remaining
        )
        prefill_schedule.append((req, chunk_size))
        remaining -= chunk_size

    return decode_reqs, prefill_schedule
```

## 6. 最优 Chunk Size 选择指南

### 6.1 理论分析框架

最优 chunk size 应该平衡以下目标：

$$\text{chunk\_size}^* = \arg\min_{c} \; \lambda_1 \cdot TTFT(c) + \lambda_2 \cdot TBT_{p99}(c) - \lambda_3 \cdot Throughput(c)$$

其中 $\lambda_1, \lambda_2, \lambda_3$ 是业务权重。

### 6.2 经验法则

| 场景 | 推荐 chunk_size | 理由 |
|------|----------------|------|
| 交互式聊天（低延迟优先） | 256 ~ 512 | TBT 敏感 |
| 代码补全（快速响应） | 512 ~ 1024 | 平衡 TTFT 和 TBT |
| 批量文档处理（吞吐优先） | 2048 ~ 4096 | 最大化 GPU 利用率 |
| 长上下文应用（128K+ prompt） | 1024 ~ 2048 | 避免极长 prefill 阻塞 |
| 混合工作负载 | 512 | 通用平衡点 |

### 6.3 硬件相关考量

```
A100 (80GB):
  Compute: 312 TFLOPS (BF16)
  Memory BW: 2 TB/s
  → 在 chunk_size ≥ 256 时，prefill 已经 compute-bound
  → 推荐 chunk_size: 512 ~ 2048

H100 (80GB):
  Compute: 989 TFLOPS (BF16)
  Memory BW: 3.35 TB/s
  → 需要更大 chunk_size 才能 compute-bound
  → 推荐 chunk_size: 1024 ~ 4096

H200 (141GB):
  Compute: 989 TFLOPS (BF16)
  Memory BW: 4.8 TB/s
  → 更高带宽，需要更大 chunk
  → 推荐 chunk_size: 1024 ~ 4096
```

### 6.4 实测调优流程

```bash
# Step 1: 基准测试（无 chunked prefill）
vllm serve model_name --disable-chunked-prefill

# Step 2: 启用 chunked prefill，测试不同 max_num_batched_tokens
for budget in 512 1024 2048 4096 8192; do
    vllm serve model_name \
        --enable-chunked-prefill \
        --max-num-batched-tokens $budget

    # 用 benchmark 工具测试
    python benchmarks/benchmark_serving.py \
        --dataset sharegpt \
        --request-rate 10 \
        --num-prompts 1000
done

# Step 3: 对比 TTFT P50/P99、TBT P50/P99、吞吐量
# 选择满足 SLA 且吞吐量最高的配置
```

## 7. 高级话题：Chunked Prefill 与其他优化的交互

### 7.1 与 Prefix Caching 的交互

当启用 prefix caching（Ch02）时，chunked prefill 的行为会发生变化：

```
Prompt = [system_prompt(1024)] + [user_input(2048)]
如果 system_prompt 已经 cached:

Without prefix caching:
  需要 prefill 3072 tokens → 6 个 512-token chunks

With prefix caching:
  cached 1024 tokens → 只需 prefill 2048 tokens → 4 个 512-token chunks
  TTFT 降低 ~33%
```

调度器在计算 chunk 时会先扣除已缓存的 token：

```python
num_to_prefill = req.num_prompt_tokens - req.num_computed_tokens
# num_computed_tokens 包含了 prefix cache hit 的部分
```

### 7.2 与 Speculative Decoding 的交互

Speculative Decoding（Ch07）生成多个 draft tokens，验证时需要处理多个 token。这与 chunked prefill 的预算控制产生交互：

```
正常 decode: 每请求 1 token
Speculative decode (k=5): 每请求最多 5 tokens

→ 单个 speculative decode 请求消耗更多 budget
→ 需要调整 max_num_batched_tokens 来适应
```

### 7.3 与 Disaggregated Serving 的交互

在 Prefill-Decode 分离架构（Ch05）中，chunked prefill 变得不那么必要——因为 prefill 和 decode 在不同的 GPU 上执行，不存在互相阻塞的问题。

但在 prefill 节点内部，如果有多个 prefill 请求排队，chunked prefill 仍然有助于提高 prefill 节点的利用率和公平性。

## 8. 参考资料

- [Sarathi-Serve: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills (2024)](https://arxiv.org/abs/2403.02310)
- [S3: Increasing GPU Utilization during Generative Inference for Higher Throughput (2023)](https://arxiv.org/abs/2306.06000)
- [vLLM Chunked Prefill 文档](https://docs.vllm.ai/en/latest/features/performance.html)
- [vLLM v1 Scheduler 源码](https://github.com/vllm-project/vllm/tree/main/vllm/v1/core/sched)
