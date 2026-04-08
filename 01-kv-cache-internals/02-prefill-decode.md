# Prefill vs Decode 的本质差异

> 理解 Prefill 和 Decode 两个阶段的计算特性差异，是优化 LLM Serving 性能的基础。

## 1. 两阶段概述

LLM 推理的自回归生成过程分为两个截然不同的阶段：

```
用户输入 prompt (1000 tokens)          模型生成 response (200 tokens)
┌────────────────────────────┐   ┌─────────────────────────────────┐
│       Prefill 阶段         │   │          Decode 阶段             │
│                            │   │                                 │
│  一次性处理 1000 个 token   │   │  每次处理 1 个 token, 共 200 次  │
│  生成 1000 组 KV Cache     │   │  每次追加 1 组 KV Cache          │
│  输出第一个 token          │   │  逐个输出后续 token              │
│                            │   │                                 │
│  计算密集 (Compute-bound)  │   │  访存密集 (Memory-bound)        │
│  GPU 利用率高              │   │  GPU 利用率低                   │
└────────────────────────────┘   └─────────────────────────────────┘
         TTFT                              TBT × N
```

---

## 2. Prefill 阶段：计算密集

### 2.1 计算过程

Prefill 阶段一次性处理整个 prompt，生成所有 token 的 KV Cache：

```python
# Prefill: 输入完整 prompt
prompt_tokens = [tok_0, tok_1, ..., tok_{s-1}]  # s 个 token

# 对每一层 Transformer:
# 1. 计算 Q, K, V (矩阵乘法)
Q = X @ W_Q   # [s, d] × [d, d] → [s, d]    大矩阵乘法
K = X @ W_K   # [s, d] × [d, d] → [s, d]
V = X @ W_V   # [s, d] × [d, d] → [s, d]

# 2. 计算 Attention
#    QK^T: [s, d_h] × [d_h, s] → [s, s]    大矩阵乘法
#    score × V: [s, s] × [s, d_h] → [s, d_h]
attn_output = softmax(Q @ K.T / sqrt(d_h)) @ V

# 3. 存储 K, V 到 KV Cache
cache.update(K, V)
```

**关键特征**：所有操作都是**大矩阵乘法**（GEMM），GPU 的计算单元被充分利用。

### 2.2 计算量分析

对于单层 Transformer 的 Prefill，主要 FLOPS 来自：

| 操作 | 形状 | FLOPS |
|------|------|-------|
| $Q = XW_Q$ | $[s, d] \times [d, d]$ | $2 \times s \times d^2$ |
| $K = XW_K$ | $[s, d] \times [d, d]$ | $2 \times s \times d^2$ |
| $V = XW_V$ | $[s, d] \times [d, d]$ | $2 \times s \times d^2$ |
| $QK^T$ | $[s, d] \times [d, s]$ | $2 \times s^2 \times d$ |
| $\text{score} \times V$ | $[s, s] \times [s, d]$ | $2 \times s^2 \times d$ |
| $O = \text{attn} \cdot W_O$ | $[s, d] \times [d, d]$ | $2 \times s \times d^2$ |
| FFN (上投影) | $[s, d] \times [d, 4d]$ | $2 \times s \times 4d^2$ |
| FFN (下投影) | $[s, 4d] \times [4d, d]$ | $2 \times s \times 4d^2$ |

**Prefill 总 FLOPS ≈ $2 \times s \times (12d^2 + 2s \times d) \times L$**

当 $s \ll 12d$（通常成立，如 $d=8192, s=4096$）时，FLOPS 与 $s$ 近似**线性关系**，由线性层的矩阵乘法主导。

### 2.3 Arithmetic Intensity

**Arithmetic Intensity（算术强度）** 定义为每字节内存访问对应的浮点运算数：

$$
\text{AI} = \frac{\text{FLOPS}}{\text{Bytes accessed}}
$$

对于 Prefill 中的矩阵乘法 $[s, d] \times [d, d]$：

$$
\text{AI}_\text{prefill} = \frac{2 \times s \times d^2}{(s \times d + d^2 + s \times d) \times \text{sizeof}} \approx \frac{2 \times s \times d}{(2s + d) \times \text{sizeof}}
$$

以 LLaMA-3-70B（$d=8192$）、$s=2048$、FP16 为例：

$$
\text{AI}_\text{prefill} = \frac{2 \times 2048 \times 8192}{(2 \times 2048 + 8192) \times 2} \approx \frac{33.5M}{24.6K} \approx 1365 \text{ FLOP/Byte}
$$

这远高于 H100 的 **ridge point**（约 295 FLOP/Byte for FP16），因此 Prefill 是 **compute-bound**。

---

## 3. Decode 阶段：访存密集

### 3.1 计算过程

Decode 阶段每步仅处理 **1 个新 token**：

```python
# Decode: 每步只有 1 个新 token
new_token = [tok_s]  # 1 个 token

# 对每一层 Transformer:
# 1. 计算新 token 的 Q, K, V
q = x @ W_Q   # [1, d] × [d, d] → [1, d]    矩阵-向量乘法！
k = x @ W_K   # [1, d] × [d, d] → [1, d]
v = x @ W_V   # [1, d] × [d, d] → [1, d]

# 2. 追加 k, v 到 cache
cache.append(k, v)

# 3. 计算 Attention (需要读取所有历史 KV)
#    q @ K_cache^T: [1, d_h] × [d_h, s+1] → [1, s+1]  向量-矩阵乘法
#    score × V_cache: [1, s+1] × [s+1, d_h] → [1, d_h]
K_all = cache.get_keys()    # 需要从显存读取 s+1 组 K
V_all = cache.get_values()  # 需要从显存读取 s+1 组 V
attn_output = softmax(q @ K_all.T / sqrt(d_h)) @ V_all
```

**关键问题**：权重矩阵 $W_Q, W_K, W_V, W_O$ 和 FFN 权重需要**完整加载**，但只用来处理 1 个 token。这意味着大量的显存带宽被权重读取消耗。

### 3.2 Arithmetic Intensity 分析

Decode 阶段的线性层变成了矩阵-向量乘法 $[1, d] \times [d, d]$：

$$
\text{AI}_\text{decode} = \frac{2 \times 1 \times d^2}{(1 \times d + d^2 + 1 \times d) \times \text{sizeof}} \approx \frac{2d}{(d + 2) \times \text{sizeof}} \approx \frac{2}{\text{sizeof}}
$$

对于 FP16（sizeof=2）：

$$
\text{AI}_\text{decode} \approx 1 \text{ FLOP/Byte}
$$

H100 的 ridge point 约为 260 FLOP/Byte，而 Decode 的 AI 仅为 1，远低于 ridge point。这意味着 Decode 是**极度 memory-bound**，GPU 的计算单元大部分时间在等待数据从显存传输。

### 3.3 Roofline Model 可视化

```
性能 (TFLOPS)
    │
312 ├─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ── Compute Ceiling (H100 FP16)
    │                              ╱
    │                           ╱
    │                        ╱
    │                     ╱
    │                  ╱ ← Memory Bandwidth Ceiling (3.35 TB/s)
    │               ╱
    │            ╱
    │         ╱    ↑
    │      ╱       │ Ridge Point ≈ 295 (FP16, H100 理论值)
    │   ╱          │
    │╱─────────────┼──────────────────────────────
    0   1    10   100   295  1000  10000
              Arithmetic Intensity (FLOP/Byte)
              ↑              ↑
           Decode          Prefill
        (AI ≈ 1)        (AI ≈ 1000+)
```

**解读**：
- **Decode**（AI ≈ 1）位于 roofline 的斜坡上，性能受限于**显存带宽**
- **Prefill**（AI ≈ 1000+）位于 roofline 的平顶，性能受限于**计算能力**

---

## 4. 为什么 Prefill 吞吐量远高于 Decode？

### 4.1 Token 处理效率对比

| 指标 | Prefill (s=2048) | Decode (单步) | 差距 |
|------|-----------------|--------------|------|
| 处理 token 数 | 2048 | 1 | 2048x |
| 权重加载次数 | 1 次 | 1 次 | 相同 |
| 计算利用率 | ~80%+ | ~1-5% | ~20-80x |
| 有效吞吐量 | 数万 tok/s | 数十 tok/s | ~1000x |

**核心原因**：Decode 每步处理 1 个 token，但必须加载全部模型权重。权重加载的开销被"摊销"到仅 1 个 token 上，利用率极低。

### 4.2 Batching 缓解 Decode 低效

Decode 阶段可以通过 **Continuous Batching** 提升效率——将多个请求的 decode 步合并为一个 batch：

```
无 Batching (batch_size=1):
  权重加载: d × d × sizeof = 134 MB (LLaMA-70B 单层 W_Q)
  计算: 2 × d × d = 134M FLOPS (1 token)
  AI ≈ 1 FLOP/Byte

Batching (batch_size=32):
  权重加载: d × d × sizeof = 134 MB (同样, 只加载一次)
  计算: 2 × 32 × d × d = 4.29G FLOPS (32 tokens)
  AI ≈ 32 FLOP/Byte

Batching (batch_size=256):
  AI ≈ 256 FLOP/Byte → 接近 ridge point!
```

但 batch size 增大会增加 KV Cache 显存占用，存在**显存限制**：

$$
\text{max\_batch\_size} = \frac{\text{KV Cache 可用显存}}{2 \times L \times n_{kv} \times d_h \times s_\text{avg} \times \text{sizeof}}
$$

---

## 5. Batch 中混合 Prefill 和 Decode 的挑战

### 5.1 问题描述

在生产环境中，同一时刻既有新到达的请求需要 Prefill，也有正在生成的请求需要 Decode。如何在同一个 batch 中混合处理？

```
时刻 T 的请求状态:
  请求 A: 正在 Decode (已生成 50 tokens)
  请求 B: 正在 Decode (已生成 120 tokens)
  请求 C: 新到达, 需要 Prefill (prompt = 2000 tokens)
  请求 D: 新到达, 需要 Prefill (prompt = 500 tokens)
```

### 5.2 混合 Batch 的困难

**问题 1：序列长度不一致**

```
Prefill 请求: 输入 [2000 tokens] → 输出 [2000 组 KV]
Decode 请求:  输入 [1 token]     → 输出 [1 组 KV]

# 无法简单拼成一个规则的 batch tensor
# Prefill: input_ids shape = [1, 2000]
# Decode:  input_ids shape = [1, 1]
# 合并？ → 需要 padding 到 [3, 2000]，浪费巨大
```

**问题 2：Prefill 和 Decode 的最优策略不同**

| 维度 | Prefill 最优 | Decode 最优 |
|------|-------------|-------------|
| Attention kernel | FlashAttention (因果 mask) | PagedAttention (稀疏读取) |
| 计算特征 | 大矩阵乘法，高并行度 | 矩阵-向量乘法，低并行度 |
| 调度优先级 | 尽快完成降低 TTFT | 持续产出降低 TBT |
| Memory pattern | 写入大块连续 KV | 读取分散的 KV blocks |

**问题 3：Prefill 拖慢 Decode**

如果在同一 batch 中混入一个长 Prefill 请求，Decode 请求必须等待 Prefill 完成才能继续：

```
时间 ──────────────────────────────────────────→

无混合:
  Decode batch:  [A, B] ──→ 完成 (10ms)
  Prefill:       [C] ──────────────────→ 完成 (100ms)

混合:
  Mixed batch:   [A, B, C] ──────────────────→ 完成 (100ms+)
                  ↑ A 和 B 被 C 拖慢！TBT 退化 10x
```

### 5.3 解决方案

**方案 1：Chunked Prefill（分块 Prefill）**

将长 Prefill 拆分为多个 chunk，每个 chunk 的大小与 Decode batch 相当：

```python
# 将 2000 token 的 Prefill 拆分为 4 个 chunk
chunk_size = 512
chunks = [prompt[i:i+chunk_size] for i in range(0, len(prompt), chunk_size)]

# 每个 iteration 混合 1 个 prefill chunk + decode 请求
iteration_1: [decode_A, decode_B, prefill_C_chunk_0]  # 512+1+1 tokens
iteration_2: [decode_A, decode_B, prefill_C_chunk_1]  # 512+1+1 tokens
iteration_3: [decode_A, decode_B, prefill_C_chunk_2]  # 512+1+1 tokens
iteration_4: [decode_A, decode_B, prefill_C_chunk_3]  # 512+1+1 tokens
```

vLLM 和 SGLang 都支持 Chunked Prefill，通过 `--enable-chunked-prefill` 启用。

**方案 2：Disaggregated Prefill（分离式架构）**

将 Prefill 和 Decode 部署在不同的 GPU 集群上（详见 Chapter 05）。

---

## 6. vLLM 中 Prefill 和 Decode 的代码路径差异

### 6.1 调度器中的区分

vLLM 的调度器根据请求状态决定处理方式：

```python
# vllm/v1/core/scheduler.py (简化)
class Scheduler:
    def schedule(self) -> SchedulerOutput:
        """调度下一个 batch。"""
        scheduled_new_reqs = []      # 需要 Prefill 的新请求
        scheduled_running_reqs = []  # 正在 Decode 的请求
        
        # 1. 优先处理正在运行的 Decode 请求
        for req in self.running:
            if self._can_schedule(req):
                scheduled_running_reqs.append(req)
        
        # 2. 从等待队列中取出新请求进行 Prefill
        for req in self.waiting:
            if self._can_schedule(req):
                scheduled_new_reqs.append(req)
                self.running.append(req)
        
        return SchedulerOutput(
            scheduled_new_reqs=scheduled_new_reqs,
            scheduled_running_reqs=scheduled_running_reqs,
            # ...
        )
```

### 6.2 Model Runner 中的处理

```python
# vllm/v1/worker/gpu_model_runner.py (简化)
class GPUModelRunner:
    def execute_model(self, scheduler_output: SchedulerOutput):
        """执行一个 batch 的前向传播。"""
        
        # 1. 构建输入
        #    Prefill 请求: 完整 prompt token IDs
        #    Decode 请求: 仅最新生成的 1 个 token ID
        input_ids = self._prepare_inputs(scheduler_output)
        
        # 2. 构建 Attention metadata
        #    这里是 Prefill vs Decode 路径分叉的关键
        attn_metadata = self._prepare_attention_metadata(
            scheduler_output
        )
        #    attn_metadata 包含:
        #    - num_prefill_tokens: Prefill 阶段的 token 总数
        #    - num_decode_tokens: Decode 阶段的 token 总数
        #    - prefill_seq_lens: 每个 Prefill 请求的长度
        #    - block_tables: Decode 请求的 Block Table
        
        # 3. 前向传播
        output = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=self.gpu_cache,
            attn_metadata=attn_metadata,
        )
        
        return output
```

### 6.3 Attention Kernel 的分支

在 Attention 计算层面，Prefill 和 Decode 使用不同的 kernel：

```python
# vllm/attention/backends/flash_attn.py (简化)
class FlashAttentionImpl:
    def forward(self, query, key, value, kv_cache, attn_metadata):
        
        if attn_metadata.num_prefill_tokens > 0:
            # ---- Prefill 路径 ----
            # 使用 FlashAttention kernel
            # 完整的 causal attention，QKV 形状为 [num_prefill_tokens, ...]
            prefill_output = flash_attn_varlen_func(
                q=query[:num_prefill_tokens],
                k=key[:num_prefill_tokens],
                v=value[:num_prefill_tokens],
                cu_seqlens_q=attn_metadata.prefill_seq_start_loc,
                cu_seqlens_k=attn_metadata.prefill_seq_start_loc,
                max_seqlen_q=attn_metadata.max_prefill_seq_len,
                max_seqlen_k=attn_metadata.max_prefill_seq_len,
                causal=True,
            )
        
        if attn_metadata.num_decode_tokens > 0:
            # ---- Decode 路径 ----
            # 使用 PagedAttention kernel
            # 从分页 KV Cache 中读取历史 KV
            decode_output = paged_attention_v1(
                query=query[num_prefill_tokens:],
                key_cache=kv_cache[0],
                value_cache=kv_cache[1],
                block_tables=attn_metadata.block_tables,
                seq_lens=attn_metadata.decode_seq_lens,
                # ...
            )
```

---

## 7. `is_prompt` 标志的传播

### 7.1 请求级别

```python
# 每个 SequenceGroup 有一个 is_prefill 属性
# vllm/sequence.py (概念模型)
class SequenceGroup:
    @property
    def is_prefill(self) -> bool:
        """判断该请求是否处于 Prefill 阶段。"""
        # 如果已经生成了至少一个 token，则为 Decode
        return self.get_num_computed_tokens() < self.get_prompt_len()
```

### 7.2 传播链路

```
用户请求到达
    │
    ▼
API Server (OpenAI-compatible)
    │ 创建 SequenceGroup, is_prefill=True
    ▼
Scheduler.schedule()
    │ 区分 new_reqs (prefill) vs running (decode)
    │ 构建 SchedulerOutput
    ▼
ModelRunner._prepare_attention_metadata()
    │ 计算 num_prefill_tokens, num_decode_tokens
    │ 生成不同的 attention 元数据
    ▼
Attention Layer
    │ 根据 num_prefill_tokens > 0 选择 kernel
    │ Prefill → FlashAttention
    │ Decode  → PagedAttention
    ▼
KV Cache 写入
    │ Prefill: 写入 s 个 token 的 KV
    │ Decode:  写入 1 个 token 的 KV
    ▼
输出 token(s)
    │ Prefill 完成后: is_prefill → False
    │ 后续步都走 Decode 路径
```

---

## 8. 用 Profiler 区分 Prefill 和 Decode 的 GPU 利用率

### 8.1 使用 PyTorch Profiler

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    # 运行推理
    output = model.generate(input_ids, max_new_tokens=100)

# 查看耗时分布
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

### 8.2 Prefill vs Decode 的 Profiler 特征

| 指标 | Prefill | Decode |
|------|---------|--------|
| SM Utilization | 80-95% | 5-20% |
| Memory Bandwidth Utilization | 30-50% | 80-95% |
| 主要 kernel | `ampere_fp16_s16816gemm_*` | `paged_attention_v1_*` |
| kernel 耗时分布 | GEMM 占 70%+ | Attention 占 40%+, GEMM 占 30% |
| 单步耗时 | 较长（数十~数百 ms） | 较短（数~数十 ms） |

### 8.3 使用 NVIDIA Nsight Systems

```bash
# 使用 nsys 收集 trace
nsys profile -o llm_trace \
    python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8B-Instruct

# 在 Nsight Systems GUI 中:
# 1. 找到第一个大块 GEMM kernel 集群 → Prefill 阶段
# 2. 之后的周期性小 kernel 集群 → Decode 阶段
# 3. 对比两者的 SM Occupancy 和 Memory Throughput
```

### 8.4 典型 Profiling 结果示例

```
LLaMA-3-8B, Prompt=1024 tokens, Generate=128 tokens, H100:

Prefill 阶段:
  总耗时: 45 ms
  吞吐量: 1024 / 0.045 = 22,756 tokens/s
  GPU SM 利用率: 87%
  显存带宽利用率: 35%

Decode 阶段 (128 步):
  每步耗时: 8 ms
  总耗时: 128 × 8 = 1,024 ms
  吞吐量: 128 / 1.024 = 125 tokens/s
  GPU SM 利用率: 12%
  显存带宽利用率: 82%

比较:
  Prefill 吞吐量 / Decode 吞吐量 = 22,756 / 125 ≈ 182x
  总推理时间: 45 + 1024 = 1,069 ms
  Prefill 占比: 4.2%
  Decode 占比: 95.8%
```

---

## 9. TTFT vs TBT 的权衡

### 9.1 定义

| 指标 | 全称 | 含义 | 对应阶段 |
|------|------|------|---------|
| **TTFT** | Time To First Token | 从请求到达到第一个 token 生成的延迟 | Prefill |
| **TBT** | Time Between Tokens | 相邻两个输出 token 之间的延迟 | Decode |
| **E2E Latency** | End-to-End Latency | 整个请求的端到端延迟 | TTFT + TBT × (n-1) |

### 9.2 用户体验视角

```
用户发送请求
    │
    │←── TTFT ──→│
    │             │← TBT →│← TBT →│← TBT →│
    │             ▼        ▼        ▼        ▼
    ────────────[token1] [token2] [token3] [token4] ...
    等待中...     开始流式输出

用户感知:
  - TTFT 决定 "多久开始看到回复" → 对交互式应用至关重要
  - TBT 决定 "回复的流畅度" → 影响阅读体验
  - 通常 TBT < 50ms 用户感受流畅
  - TTFT < 2-3s 用户可以接受
```

### 9.3 不同应用场景的优先级

| 场景 | TTFT 优先级 | TBT 优先级 | 原因 |
|------|-----------|-----------|------|
| 聊天机器人 | **高** | **高** | 用户等待首个回复 + 流式体验 |
| 代码补全 | **极高** | 中 | 需要即时响应 |
| 批量翻译 | 低 | 低 | 非交互，关注总吞吐量 |
| RAG 检索增强 | **高** | 中 | 长 prompt 导致 TTFT 天然较高 |
| 长文档生成 | 中 | **高** | 生成数千 token，TBT 累积效应显著 |
| Agent 工具调用 | **极高** | **极高** | 每一轮都是 Prefill + 短 Decode |

### 9.4 TTFT 与 TBT 的矛盾

优化 TTFT 和 TBT 往往存在 **trade-off**：

**增大 batch size**：
- TBT ↑（每个 decode 步要处理更多请求）
- TTFT ↑（新请求等待 batch 中有空位）
- 吞吐量 ↑（整体 token/s 提升）

**启用 Chunked Prefill**：
- TTFT ↑（Prefill 被分成多步完成，首 token 延迟增加）
- TBT ↓（Decode 请求不会被长 Prefill 阻塞）

**Prefill-Decode 分离**：
- TTFT ↓（Prefill 机器不受 Decode 影响）
- TBT ↓（Decode 机器不受 Prefill 影响）
- 成本 ↑（需要更多硬件 + 跨节点 KV 传输）

### 9.5 实际度量方法

```python
import time

# 度量 TTFT
start = time.perf_counter()
stream = client.chat.completions.create(
    model="llama-3-70b",
    messages=[{"role": "user", "content": prompt}],
    stream=True,
)

ttft = None
tbt_list = []
prev_time = None

for chunk in stream:
    now = time.perf_counter()
    if chunk.choices[0].delta.content:
        if ttft is None:
            ttft = now - start  # Time To First Token
        elif prev_time is not None:
            tbt_list.append(now - prev_time)  # Time Between Tokens
        prev_time = now

print(f"TTFT: {ttft*1000:.1f} ms")
print(f"Avg TBT: {sum(tbt_list)/len(tbt_list)*1000:.1f} ms")
print(f"P99 TBT: {sorted(tbt_list)[int(len(tbt_list)*0.99)]*1000:.1f} ms")
```

---

## 10. 总结

| 维度 | Prefill | Decode |
|------|---------|--------|
| **处理 token 数** | 整个 prompt ($s$ tokens) | 每步 1 token |
| **计算特性** | Compute-bound (GEMM) | Memory-bound (GEMV) |
| **AI (FP16)** | ~1000+ FLOP/Byte | ~1 FLOP/Byte |
| **GPU 利用率** | 80-95% | 5-20% (单请求) |
| **KV Cache 操作** | 批量写入 | 追加 1 组 + 读取全部 |
| **Attention kernel** | FlashAttention (dense) | PagedAttention (sparse read) |
| **对应指标** | TTFT | TBT |
| **优化方向** | 更快的 GEMM, prefix caching | Batching, 量化, 投机解码 |

**一句话总结**：Prefill 是"做很多计算、写少量数据"，Decode 是"做很少计算、读大量数据"——两者本质上需要完全不同的优化策略，这也是为什么 Disaggregated Serving 架构会将它们分离到不同的硬件上运行。

---

## 参考资料

- [Efficiently Scaling Transformer Inference (Google, 2022)](https://arxiv.org/abs/2211.05102) — Roofline 分析的经典参考
- [Splitwise: Efficient Generative LLM Inference (ISCA 2024)](https://arxiv.org/abs/2311.18677) — Prefill-Decode 分离的早期工作
- [Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2308.16369) — Chunked Prefill 原始论文
- [vLLM 源码](https://github.com/vllm-project/vllm) — 调度器和 Model Runner 实现
