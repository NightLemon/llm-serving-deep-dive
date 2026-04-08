# Context Parallel

> 本节分析超长上下文推理（>128K tokens）的挑战，Context Parallel 的原理与实现（Ring Attention 等），以及在 vLLM 中的支持现状和适用场景。

## 1. 超长上下文推理的挑战

### 1.1 长上下文的趋势

```
主流模型上下文长度演进:
  GPT-3 (2020):         2K tokens
  GPT-3.5 (2023):       4K → 16K tokens
  Claude 2 (2023):      100K tokens
  GPT-4 Turbo (2023):   128K tokens
  Claude 3 (2024):      200K tokens
  Gemini 1.5 (2024):    1M → 10M tokens
  Llama-3.1 (2024):     128K tokens

应用场景:
  - 长文档问答 (论文、法律合同、代码库)
  - 多轮长对话 (累积上下文)
  - RAG with large context (检索后注入大量文档)
  - 代码理解 (整个 repo 作为上下文)
```

### 1.2 显存挑战：KV Cache 的线性增长

```
KV Cache 显存 = 2 × num_layers × num_kv_heads × head_dim × seq_len × dtype_bytes

以 Llama-3.1-70B (80 层, 8 KV heads, head_dim=128, bf16) 为例:

  seq_len = 4K:    2 × 80 × 8 × 128 × 4096 × 2     = 1.28 GB
  seq_len = 32K:   2 × 80 × 8 × 128 × 32768 × 2    = 10.24 GB
  seq_len = 128K:  2 × 80 × 8 × 128 × 131072 × 2   = 40.96 GB
  seq_len = 512K:  2 × 80 × 8 × 128 × 524288 × 2   = 163.84 GB
  seq_len = 1M:    2 × 80 × 8 × 128 × 1048576 × 2  = 327.68 GB

TP=8 (每卡分到的 KV Cache):
  128K:  40.96 / 8 = 5.12 GB/卡  → 可接受
  512K:  163.84 / 8 = 20.48 GB/卡 → 已经很紧张
  1M:    327.68 / 8 = 40.96 GB/卡 → 几乎占满整张 H100
```

### 1.3 计算挑战：Attention 的二次复杂度

```
标准 Self-Attention 计算量:
  QK^T:       O(n × n × d)     n = seq_len, d = head_dim
  Softmax:    O(n × n)
  Score × V:  O(n × n × d)
  
  总计: O(n² × d) per head per layer

Prefill 阶段 (全量 attention):
  seq_len = 128K: 128K² ≈ 16.4B 操作 per head per layer
  seq_len = 1M:   1M²   ≈ 1T 操作 per head per layer
  
  1M context 的计算量是 128K 的 ~64 倍!

Decode 阶段 (增量 attention):
  每步只有 1 个新 query, 但要 attend 到所有 KV:
  计算量 = O(n × d) per head per layer
  线性增长, 但 n=1M 时仍然很大
```

### 1.4 单 GPU 放不下的场景

```
当 KV Cache 超出 TP 切分后的单卡显存:

场景: Llama-3.1-70B, 1M context, TP=8
  权重/卡: ~17.5 GB
  KV Cache/卡: ~41 GB
  总计: ~58.5 GB → 超出 H100 80GB 的可用空间
  (考虑 CUDA context、workspace 等开销)

此时有两个选择:
  1. 增加 TP size → 但 TP>8 通常需要跨节点, 通信延迟大
  2. Context Parallel → 沿序列维度切分, 每卡只存部分 context
```

## 2. Context Parallel 原理

### 2.1 基本概念

```
Context Parallel (CP, 也称 Sequence Parallel for Attention):

核心思想: 将输入序列按 context 维度切分到多个 GPU

                 ┌──── GPU 0 ────┐┌──── GPU 1 ────┐┌──── GPU 2 ────┐┌──── GPU 3 ────┐
Token 序列:      │ T[0:N/4]      ││ T[N/4:N/2]    ││ T[N/2:3N/4]  ││ T[3N/4:N]    │
KV Cache:        │ KV[0:N/4]    ││ KV[N/4:N/2]   ││ KV[N/2:3N/4] ││ KV[3N/4:N]   │
                 └───────────────┘└───────────────┘└───────────────┘└───────────────┘

每个 GPU:
  - 持有序列的 1/cp_size 部分的 KV Cache
  - Prefill 时只计算自己那部分的 Q 与所有 KV 的 attention
  - 需要通信来获取其他 GPU 上的 KV
```

### 2.2 与 TP 的区别

```
TP (Tensor Parallel):
  切分维度: head 维度 (或 hidden 维度)
  每个 GPU: 所有 token, 部分 head
  KV Cache: 每卡存部分 head 的全长序列
  通信: AllReduce (聚合不同 head 的计算结果)

CP (Context Parallel):
  切分维度: sequence 维度
  每个 GPU: 部分 token, 所有 head (或部分 head)
  KV Cache: 每卡存部分序列的全部 head
  通信: Ring pass KV / AllGather (交换不同 GPU 上的 KV)

可以组合使用:
  TP: 切 head 维度
  CP: 切 sequence 维度
  两者正交
```

### 2.3 为什么 CP 比增加 TP 更适合长上下文

```
方案 1: 单纯增加 TP (TP=16 跨 2 节点)
  - 每层 2 次 AllReduce × 跨节点 → 延迟大
  - 160 次跨节点 AllReduce per forward pass
  - 不适合延迟敏感的 decode 阶段

方案 2: TP=8 + CP=2 (节点内 TP, 跨/节点内 CP)
  - TP 通信仍在节点内 (NVLink, 快)
  - CP 通信只涉及 KV 的传递 (可以 overlap)
  - 且 CP 通信量 = KV data, 而 TP 通信量 = hidden states
  - CP 更适合 prefill 阶段 (数据量大, 可充分 overlap)
```

## 3. Ring Attention

### 3.1 基本原理

Ring Attention (论文: "Ring Attention with Blockwise Transformers for Near-Infinite Context", 2023) 是实现 Context Parallel 的核心算法。

```
Ring Attention 工作原理:

假设 4 个 GPU, 序列切为 4 块: [S0, S1, S2, S3]

Step 0: 各 GPU 用本地 KV 计算 attention
  GPU 0: Attn(Q0, KV0)  → partial_result_0
  GPU 1: Attn(Q1, KV1)  → partial_result_1
  GPU 2: Attn(Q2, KV2)  → partial_result_2
  GPU 3: Attn(Q3, KV3)  → partial_result_3
  
  同时: 沿 ring 传递 KV 到下一个 GPU
  GPU 0 → GPU 1: KV0
  GPU 1 → GPU 2: KV1
  GPU 2 → GPU 3: KV2
  GPU 3 → GPU 0: KV3

Step 1: 用接收到的 KV 计算 attention, 并继续传递
  GPU 0: Attn(Q0, KV3) + partial_result_0 → partial_result_0'
  GPU 1: Attn(Q1, KV0) + partial_result_1 → partial_result_1'
  GPU 2: Attn(Q2, KV1) + partial_result_2 → partial_result_2'
  GPU 3: Attn(Q3, KV2) + partial_result_3 → partial_result_3'
  
  同时: 继续沿 ring 传递 KV
  ...

Step 2: 同上
Step 3: 同上 (最后一步, 完成所有 KV 的 attention)

总共需要 cp_size 步, 每步传递 + 计算
```

### 3.2 通信与计算重叠

```
Ring Attention 的关键优势: 通信和计算可以完美重叠

Timeline (GPU 0):
  ┌───────────────────────────────────────────────────────────┐
  │ Step 0                                                    │
  │ ┌──────────────────────┐                                  │
  │ │ Compute: Attn(Q0,KV0)│                                  │
  │ ├──────────┬───────────┤                                  │
  │ │ Send KV0 │ Recv KV3  │ ← 通信与计算重叠                  │
  │ └──────────┴───────────┘                                  │
  │                                                           │
  │ Step 1                                                    │
  │ ┌──────────────────────┐                                  │
  │ │ Compute: Attn(Q0,KV3)│                                  │
  │ ├──────────┬───────────┤                                  │
  │ │ Send KV3 │ Recv KV2  │ ← 通信与计算重叠                  │
  │ └──────────┴───────────┘                                  │
  │ ...                                                       │
  └───────────────────────────────────────────────────────────┘

条件: 通信时间 ≤ 计算时间 → 通信完全被隐藏

计算时间 ∝ (N/cp_size)² × d   (block attention)
通信时间 ∝ (N/cp_size) × d    (传递 KV block)

通信/计算比 ∝ cp_size / N
N 足够大时, 通信可以完全被隐藏!
```

### 3.3 Online Softmax 的挑战

```
Ring Attention 需要在不同 KV block 间合并 attention 结果
标准 softmax 需要全局 max 和 sum → 不能简单地相加

解决方案: Online Softmax (FlashAttention 风格)

每个 block 的 attention:
  score_i = Q × K_i^T / sqrt(d)
  max_i = max(score_i)
  exp_i = exp(score_i - max_i)
  sum_i = sum(exp_i)
  out_i = exp_i × V_i / sum_i

合并两个 block 的结果:
  max_new = max(max_0, max_1)
  
  # 修正 block 0 的结果
  correction_0 = exp(max_0 - max_new)
  # 修正 block 1 的结果
  correction_1 = exp(max_1 - max_new)
  
  sum_new = sum_0 × correction_0 + sum_1 × correction_1
  out_new = (out_0 × sum_0 × correction_0 + out_1 × sum_1 × correction_1) / sum_new

这个合并过程是数值稳定的, 且可以增量进行
```

### 3.4 Ring Attention 伪代码

```python
def ring_attention(
    Q_local: torch.Tensor,    # [local_seq_len, num_heads, head_dim]
    K_local: torch.Tensor,    # [local_seq_len, num_kv_heads, head_dim]
    V_local: torch.Tensor,    # [local_seq_len, num_kv_heads, head_dim]
    cp_group: ProcessGroup,
    cp_size: int,
    cp_rank: int,
):
    """Ring Attention 实现 (简化)"""
    
    # 初始化输出和在线 softmax 状态
    out = torch.zeros_like(Q_local)
    lse = torch.full((Q_local.shape[0], Q_local.shape[1]), float('-inf'))
    
    # 当前持有的 KV
    K_recv = K_local.clone()
    V_recv = V_local.clone()
    
    for step in range(cp_size):
        # 异步发送当前 KV 到下一个 rank, 接收上一个 rank 的 KV
        if step < cp_size - 1:
            send_op = async_send(K_recv, next_rank(cp_rank), cp_group)
            recv_op = async_recv(K_buf, prev_rank(cp_rank), cp_group)
            send_op_v = async_send(V_recv, next_rank(cp_rank), cp_group)
            recv_op_v = async_recv(V_buf, prev_rank(cp_rank), cp_group)
        
        # 计算 attention with 当前 KV block
        # 使用 online softmax 更新
        block_out, block_lse = flash_attention(Q_local, K_recv, V_recv)
        
        # 合并结果 (online softmax merge)
        out, lse = merge_attention_output(out, lse, block_out, block_lse)
        
        # 等待通信完成
        if step < cp_size - 1:
            send_op.wait()
            recv_op.wait()
            send_op_v.wait()
            recv_op_v.wait()
            K_recv = K_buf.clone()
            V_recv = V_buf.clone()
    
    return out
```

## 4. CP 的其他实现方式

### 4.1 AllGather-based CP

```
AllGather 方式 (替代 Ring):

Step 1: AllGather 所有 GPU 的 KV
  每个 GPU 获得完整的 KV: [KV0, KV1, KV2, KV3]
  
Step 2: 每个 GPU 计算自己的 Q 与完整 KV 的 attention
  GPU 0: Attn(Q0, [KV0,KV1,KV2,KV3])
  GPU 1: Attn(Q1, [KV0,KV1,KV2,KV3])
  ...

优点:
  - 实现简单
  - 一次通信后计算独立

缺点:
  - 需要在每个 GPU 上临时存储完整的 KV → 显存峰值高
  - 通信不能与计算重叠（先通信再计算）
  - 适合显存充裕、序列不极端长的场景
```

### 4.2 Striped Attention

```
Striped Attention (交错切分):

普通切分:
  GPU 0: [T0, T1, ..., T_{N/4-1}]      (连续块)
  GPU 1: [T_{N/4}, ..., T_{N/2-1}]

交错切分:
  GPU 0: [T0, T4, T8, ...]              (间隔取样)
  GPU 1: [T1, T5, T9, ...]
  GPU 2: [T2, T6, T10, ...]
  GPU 3: [T3, T7, T11, ...]

优点:
  - 更均匀的 attention 分布 (causal mask 下)
  - 减少 load imbalance

原因:
  Causal attention 中, 前面的 token attend 到更少的 KV
  连续切分: GPU 0 (前部 token) 的计算量 < GPU 3 (后部 token)
  交错切分: 每个 GPU 的计算量更均衡
```

## 5. vLLM Context Parallel 支持

### 5.1 配置

```bash
# vLLM CP 配置 (需要较新版本)
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --context-parallel-size 2
# 总共需要 4 × 2 = 8 GPU

# 注意: CP 的使用通常需要:
# 1. 模型支持 (位置编码需要适配)
# 2. 序列足够长 (短序列用 CP 反而增加开销)
```

### 5.2 CP 在 vLLM 中的工作流

```
Prefill 阶段 (CP 主要发挥作用):

1. 长序列被切分到 CP group 的各 GPU:
   序列 [0:128K] → GPU 0: [0:64K], GPU 1: [64K:128K]

2. 每个 GPU 独立计算 embedding + 位置编码

3. 对于每一层:
   a. Attention: 使用 Ring Attention (或 AllGather)
      - 本地 QKV projection
      - Ring pass KV blocks
      - 计算 attention output
   b. MLP: 正常执行 (每个 GPU 独立处理自己的 token)
   c. LayerNorm: 需要 AllReduce (跨 CP 的 normalization)

4. 最终输出在 rank 0 上聚合

Decode 阶段:
  CP 在 decode 阶段的作用有限:
  - 每步只有 1 个新 token (per request)
  - 但需要 attend 到分散在各 GPU 上的 KV Cache
  - 可能需要 AllGather KV 或 Ring Attention
  - 通信开销相对计算可能较大
```

### 5.3 CP + TP 的交互

```
TP 和 CP 同时使用时的通信:

假设 TP=4, CP=2, 8 GPUs

GPU 映射:
  TP Group 0 (CP Rank 0): [GPU 0, 1, 2, 3]
  TP Group 1 (CP Rank 1): [GPU 4, 5, 6, 7]

每层通信:
  1. QKV Projection (Column Parallel) → 无通信
  2. Attention:
     a. CP 通信: Ring pass KV between TP groups
        GPU 0 ↔ GPU 4, GPU 1 ↔ GPU 5, ...
     b. TP 通信: 无 (attention 在各 head 上独立)
  3. Output Projection (Row Parallel) → TP AllReduce (节点内)
  4. MLP → TP AllReduce (节点内)

通信总结:
  - TP: 2 次 AllReduce/层 (节点内 NVLink)
  - CP: 1 次 Ring KV pass/层 (可能跨节点)
```

## 6. CP 的性能分析

### 6.1 显存节省

```python
def cp_memory_analysis(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    tp_size: int,
    cp_size: int,
    dtype_bytes: int = 2,
):
    """CP 的显存节省分析"""
    
    # 无 CP:
    kv_per_gpu_no_cp = (
        2 * num_layers * (num_kv_heads // tp_size) * head_dim 
        * seq_len * dtype_bytes
    )
    
    # 有 CP:
    kv_per_gpu_with_cp = (
        2 * num_layers * (num_kv_heads // tp_size) * head_dim 
        * (seq_len // cp_size) * dtype_bytes
    )
    
    return {
        "no_cp_GB": kv_per_gpu_no_cp / 1e9,
        "with_cp_GB": kv_per_gpu_with_cp / 1e9,
        "savings": 1 - 1/cp_size,
    }

# Llama-3.1-70B, 1M context, TP=8:
result = cp_memory_analysis(80, 8, 128, 1_000_000, 8, 1)
# no_cp: 40.96 GB/GPU → 超出单卡容量

result = cp_memory_analysis(80, 8, 128, 1_000_000, 8, 4)
# with_cp: 10.24 GB/GPU → 可行!
```

### 6.2 通信开销

```
Ring Attention 通信分析:

每步传递的 KV 数据量:
  KV_block_size = 2 × (num_kv_heads/tp_size) × head_dim × (seq_len/cp_size) × dtype_bytes

总通信量 (cp_size 步):
  total_comm = (cp_size - 1) × KV_block_size
  (最后一步不需要传递)

以 Llama-3.1-70B, seq_len=1M, TP=8, CP=4 为例:
  KV_block_size = 2 × 1 × 128 × 250000 × 2 = 128 MB
  total_comm = 3 × 128 MB = 384 MB

通信时间 (NVLink 450 GB/s):
  384 MB / 450 GB/s ≈ 0.85 ms (可以被计算隐藏)

通信时间 (IB NDR 50 GB/s):
  384 MB / 50 GB/s ≈ 7.68 ms
```

### 6.3 计算与通信的 Overlap 条件

```
Ring Attention 通信隐藏条件:

计算时间 per step: T_comp = O((N/cp)² × d × num_heads) / GPU_FLOPS
通信时间 per step: T_comm = KV_block_size / bandwidth

需要: T_comp ≥ T_comm

以 Llama-3.1-70B, 1M context, TP=8, CP=4 为例:
  N/cp = 250K tokens
  单层 attention 计算量 ≈ 2 × 250K × 250K × 128 × 1 ≈ 16.4 TFLOPS
  H100 实际吞吐 (bf16): ~500 TFLOPS
  T_comp ≈ 16.4T / 500T = 32.8 ms per layer

  T_comm (NVLink) ≈ 0.85 ms ≪ T_comp → 完全隐藏!
  T_comm (IB) ≈ 7.68 ms < T_comp → 可以隐藏!

结论: 长上下文场景下, Ring Attention 的通信可以被计算有效隐藏
      这是 CP 在长上下文推理中效率很高的根本原因
```

## 7. CP 的适用场景分析

### 7.1 适用场景

```
1. 超长 Prefill (>128K tokens):
   - KV Cache 显存需求超出 TP 切分后的单卡容量
   - Attention 计算量大, 可以充分隐藏通信
   - CP 效率高

2. 超长上下文对话 (>64K 累积 tokens):
   - 多轮对话的累积上下文
   - 需要保持完整的 KV Cache
   - CP 分散 KV Cache 存储

3. 长文档分析 (100K+ tokens):
   - 一次性处理大量文档
   - Prefill 阶段 CP 效果好
   - Decode 阶段 CP 的收益取决于 KV Cache 大小
```

### 7.2 不适用场景

```
1. 短序列 (< 32K tokens):
   - KV Cache 可以放入 TP 切分后的单卡
   - CP 的通信开销无法被短序列的计算隐藏
   - 不如单纯增加 TP

2. Decode 密集型场景:
   - 大量短 prompt + 长生成
   - Decode 阶段 CP 效率低 (通信/计算比高)
   - 不如用 DP 增加吞吐

3. 低 batch size:
   - 单请求的计算量不足以隐藏 CP 通信
   - 多请求 batch 才能充分利用 CP
```

### 7.3 CP 的使用决策

```python
def should_use_cp(
    seq_len: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    tp_size: int,
    gpu_memory_GB: float = 80,
    weight_per_gpu_GB: float = 17.5,
) -> dict:
    """决策是否需要 CP"""
    
    available_memory = gpu_memory_GB - weight_per_gpu_GB - 3  # 3 GB overhead
    
    kv_per_gpu = (
        2 * num_layers * (num_kv_heads / tp_size) * head_dim 
        * seq_len * 2 / 1e9  # bf16
    )
    
    if kv_per_gpu <= available_memory * 0.8:  # 80% 阈值
        return {
            "need_cp": False,
            "reason": f"KV Cache ({kv_per_gpu:.1f}GB) fits in available memory ({available_memory:.1f}GB)",
        }
    else:
        min_cp = int(kv_per_gpu / (available_memory * 0.8)) + 1
        # Round up to power of 2
        cp_size = 1
        while cp_size < min_cp:
            cp_size *= 2
        return {
            "need_cp": True,
            "min_cp_size": cp_size,
            "reason": f"KV Cache ({kv_per_gpu:.1f}GB) exceeds available memory ({available_memory:.1f}GB)",
        }

# 示例: Llama-3.1-70B, TP=8
should_use_cp(128_000, 80, 8, 128, 8)
# → need_cp: False (5.12 GB < 59.5 GB)

should_use_cp(1_000_000, 80, 8, 128, 8)
# → need_cp: True, min_cp=2 (40.96 GB > 47.6 GB threshold)
```

## 8. CP 的前沿发展

### 8.1 与 FlashAttention 的集成

```
FlashAttention + Ring Attention:
  - FlashAttention 提供高效的单 GPU attention kernel
  - Ring Attention 提供跨 GPU 的序列并行
  - 两者可以结合: 每个 GPU 内部用 FlashAttention, 
    GPU 间用 Ring pass KV
  
  实现: FlashAttention-3 已原生支持 Ring Attention 模式
```

### 8.2 与 KV Cache 压缩的结合

```
CP + KV Cache 压缩:
  - 压缩减少每 token 的 KV 大小
  - CP 减少每 GPU 的 token 数
  - 两者可以叠加使用

  组合效果:
  原始 KV Cache: 40 GB/GPU (1M context, TP=8)
  + KV 压缩 4x: 10 GB/GPU
  + CP=2:        5 GB/GPU
  或
  + CP=4:        2.5 GB/GPU (无需压缩即可)

  根据压缩对质量的影响选择合适的组合
```

### 8.3 Infinite Context

```
向 Infinite Context 的演进:

1. Ring Attention → 基础序列并行
2. Ring Attention + Chunked Prefill → 动态分块
3. Distributed KV Cache → KV 存储在分布式系统中
4. Streaming Attention → 流式处理无限长序列

Infinite-LLM (2024):
  - 将 KV Cache 管理从本地 GPU 扩展到分布式集群
  - 使用 rFMI (remote Function Memory Interface) 远程访问 KV
  - 支持动态的 KV Cache 分布和迁移
```

## 9. 总结

| 要点 | 说明 |
|------|------|
| 核心问题 | KV Cache 随序列长度线性增长，超出单卡容量 |
| CP 原理 | 沿序列维度切分，每卡只存部分 context 的 KV |
| Ring Attention | 通过 KV ring pass + online softmax 实现 CP |
| 通信隐藏 | 长序列的计算量远大于通信量，可完美 overlap |
| 适用场景 | 超长上下文 (>128K)，Prefill 密集型 |
| 与 TP 的关系 | 正交维度，可组合使用 (TP 切 head，CP 切 sequence) |

---

> **下一节**：[多维并行组合决策](06-hybrid-parallelism.md) —— 如何为实际部署选择最优的并行策略组合
