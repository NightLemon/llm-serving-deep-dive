# PagedAttention 原理深入

> 本节精读 PagedAttention 论文（SOSP 2023），从操作系统虚拟内存类比出发，深入理解分页式 KV Cache 管理的设计动机、核心机制与 kernel 实现。

## 1. 问题背景：LLM 推理的显存危机

### 1.1 传统 KV Cache 分配的浪费

在 PagedAttention 出现之前，主流推理框架（FasterTransformer、HuggingFace TGI 早期版本）采用**连续内存预分配**策略：

```
// 传统方案：为每个 request 预分配 max_seq_len 的连续显存
kv_cache[request_id] = allocate(max_seq_len * num_layers * 2 * num_kv_heads * head_dim * dtype_size)
```

这带来三类严重浪费：

| 浪费类型 | 说明 | 典型占比 |
|----------|------|----------|
| **预留浪费 (Reserved)** | 预分配 `max_seq_len` 但实际只用了一部分 | 最高可达序列长度比例 |
| **内部碎片 (Internal)** | 分配粒度与实际使用不匹配 | 取决于对齐要求 |
| **外部碎片 (External)** | 已释放的小块无法合并为大块 | 随运行时间累积 |

PagedAttention 论文的关键数据：在 OPT-13B 模型上，传统方案的显存浪费率高达 **60-80%**。这意味着一台 A100-80GB 的 GPU，实际用于存储有效 KV Cache 的显存可能只有 16-32GB。

### 1.2 为什么这是一个操作系统问题？

LLM 推理的 KV Cache 管理与操作系统的内存管理面临惊人相似的挑战：

| 特征 | 操作系统进程内存 | LLM 请求 KV Cache |
|------|-----------------|-------------------|
| **动态增长** | 进程 heap 按需增长 | 每 decode 一个 token，KV 增长一行 |
| **不可预知长度** | 进程运行时间不确定 | 生成长度取决于模型和 prompt |
| **多租户共享** | 多进程共享物理内存 | 多请求共享 GPU 显存 |
| **生命周期各异** | 进程启停时间不同 | 请求到达和结束时间不同 |
| **需要隔离** | 进程间内存隔离 | 请求间 KV Cache 隔离 |

操作系统在 1960 年代就通过**虚拟内存 + 分页机制**解决了这些问题。PagedAttention 的核心洞察是：**将同样的分页思想引入 GPU 显存管理**。

## 2. 核心概念：从 OS 到 GPU

### 2.1 概念映射

PagedAttention 建立了一套从操作系统到 LLM 推理的精确类比：

```
┌──────────────────────────────────────────────────────────┐
│                    操作系统                                │
│  Virtual Address Space  ──→  Physical Page Frames        │
│  Page Table             ──→  VA → PA 映射                │
│  Page Size (4KB)        ──→  固定大小的管理粒度            │
│  Swap Space             ──→  磁盘上的后备存储              │
│  Copy-on-Write          ──→  fork() 时延迟复制            │
├──────────────────────────────────────────────────────────┤
│                    PagedAttention                         │
│  Logical KV Cache       ──→  Physical KV Blocks          │
│  Block Table            ──→  逻辑 block → 物理 block 映射 │
│  Block Size (16 tokens) ──→  固定大小的管理粒度            │
│  CPU Memory             ──→  换出目标（Swap）              │
│  Copy-on-Write          ──→  beam search 时共享 KV        │
└──────────────────────────────────────────────────────────┘
```

### 2.2 KV Block 的定义

一个 KV Block 是 PagedAttention 的最小管理单元。每个 block 存储固定数量 token 的 Key 和 Value 向量：

```python
# 单个 KV Block 的逻辑形状
# 对于单个 attention layer、单个 KV head group:
block_shape = (block_size, num_kv_heads, head_dim)
# Key block: (block_size, num_kv_heads, head_dim)
# Value block: (block_size, num_kv_heads, head_dim)

# 实际 GPU 显存中，所有 block 组成一个大的预分配张量:
# key_cache:   (num_blocks, block_size, num_kv_heads, head_dim)  -- 具体 layout 取决于实现
# value_cache: (num_blocks, block_size, num_kv_heads, head_dim)
```

### 2.3 Block Table

Block Table 类似 OS 的 Page Table，记录每个请求的逻辑 block 到物理 block 的映射：

```
Request A (seq_len=42, block_size=16):
  逻辑 block 0 → 物理 block 7    (tokens 0-15)
  逻辑 block 1 → 物理 block 3    (tokens 16-31)
  逻辑 block 2 → 物理 block 12   (tokens 32-41, 还有 6 个空 slot)

Request B (seq_len=10, block_size=16):
  逻辑 block 0 → 物理 block 1    (tokens 0-9, 还有 6 个空 slot)
```

注意：物理 block 不需要连续。这就是 PagedAttention 消除外部碎片的关键——任何空闲的物理 block 都可以被任何请求使用，无需寻找连续空间。

## 3. Block Size 的选择：为什么是 16？

### 3.1 block_size 的权衡

Block size 的选择涉及多个维度的权衡：

| block_size | 内部碎片 | Kernel 效率 | Block Table 大小 | 内存管理开销 |
|-----------|---------|------------|-----------------|------------|
| 1 | 最小 (0%) | 最差 | 极大 | 极高 |
| 8 | 较小 | 较好 | 适中 | 适中 |
| **16** | **适中 (~3%)** | **很好** | **适中** | **适中** |
| 32 | 较大 | 极好 | 较小 | 较低 |
| 128 | 很大 | 极好 | 很小 | 很低 |

### 3.2 16 tokens/block 的来源

vLLM 默认 `block_size=16` 的原因涉及 GPU 硬件特性：

1. **GPU Warp Size**：NVIDIA GPU 的 warp 包含 32 个线程。`block_size=16` 意味着处理一个 block 的 key/value 可以很好地映射到半个 warp 或一个 warp 上（取决于 head_dim）。

2. **内存对齐**：以 FP16 为例，16 个 token 的一个 head 占用 `16 * head_dim * 2 bytes`。对于 `head_dim=128`，这是 4096 bytes = 4KB，恰好对齐到常见的内存页大小。

3. **L2 Cache 友好**：A100 的 L2 cache line 为 128 bytes。`block_size=16` 配合 `head_dim=128`（FP16）的单行数据为 256 bytes = 2 cache lines，访问模式对 L2 友好。

4. **碎片率可控**：假设序列长度均匀分布在 1-2048 之间，平均内部碎片为 `(16-1)/(2*1024) ≈ 0.7%`。即使对于短序列（平均 100 tokens），碎片率也仅约 `7.5/100 = 7.5%`。

### 3.3 可调性

vLLM 允许通过 `--block-size` 参数调整：

```bash
# 默认值
python -m vllm.entrypoints.openai.api_server --block-size 16

# 更大 block 可提升 kernel 效率但增加碎片
python -m vllm.entrypoints.openai.api_server --block-size 32
```

在 vLLM v1 架构中，block_size 还需要与 attention backend 兼容。FlashAttention 和 FlashInfer 各自对 block_size 有偏好的值。

## 4. Copy-on-Write：Beam Search 场景

### 4.1 问题

Beam search 是一种常用的解码策略，维护 `beam_width` 个候选序列。在传统实现中，每个 beam 都需要独立的 KV Cache 副本：

```
Beam Search (beam_width=4):
  beam 0: "The cat sat on the"    → KV Cache 副本 0
  beam 1: "The cat sat on the"    → KV Cache 副本 1 (与 0 完全相同！)
  beam 2: "The cat sat on the"    → KV Cache 副本 2 (与 0 完全相同！)
  beam 3: "The cat sat on the"    → KV Cache 副本 3 (与 0 完全相同！)
```

这意味着显存用量直接乘以 `beam_width`，极度浪费。

### 4.2 Copy-on-Write 解决方案

借鉴 OS 中 `fork()` 的 Copy-on-Write 机制，PagedAttention 允许多个 beam 共享相同的物理 block：

```
步骤 1：所有 beam 共享相同的 block table
  beam 0-3 的逻辑 block 0 → 物理 block 7 (ref_count = 4)
  beam 0-3 的逻辑 block 1 → 物理 block 3 (ref_count = 4)

步骤 2：beam 0 生成 token "mat"，beam 1 生成 token "rug"
  → 需要在最后一个 block 写入不同的 token
  → 触发 Copy-on-Write：
    beam 0 的逻辑 block 2 → 物理 block 12 (复制自 block 3, 写入 "mat")
    beam 1 的逻辑 block 2 → 物理 block 15 (复制自 block 3, 写入 "rug")
    beam 2-3 的逻辑 block 1 → 物理 block 3 (ref_count = 2, 未修改)
```

引用计数（reference count）是实现 Copy-on-Write 的关键：

```python
# 伪代码：Copy-on-Write 逻辑
def append_token(block_table, beam_id, token_kv):
    last_block = block_table[beam_id][-1]
    
    if ref_count[last_block] > 1:
        # 多个 beam 共享此 block，需要 copy
        new_block = allocate_block()
        copy_block(src=last_block, dst=new_block)
        ref_count[last_block] -= 1
        block_table[beam_id][-1] = new_block
        last_block = new_block
    
    # 现在安全地写入
    write_kv(last_block, token_kv)
```

### 4.3 显存节省

对于 beam_width=4 的场景，Copy-on-Write 可以将显存使用量从 4x 降低到接近 1x（在 beam 分叉前），平均节省约 **55%** 的 KV Cache 显存。

## 5. PagedAttention vs 传统方案的全面对比

```
┌─────────────────────────────────────────────────────────────┐
│                传统连续分配 (Contiguous Allocation)           │
│                                                             │
│  Request A: [██████████░░░░░░░░░░░░░░░░░░░░░░]  max=2048   │
│             ↑ used=10  ↑ wasted=2038                        │
│                                                             │
│  Request B: [████████████████████░░░░░░░░░░░░░]  max=2048   │
│             ↑ used=20         ↑ wasted=2028                 │
│                                                             │
│  浪费率: (2038 + 2028) / (2048 * 2) ≈ 99% (极端例子)        │
│  典型浪费率: 60-80%                                          │
├─────────────────────────────────────────────────────────────┤
│                PagedAttention (Paged Allocation)             │
│                                                             │
│  物理 Block 池: [0][1][2][3][4][5][6][7][8][9]...           │
│                                                             │
│  Request A (10 tokens): block 7 (仅 1 个 block)              │
│  Request B (20 tokens): block 1 → block 5 (2 个 block)       │
│  空闲 blocks: 0, 2, 3, 4, 6, 8, 9, ...                      │
│                                                             │
│  浪费: 仅最后一个 block 的内部碎片                             │
│  Request A: 10 tokens / 16 = 0.625 blocks → 1 block 浪费 6  │
│  Request B: 20 tokens / 16 = 1.25 blocks → 2 blocks 浪费 12 │
│  浪费率: (6 + 12) / (16 * 3) ≈ 37% (极短序列的最坏情况)      │
│  典型浪费率: < 4%                                            │
└─────────────────────────────────────────────────────────────┘
```

量化对比（基于论文实验数据，OPT-13B，A100-40GB）：

| 指标 | 传统方案 (FasterTransformer) | PagedAttention (vLLM) |
|------|---------------------------|----------------------|
| 最大并发请求数 | ~16 | ~48 |
| 吞吐量 (req/s) | 基准 | **2-4x** |
| 显存利用率 | 20-40% | **>96%** |
| Beam search (k=4) 显存 | 4x KV Cache | ~1.5x KV Cache |

## 6. PagedAttention Kernel 实现原理

### 6.1 核心挑战

传统的 attention kernel（如 FlashAttention）假设 KV Cache 在内存中是连续的。PagedAttention 需要一个能处理**非连续内存**的 attention kernel。

### 6.2 Kernel 伪代码

```cuda
// PagedAttention kernel 的简化逻辑
// 输入: query (当前 token), block_table, key_cache, value_cache
// 输出: attention output

__global__ void paged_attention_kernel(
    float* output,           // [num_heads, head_dim]
    const float* query,      // [num_heads, head_dim]
    const float* key_cache,  // [num_blocks, block_size, num_kv_heads, head_dim]
    const float* value_cache,// [num_blocks, block_size, num_kv_heads, head_dim]
    const int* block_table,  // [max_num_blocks_per_seq]
    int context_len,
    int block_size
) {
    int head_idx = blockIdx.x;   // 每个 CUDA block 处理一个 attention head
    int thread_idx = threadIdx.x;
    
    float qk_max = -INFINITY;
    
    // Phase 1: 计算 QK^T（遍历所有物理 block）
    int num_blocks = (context_len + block_size - 1) / block_size;
    for (int b = 0; b < num_blocks; b++) {
        int physical_block = block_table[b];  // 关键：通过 block table 查找物理位置
        
        int tokens_in_block = min(block_size, context_len - b * block_size);
        for (int t = 0; t < tokens_in_block; t++) {
            // 从非连续的物理位置读取 key
            float* key = &key_cache[physical_block * block_size * num_kv_heads * head_dim
                                    + t * num_kv_heads * head_dim
                                    + head_idx * head_dim];
            
            float qk = dot_product(query + head_idx * head_dim, key, head_dim);
            qk_max = max(qk_max, qk);
            // 存储 qk 到 shared memory
        }
    }
    
    // Phase 2: Softmax（需要 online softmax 以避免数值溢出）
    // ... safe softmax with running max ...
    
    // Phase 3: 加权求和 V
    for (int b = 0; b < num_blocks; b++) {
        int physical_block = block_table[b];
        // 类似地从非连续位置读取 value 并累加
    }
}
```

### 6.3 PagedAttention V1 vs V2

论文和 vLLM 实现中有两个版本的 kernel：

**PagedAttention V1**：
- 每个 CUDA thread block 处理一个 query head 的完整 attention
- 适合 context length 较短的场景
- 简单但并行度受限

**PagedAttention V2**：
- 将 KV blocks 的处理**分割**到多个 CUDA thread block
- 使用 **reduce** 步骤合并各 thread block 的部分结果
- 类似 FlashDecoding 的思想：在 sequence 维度上并行
- 适合长 context 场景，GPU 利用率更高

```
PagedAttention V1:
  Thread Block 0 → Head 0, 遍历所有 KV blocks
  Thread Block 1 → Head 1, 遍历所有 KV blocks
  ...

PagedAttention V2:
  Thread Block 0 → Head 0, KV blocks 0-3    ─┐
  Thread Block 1 → Head 0, KV blocks 4-7    ─┼→ Reduce → Head 0 output
  Thread Block 2 → Head 0, KV blocks 8-11   ─┘
  Thread Block 3 → Head 1, KV blocks 0-3    ─┐
  Thread Block 4 → Head 1, KV blocks 4-7    ─┼→ Reduce → Head 1 output
  ...
```

### 6.4 与 FlashAttention / FlashInfer 的关系

在现代 vLLM（v0.6+）中，PagedAttention kernel 已经不是唯一选择：

| Backend | 特点 | 适用场景 |
|---------|------|---------|
| **PagedAttention V1/V2** | vLLM 原生 kernel | 兼容性最好，作为 fallback |
| **FlashAttention** | 需要适配 paged layout | Prefill 阶段性能最优 |
| **FlashInfer** | 原生支持 paged/ragged layout | Decode 阶段可利用 cascade attention |
| **FlashDecoding** | sequence 维度并行 | 长 context decode |

vLLM v1 架构中，`--attention-backend` 参数控制使用哪个 backend。实际上，FlashInfer 已经在很多场景下替代了原始的 PagedAttention kernel，因为 FlashInfer 从设计之初就支持 paged KV cache 的非连续内存访问模式。

## 7. 论文精读要点

### 7.1 论文信息

- **标题**: Efficient Memory Management for Large Language Model Serving with PagedAttention
- **会议**: SOSP 2023 (ACM Symposium on Operating Systems Principles)
- **作者**: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica (UC Berkeley)
- **意义**: 这是一篇发表在顶级系统会议上的论文，体现了 LLM 推理是一个**系统问题**而非纯 ML 问题

### 7.2 核心贡献

1. **识别了 KV Cache 管理是 LLM serving 吞吐量的核心瓶颈**——不是计算，而是内存管理
2. **提出 PagedAttention 算法**——将 OS 虚拟内存技术引入 GPU 显存管理
3. **实现 vLLM 系统**——包含完整的分页管理、调度、和优化

### 7.3 实验结果亮点

- 在相同硬件上，vLLM 的吞吐量比 FasterTransformer 高 **2-4x**，比 Orca 高 **2-3x**
- 在 beam search 场景下，由于 Copy-on-Write，优势更加明显
- 复杂的 sampling 方法（parallel sampling, beam search）中，内存节省高达 **55%**

### 7.4 局限性与后续发展

论文的原始设计有一些局限，在后续版本中得到了改进：

| 局限 | 后续改进 |
|------|---------|
| 不支持 prefix caching | vLLM v0.3+ 引入 Automatic Prefix Caching |
| Kernel 效率不如 FlashAttention | 集成 FlashAttention / FlashInfer backend |
| 单机设计 | vLLM 支持 TP/PP 分布式推理 |
| block_size 固定 | 可配置，且与 backend 协调 |

## 8. 小结

PagedAttention 的核心价值在于**将成熟的操作系统思想迁移到新的应用领域**。它证明了 LLM serving 不仅是一个深度学习问题，更是一个系统工程问题。通过分页管理，PagedAttention 将 KV Cache 的显存利用率从 20-40% 提升到 96% 以上，直接转化为 2-4 倍的吞吐量提升。

理解 PagedAttention 是深入 vLLM 源码的基础。下一节我们将走读 vLLM v1 中 BlockPool、KVCacheManager 等核心组件的具体实现。

---

**延伸阅读：**
- [原始论文 PDF](https://arxiv.org/abs/2309.06180)
- [SOSP 2023 演讲视频](https://www.youtube.com/watch?v=KLFadWdomyI)
- [vLLM 官方博客：vLLM: Easy, Fast, and Cheap LLM Serving](https://blog.vllm.ai/2023/06/20/vllm.html)
