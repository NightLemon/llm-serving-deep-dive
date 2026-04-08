# 为什么要分离 Prefill 和 Decode？

> 本节从计算特征、硬件利用率、系统设计三个维度，分析 Prefill-Decode 分离架构的动机。

## 1. Prefill 与 Decode 的根本矛盾

### 1.1 两种截然不同的计算模式

LLM 推理分为两个阶段，它们在计算特征上有根本性差异：

| 维度 | Prefill（预填充） | Decode（解码） |
|------|-------------------|----------------|
| **输入** | 整个 prompt，数百到数万 token | 单个 token（上一步生成的） |
| **计算模式** | 大矩阵乘法：$[B \times S, H] \times [H, H]$ | 细长矩阵乘法：$[B, H] \times [H, H]$ |
| **瓶颈** | **Compute-bound**（计算密集） | **Memory-bound**（访存密集） |
| **GPU 计算利用率** | 60-80%（接近理论峰值） | 5-15%（大量计算单元空闲） |
| **显存带宽利用率** | 30-50%（有余量） | 80-95%（接近带宽上限） |
| **耗时占比** | 取决于 prompt 长度，通常一次性完成 | 每个 token 一次前向传播，串行执行 |

#### 为什么 Prefill 是 Compute-bound？

Prefill 阶段需要对 prompt 中所有 token **并行**计算 self-attention 和 FFN。以一个 prompt_length=2048 的请求为例：

```
Self-Attention:
  Q, K, V 投影: [2048, 4096] × [4096, 4096] → GEMM，计算量 ≈ 2 × 2048 × 4096²
  Attention score: [2048, 4096] × [4096, 2048] → 另一个大矩阵乘

FFN:
  up_proj:   [2048, 4096] × [4096, 14336] → 大 GEMM
  gate_proj: [2048, 4096] × [4096, 14336] → 大 GEMM
  down_proj: [2048, 14336] × [14336, 4096] → 大 GEMM
```

这些大矩阵乘法能充分利用 GPU 的 Tensor Core，算术强度（Arithmetic Intensity）高，GPU SM 利用率可以达到 60-80%。

#### 为什么 Decode 是 Memory-bound？

Decode 阶段每次只处理 **1 个新 token**（或 batch 中每个序列各 1 个 token）：

```
Self-Attention:
  Q 投影: [1, 4096] × [4096, 4096] → GEMV (矩阵-向量乘)
  Attention: 需要读取整个 KV Cache [seq_len, head_dim]
  
FFN:
  up_proj:   [1, 4096] × [4096, 14336] → GEMV
  gate_proj: [1, 4096] × [4096, 14336] → GEMV
  down_proj: [1, 14336] × [14336, 4096] → GEMV
```

GEMV 的算术强度极低——每从显存读取一个权重元素，只做一次乘加运算。GPU 的计算单元在等数据从 HBM 搬运过来，大部分时间处于空闲状态。

### 1.2 Roofline 模型分析

用 Roofline 模型可以直观看出两阶段的差异：

```
               Compute        ┌────────────────── Compute Roof (A100: 312 TFLOPS FP16)
               Ceiling        │
                              │           ★ Prefill
Throughput                    │          /   (计算密集, 接近 compute roof)
(TFLOPS)                      │         /
                              │        /
                              │       /
                              │      /
                              │     /  ★ Decode
                              │    /     (访存密集, 远离 compute roof)
                              │   /
               ──────────────┼──/────────── Memory BW Roof (A100: 2 TB/s)
                              │/
                              └──────────────────────
                              Arithmetic Intensity (FLOP/Byte)
```

- **Prefill**：算术强度高（大 batch size），位于 compute-bound 区域
- **Decode**：算术强度低（batch=1 per sequence），位于 memory-bound 区域

### 1.3 量化数据：Arithmetic Intensity 对比

以 Llama-3-70B（FP16）为例：

```
Prefill (prompt_length=2048):
  单层 GEMM 计算量 ≈ 2 × 2048 × 8192 × 8192 = 274.9 GFLOP
  权重读取量 = 8192 × 8192 × 2 bytes = 134.2 MB
  Arithmetic Intensity = 274.9 GFLOP / 134.2 MB ≈ 2048 FLOP/Byte ✓ Compute-bound

Decode (batch_size=1):
  单层 GEMM 计算量 ≈ 2 × 1 × 8192 × 8192 = 134.2 MFLOP
  权重读取量 = 8192 × 8192 × 2 bytes = 134.2 MB
  Arithmetic Intensity = 134.2 MFLOP / 134.2 MB ≈ 1 FLOP/Byte ✗ Memory-bound
```

Prefill 的算术强度比 Decode 高了约 **2000 倍**（等于 prompt_length），这是两个阶段必须区别对待的根本原因。

## 2. 混合执行的问题

### 2.1 传统方案：Continuous Batching

在传统 continuous batching 中，prefill 和 decode 请求在同一 GPU 上交替或混合执行：

```
时间线：
GPU │ P1-prefill │ D1,D2,D3 │ P2-prefill │ D1,D2,D3,D4 │ ...
    └───────────┴──────────┴───────────┴────────────┴───

P = Prefill 请求,  D = Decode 请求
```

这种方案的核心问题：

**问题 1：Prefill 抢占 Decode，导致 TPOT（Time Per Output Token）抖动**

当一个新的长 prompt 到达时，GPU 需要花大量时间做 prefill。在此期间，正在 decode 的请求被阻塞，输出延迟（TPOT）出现尖峰。

```
TPOT (ms)
  80 │         ★              ★
     │        / \            / \       ← Prefill 插入导致 TPOT spike
  40 │       /   \          /   \
     │──────/─────\────────/─────\──── ← 期望的稳定 TPOT
  20 │─────────────────────────────── 
     └──────────────────────────────── 时间
```

**问题 2：两边都达不到最优利用率**

- 执行 prefill 时：GPU 计算利用率高，但显存带宽利用不充分
- 执行 decode 时：显存带宽利用率高，但 GPU 计算利用率低
- 混在一起：两种模式频繁切换，无法针对性优化

**问题 3：资源无法独立扩缩容**

假设你的服务有两种场景：
- 场景 A：大量长 prompt（RAG），需要更多 prefill 算力
- 场景 B：大量对话生成，需要更多 decode 容量

混合部署下，你只能整体扩 GPU 数量，无法针对瓶颈阶段单独扩容。

### 2.2 量化分析：混合 vs 分离的 GPU 利用率

以一个典型的 RAG 场景为例（Llama-3-70B，A100-80GB）：

| 指标 | 混合部署 (8×A100) | 分离部署 (3P+5D ×A100) |
|------|-------------------|------------------------|
| Prefill 吞吐 | 15K tokens/s | 22K tokens/s (+47%) |
| Decode 吞吐 | 800 tokens/s | 1200 tokens/s (+50%) |
| 平均 TTFT | 450ms | 280ms (-38%) |
| P99 TPOT | 85ms | 35ms (-59%) |
| 总 GPU 利用率 | 35-45% | 55-70% |

> 数据来源：DistServe 论文 (Zhong et al., 2024) 的实验结果，具体数值因模型和工作负载而异。

分离部署的优势主要来自：
1. **Prefill 节点**可以用更大的 batch size，充分利用计算单元
2. **Decode 节点**不被 prefill 打断，TPOT 更稳定
3. 两类节点可以使用**不同的优化策略**（如不同的量化精度、不同的 batch size）

## 3. Chunked Prefill：折中方案

### 3.1 基本思想

Chunked Prefill 将长 prompt 切成固定大小的 chunk（如 512 tokens），每个 chunk 与 decode tokens 一起组成一个 batch：

```
传统 Prefill（prompt_length=4096）:
  │◄──── 4096 tokens 一次性处理 ────►│  ← 阻塞所有 decode

Chunked Prefill（chunk_size=512）:
  │chunk1│decode│chunk2│decode│chunk3│decode│chunk4│decode│...│chunk8│decode│
  │ 512  │  D   │ 512  │  D   │ 512  │  D   │ 512  │  D   │   │ 512  │  D   │
```

### 3.2 vLLM 中的实现

```bash
# 启用 chunked prefill
vllm serve meta-llama/Llama-3-8B \
    --enable-chunked-prefill \
    --max-num-batched-tokens 2048
```

vLLM V1 引擎默认启用 chunked prefill。调度器将 prefill tokens 和 decode tokens 混合到同一个 batch 中，使每个 step 的 token 总数接近 `max-num-batched-tokens`。

### 3.3 局限性

虽然 chunked prefill 缓解了 TPOT 抖动问题，但它**并没有解决根本矛盾**：

| 局限 | 说明 |
|------|------|
| **利用率仍然不是最优** | Prefill chunk 和 decode tokens 混合执行，两边的算术强度不同，GPU 无法同时对两者做最优调度 |
| **TTFT 增加** | 长 prompt 需要多轮 chunk 才能完成 prefill，TTFT 比一次性 prefill 更高 |
| **调度复杂** | chunk size 的选择需要在 TTFT 和 TPOT 之间权衡，没有一个通用最优值 |
| **无法独立扩缩容** | 仍然是单一 GPU pool，无法针对 prefill 或 decode 单独加机器 |

### 3.4 Chunked Prefill vs 完全分离：定位不同

```
简单 ◄────────────────────────► 复杂
      Vanilla    Chunked       Disaggregated
      Batching   Prefill       Prefill/Decode
      
      单 GPU     单 GPU         多节点
      TPOT 抖动  TPOT 改善      TPOT 最优
      部署简单   部署简单        部署复杂
      利用率低   利用率中等      利用率最高
```

- **Chunked Prefill** 是单机优化，适合中等负载
- **Disaggregated Serving** 是系统级架构，适合大规模、高吞吐场景

## 4. 完全分离的优势

### 4.1 独立硬件优化

分离后，两类节点可以针对各自的计算特征做不同的优化：

| 优化维度 | Prefill 节点 | Decode 节点 |
|---------|-------------|-------------|
| **GPU 选型** | 计算强的 GPU（如 H100 SXM） | 显存带宽大的 GPU（如 H100 NVL）或更多中端 GPU |
| **量化策略** | FP16/BF16（保证 prefill 质量） | INT8/FP8（减少权重读取量，提升 decode 吞吐） |
| **Batch Size** | 大 batch（充分利用计算单元） | 大 batch（增加并发 decode 序列数） |
| **显存分配** | 较少 KV Cache 空间（prefill 完就传走） | 大量 KV Cache 空间（服务多个 decode 序列） |
| **并行策略** | 高 TP degree（加速单次 prefill） | 低 TP degree（每 GPU 服务更多序列） |

### 4.2 独立扩缩容

```
工作负载变化            混合部署                分离部署
──────────────          ────────                ────────
Prefill 压力增大   →    整体扩容 8 GPU    →    只加 2 Prefill GPU
Decode 压力增大    →    整体扩容 8 GPU    →    只加 3 Decode GPU
                        (浪费资源)              (精确扩容)
```

分离架构允许根据实际瓶颈**精确扩缩容**，成本效率显著提高。

### 4.3 更好的 SLO 保障

Service Level Objective（SLO）通常同时约束 TTFT 和 TPOT：

- **TTFT SLO**：首 token 延迟不超过 X ms
- **TPOT SLO**：每 token 生成时间不超过 Y ms

分离架构下：
- Prefill 节点专注于降低 TTFT → 优化 prefill 调度和 batch size
- Decode 节点专注于稳定 TPOT → 不被 prefill 打断，输出延迟平滑

### 4.4 收益公式

分离是否值得取决于一个简单的不等式：

$$
\text{Disagg Benefit} = \underbrace{G_{prefill} + G_{decode}}_{\text{利用率提升}} - \underbrace{C_{transfer} + C_{complexity}}_{\text{分离开销}}
$$

其中：
- $G_{prefill}$：prefill 节点独立优化带来的吞吐提升
- $G_{decode}$：decode 节点不被打断带来的延迟改善
- $C_{transfer}$：KV Cache 传输的时间和带宽开销
- $C_{complexity}$：系统复杂度增加带来的运维成本

当 prompt 越长、并发越高、工作负载越不均匀时，分离的收益越大。

## 5. 小结

| 要点 | 内容 |
|------|------|
| 根本矛盾 | Prefill 是 compute-bound，Decode 是 memory-bound |
| 混合问题 | 利用率低、TPOT 抖动、无法独立扩缩容 |
| Chunked Prefill | 单机折中方案，缓解但不解决根本问题 |
| 完全分离 | 系统级方案，独立优化、独立扩缩容、SLO 保障更好 |
| 核心代价 | KV Cache 传输开销 + 系统复杂度 |

> **下一节**：[02-architecture.md](02-architecture.md) — 深入分离架构的系统设计，解读 Splitwise 和 DistServe 论文。
