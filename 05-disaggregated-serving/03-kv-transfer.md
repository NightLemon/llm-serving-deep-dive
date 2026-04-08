# KV Transfer 协议

> 本节深入分析 Disaggregated Serving 中最关键的技术挑战——KV Cache 传输，详解 NIXL、P2P NCCL、Mooncake 三种主流方案。

## 1. KV Cache 传输为什么是核心挑战？

分离架构的核心开销是将 prefill 生成的 KV Cache 从 prefill worker 传输到 decode worker。传输延迟直接叠加到 TTFT：

$$
\text{TTFT}_{disagg} = T_{prefill} + T_{transfer} + T_{first\_decode}
$$

KV Cache 的数据量随模型大小和序列长度线性增长：

| 模型 | seq_len=1K | seq_len=4K | seq_len=16K | seq_len=128K |
|------|-----------|-----------|------------|-------------|
| Llama-3-8B (32L, 8 KV heads) | 32 MB | 128 MB | 512 MB | 4 GB |
| Llama-3-70B (80L, 8 KV heads) | 80 MB | 320 MB | 1.28 GB | 10.24 GB |
| Llama-3-405B (126L, 16 KV heads) | 504 MB | 2.02 GB | 8.06 GB | 64.5 GB |

> 计算公式: KV Cache Size = num_layers × 2 × num_kv_heads × head_dim × seq_len × dtype_size

对于 Llama-3-70B + 128K context，需要传输 **10.24 GB** 的 KV Cache。使用不同互联技术的传输时间：

| 互联方式 | 有效带宽 | 传输 10.24 GB 耗时 |
|---------|---------|-------------------|
| PCIe Gen5 x16 | ~50 GB/s | ~205 ms |
| InfiniBand HDR (200 Gbps) | ~23 GB/s | ~445 ms |
| InfiniBand NDR (400 Gbps) | ~46 GB/s | ~223 ms |
| NVLink (A100, 单向) | ~300 GB/s | ~34 ms |
| NVLink (H100, 单向) | ~450 GB/s | ~23 ms |

可以看到，**跨节点传输**是瓶颈所在——即使用 400 Gbps IB，传输一个长序列的 KV Cache 也需要几百毫秒。

## 2. NIXL (NVIDIA Inference Xfer Library)

### 2.1 概述

NIXL 是 NVIDIA 在 2025 年初开源的高性能数据传输库，专为 LLM 推理场景设计。它提供统一的 API 来支持多种传输后端：

- GPU-GPU 直接传输（NVLink, PCIe P2P）
- GPU-CPU 传输（通过 host memory staging）
- 跨节点传输（InfiniBand RDMA, RoCE, TCP fallback）

### 2.2 架构设计

```
                     ┌─────────────────────┐
                     │    NIXL User API     │
                     │  (nixl_xfer_desc_t)  │
                     └──────────┬──────────┘
                                │
                     ┌──────────▼──────────┐
                     │    Transfer Engine   │
                     │                      │
                     │  ┌────┐ ┌────┐ ┌───┐│
                     │  │RDMA│ │UCX │ │SHM││
                     │  │    │ │    │ │   ││
                     │  └────┘ └────┘ └───┘│
                     │  ┌────┐ ┌────┐      │
                     │  │GDR │ │TCP │      │
                     │  │Copy│ │    │      │
                     │  └────┘ └────┘      │
                     └─────────────────────┘
```

NIXL 的核心设计理念是**零拷贝传输**——尽量避免数据经过 CPU 内存中转，直接在 GPU 显存之间传输。关键特性：

- **GPUDirect RDMA (GDR)**：绕过 CPU，NIC 直接读写 GPU 显存
- **异步传输**：非阻塞 API，传输可以与计算重叠
- **分段传输 (scatter-gather)**：支持非连续内存布局的 KV Cache 传输
- **连接池管理**：自动管理和复用 RDMA 连接

### 2.3 核心 API

```c
// 创建传输描述符
nixl_status_t nixl_create_xfer_desc(
    nixl_xfer_desc_t *desc,
    nixl_mem_type_t src_type,      // NIXL_MEM_GPU, NIXL_MEM_HOST, etc.
    nixl_mem_type_t dst_type,
    void *src_addr,                 // 源地址（GPU 显存地址）
    void *dst_addr,                 // 目标地址
    size_t size,                    // 传输大小
    int src_dev,                    // 源 GPU ID
    int dst_dev                     // 目标 GPU ID
);

// 发起异步传输
nixl_status_t nixl_xfer_submit(
    nixl_xfer_desc_t *desc,
    nixl_xfer_handle_t *handle      // 返回句柄，用于查询完成状态
);

// 检查传输是否完成
nixl_status_t nixl_xfer_test(
    nixl_xfer_handle_t handle,
    int *completed                  // 1 = 完成, 0 = 进行中
);
```

### 2.4 vLLM NixlConnector 源码分析

vLLM 通过 `NixlConnector` 集成 NIXL。核心源码位于 `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py`：

```python
class NixlConnector(KVConnectorBase_V1):
    """基于 NIXL 的 KV Cache 传输 connector"""
    
    def __init__(self, config: KVTransferConfig, ...):
        # 初始化 NIXL agent
        self.nixl_agent = nixl.nixlAgent(self.engine_id)
        
        # 注册 KV Cache 显存区域（用于 RDMA 传输）
        self.kv_caches_base_addr = []  # 各层 KV Cache 的基地址
        
    def register_kv_caches(self, kv_caches: dict):
        """注册 KV Cache 内存区域到 NIXL，使其可被远端直接访问"""
        for layer_name, (k_cache, v_cache) in kv_caches.items():
            # 向 NIXL 注册 GPU 显存区域
            k_addr = k_cache.data_ptr()
            v_addr = v_cache.data_ptr()
            self.nixl_agent.register_memory(k_addr, k_cache.nbytes, "GPU")
            self.nixl_agent.register_memory(v_addr, v_cache.nbytes, "GPU")
    
    def send_kv_caches(self, request_id: str, ...):
        """Prefill 端: 将 KV Cache 发送到 decode worker"""
        # 构造传输描述符列表（每层 K 和 V 各一个）
        xfer_descs = []
        for layer_idx in range(self.num_layers):
            # 获取源地址（本地 KV Cache block）
            src_k_addr = self._get_block_addr(layer_idx, "key", block_ids)
            src_v_addr = self._get_block_addr(layer_idx, "value", block_ids)
            
            # 获取目标地址（远端 decode worker 的 KV Cache block）
            dst_k_addr = remote_block_addrs[layer_idx]["key"]
            dst_v_addr = remote_block_addrs[layer_idx]["value"]
            
            xfer_descs.append((src_k_addr, dst_k_addr, block_size))
            xfer_descs.append((src_v_addr, dst_v_addr, block_size))
        
        # 提交批量异步传输
        handle = self.nixl_agent.submit_xfer(xfer_descs)
        return handle
    
    def recv_kv_caches(self, request_id: str, ...):
        """Decode 端: 等待 KV Cache 传输完成"""
        # RDMA 单边操作: 发送端直接写入接收端显存
        # 接收端只需等待通知
        self._wait_for_completion(request_id)
```

**关键实现细节：**

1. **Block 级传输**：KV Cache 以 block 为粒度传输，与 PagedAttention 的 block 对齐
2. **RDMA 单边写**：发送端直接写入接收端的 GPU 显存，接收端 CPU 几乎无开销
3. **Metadata 交换**：在数据传输之前，先交换 block table 映射关系

### 2.5 NIXL 性能数据

基于 NVIDIA 公开的 benchmark 数据（H100 集群）：

| 场景 | 传输大小 | 延迟 (P50) | 有效带宽 |
|------|---------|-----------|---------|
| 同机 GPU-GPU (NVLink) | 256 MB | 0.6 ms | ~427 GB/s |
| 同机 GPU-GPU (NVLink) | 1 GB | 2.2 ms | ~454 GB/s |
| 跨机 GPU-GPU (IB NDR) | 256 MB | 6.1 ms | ~42 GB/s |
| 跨机 GPU-GPU (IB NDR) | 1 GB | 23.5 ms | ~43 GB/s |

## 3. P2P NCCL

### 3.1 概述

P2P NCCL 利用 NVIDIA NCCL 库的点对点通信原语（`ncclSend`/`ncclRecv`）在 GPU 之间传输 KV Cache。它的优势是实现简单且与 PyTorch 的 distributed 模块深度集成。

### 3.2 工作原理

```python
import torch.distributed as dist

# Prefill 端发送 KV Cache
def send_kv_cache(kv_tensor: torch.Tensor, dst_rank: int):
    """使用 NCCL P2P 发送 KV Cache"""
    dist.send(kv_tensor, dst=dst_rank)

# Decode 端接收 KV Cache
def recv_kv_cache(kv_tensor: torch.Tensor, src_rank: int):
    """使用 NCCL P2P 接收 KV Cache"""
    dist.recv(kv_tensor, src=src_rank)

# 异步版本
def async_send_recv(send_tensor, recv_tensor, dst_rank, src_rank):
    """异步发送和接收，可与计算重叠"""
    send_op = dist.isend(send_tensor, dst=dst_rank)
    recv_op = dist.irecv(recv_tensor, src=src_rank)
    return send_op, recv_op
```

### 3.3 vLLM P2P NCCL Connector

位于 `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py`：

```python
class P2pNcclConnector(KVConnectorBase_V1):
    """基于 NCCL P2P 的 KV Cache 传输"""
    
    def __init__(self, config: KVTransferConfig, ...):
        # 初始化 NCCL 通信组
        # Prefill worker 和 Decode worker 需要在同一个 NCCL group 中
        self.kv_transfer_group = dist.new_group(
            ranks=[prefill_rank, decode_rank],
            backend="nccl"
        )
    
    def send_kv_caches(self, ...):
        """逐层发送 KV Cache"""
        for layer_idx in range(self.num_layers):
            k_cache = self.get_layer_kv(layer_idx, "key")
            v_cache = self.get_layer_kv(layer_idx, "value")
            
            dist.send(k_cache, dst=self.decode_rank, 
                     group=self.kv_transfer_group)
            dist.send(v_cache, dst=self.decode_rank,
                     group=self.kv_transfer_group)
```

### 3.4 适用场景与局限

**优势：**
- 实现简单，与 PyTorch 生态无缝集成
- 同机 NVLink 场景下带宽极高（450+ GB/s on H100）
- 不需要额外的传输库依赖

**局限：**
- NCCL 的 P2P 操作需要 prefill 和 decode worker 在同一个 NCCL group 中
- 跨节点时依赖 NCCL 的 socket/IB 后端，性能不如专用 RDMA 方案
- NCCL 的集合通信语义对 KV Cache 的 scatter-gather 传输不太友好
- 不支持 GPU→CPU 或 GPU→SSD 的异构传输路径

## 4. Mooncake

### 4.1 概述

Mooncake 是月之暗面（Moonshot AI）开源的 KV Cache 分离架构方案，核心特色是 **KVCache-centric** 的设计理念——将 KV Cache 视为一等公民，围绕 KV Cache 的存储和传输来组织整个系统。

> 论文: "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving" (Qin et al., 2024)

### 4.2 架构特点

```
┌────────────────────────────────────────────────────────┐
│                    Mooncake Architecture                │
│                                                         │
│  ┌──────────┐     ┌──────────────────┐     ┌─────────┐ │
│  │ Prefill  │     │  KVCache Pool    │     │ Decode  │ │
│  │ Instance │────►│                  │────►│Instance │ │
│  │          │     │  ┌────────────┐  │     │         │ │
│  └──────────┘     │  │ GPU VRAM   │  │     └─────────┘ │
│                   │  ├────────────┤  │                  │
│                   │  │ CPU DRAM   │  │  ← 多级缓存     │
│                   │  ├────────────┤  │                  │
│                   │  │ SSD/NVMe   │  │                  │
│                   │  └────────────┘  │                  │
│                   └──────────────────┘                  │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │              RDMA Transfer Layer                  │  │
│  │  (GPUDirect RDMA + CPU DRAM RDMA + SSD DMA)      │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

**KV Cache Pool 的多级存储：**

Mooncake 的核心创新是将 KV Cache 从单纯的 GPU 显存扩展到多级存储：

1. **GPU VRAM**：最快，容量最小，存放 hot KV Cache
2. **CPU DRAM**：次快，容量中等，存放 warm KV Cache
3. **SSD/NVMe**：最慢，容量最大，存放 cold KV Cache

### 4.3 RDMA 传输机制

Mooncake 使用 RDMA（Remote Direct Memory Access）实现跨节点 KV Cache 传输：

```
传统传输路径:
  GPU → PCIe → CPU → 网络栈 → NIC → ... → NIC → CPU → PCIe → GPU
  (多次数据拷贝, CPU 参与, 高延迟)

Mooncake RDMA 路径:
  GPU → GPUDirect RDMA → NIC → ... → NIC → GPUDirect RDMA → GPU
  (零拷贝, CPU 不参与, 低延迟)
```

关键技术：

1. **GPUDirect RDMA**：NIC 直接访问 GPU 显存，绕过 CPU
2. **RDMA Verbs API**：使用 ibverbs 的 `ibv_post_send` (RDMA WRITE) 实现单边写
3. **Chunk-based 传输**：将 KV Cache 切分成固定大小的 chunk，支持流水线传输

### 4.4 vLLM Mooncake Connector

vLLM 通过 `MooncakeConnector` 集成 Mooncake 传输能力：

```python
class MooncakeConnector(KVConnectorBase_V1):
    """基于 Mooncake 的 KV Cache 传输"""
    
    def __init__(self, config: KVTransferConfig, ...):
        # 初始化 Mooncake transfer engine
        self.transfer_engine = MooncakeTransferEngine(
            rdma_device=config.rdma_device,    # e.g., "mlx5_0"
            gpu_id=config.gpu_id,
        )
        
    def send_kv_caches(self, request_id, block_ids, dst_node):
        """通过 Mooncake RDMA 发送 KV Cache"""
        for layer_idx in range(self.num_layers):
            k_data = self._gather_blocks(layer_idx, "key", block_ids)
            v_data = self._gather_blocks(layer_idx, "value", block_ids)
            
            # RDMA WRITE 直接写入远端 GPU 显存
            self.transfer_engine.rdma_write(
                remote_addr=dst_node.get_kv_addr(layer_idx),
                local_buf=k_data,
                size=k_data.nbytes,
            )
            self.transfer_engine.rdma_write(
                remote_addr=dst_node.get_kv_addr(layer_idx) + k_data.nbytes,
                local_buf=v_data,
                size=v_data.nbytes,
            )
```

### 4.5 Mooncake 的独特优势

1. **Prefix Cache 复用**：多个请求共享相同 prefix 的 KV Cache，只传输增量部分
2. **多级缓存**：Hot KV Cache 在 GPU，Warm 在 CPU，Cold 在 SSD，降低显存压力
3. **跨节点 Cache 共享**：不同节点可以通过 RDMA 直接读取其他节点 CPU/GPU 上的 KV Cache

## 5. 其他方案：LMCache

### 5.1 概述

LMCache 是另一个 KV Cache 管理和传输方案，专注于 KV Cache 的存储和复用。vLLM 也提供了 `LMCacheConnector`。

```python
class LMCacheConnector(KVConnectorBase_V1):
    """基于 LMCache 的 KV Cache 存储和传输"""
    
    # LMCache 支持:
    # 1. 本地 GPU/CPU 缓存
    # 2. 分布式缓存（类似 Redis 的 KV Cache 存储）
    # 3. 持久化到磁盘
```

LMCache 更侧重于 KV Cache 的**缓存和复用**（类似 prefix caching 的分布式版本），而非纯粹的传输性能。

## 6. 详细对比

### 6.1 功能对比

| 特性 | NIXL | P2P NCCL | Mooncake | LMCache |
|------|------|----------|----------|---------|
| **同机 GPU-GPU** | ✅ (NVLink/PCIe) | ✅ (NVLink) | ✅ | ✅ |
| **跨机 GPU-GPU** | ✅ (IB RDMA) | ✅ (NCCL Socket/IB) | ✅ (RDMA) | ✅ |
| **GPU-CPU 传输** | ✅ | ❌ | ✅ | ✅ |
| **GPU-SSD 传输** | ❌ | ❌ | ✅ | ✅ |
| **GPUDirect RDMA** | ✅ | 取决于 NCCL 后端 | ✅ | ❌ |
| **零拷贝传输** | ✅ | 部分支持 | ✅ | ❌ |
| **流水线传输** | ✅ | 需手动实现 | ✅ | ❌ |
| **多级缓存** | ❌ | ❌ | ✅ | ✅ |
| **Prefix Cache 复用** | ❌ | ❌ | ✅ | ✅ |
| **散射-聚集 (scatter-gather)** | ✅ | ❌ | ✅ | ❌ |
| **异步 API** | ✅ | ✅ | ✅ | ✅ |

### 6.2 性能对比

| 方案 | 同机带宽 | 跨机带宽 | 同机延迟 (1GB) | 跨机延迟 (1GB) |
|------|---------|---------|---------------|---------------|
| **NIXL** (H100 NVLink) | ~450 GB/s | ~43 GB/s (NDR IB) | ~2.2 ms | ~23 ms |
| **P2P NCCL** (H100 NVLink) | ~420 GB/s | ~35 GB/s (NCCL over IB) | ~2.4 ms | ~28 ms |
| **Mooncake** (RDMA) | ~380 GB/s | ~40 GB/s (GDR) | ~2.6 ms | ~25 ms |
| **LMCache** (CPU staging) | ~25 GB/s | ~15 GB/s | ~40 ms | ~67 ms |

> 注: 性能数据为典型值，实际性能取决于硬件配置、网络拓扑、CUDA 版本等。

### 6.3 适用场景

| 方案 | 最佳适用场景 | 不适用场景 |
|------|------------|-----------|
| **NIXL** | 通用场景，NVIDIA GPU 集群，需要最高传输性能 | 非 NVIDIA 硬件 |
| **P2P NCCL** | 同机多卡分离，已有 NCCL 环境，快速原型 | 大规模跨节点部署 |
| **Mooncake** | 跨节点 RDMA，需要多级缓存和 prefix 复用 | 无 RDMA 支持的网络 |
| **LMCache** | KV Cache 复用为主，不要求极致传输性能 | 对传输延迟极度敏感 |

### 6.4 部署复杂度

```
简单 ◄─────────────────────────────────────────► 复杂

  P2P NCCL          NIXL           LMCache        Mooncake
  
  只需 PyTorch      需装 NIXL 库    需 LMCache      需 RDMA 网卡
  + NCCL            + RDMA 驱动     server          + IB 驱动
                    (如有 IB)       + 分布式部署      + GPUDirect
                                                    + Mooncake lib
```

## 7. 传输优化技术

### 7.1 流水线传输 (Pipelined Transfer)

不等所有层的 KV Cache 都计算完再传输，而是逐层计算逐层传输：

```python
# 伪代码: 流水线传输
async def pipelined_prefill_and_transfer(model, prompt, dst_worker):
    transfer_handles = []
    
    for layer_idx, layer in enumerate(model.layers):
        # 计算当前层的 KV Cache
        k, v = layer.compute_kv(prompt if layer_idx == 0 else hidden_states)
        
        # 立即开始传输当前层（异步）
        handle = kv_connector.async_send(k, v, layer_idx, dst_worker)
        transfer_handles.append(handle)
        
        # 继续计算下一层（与传输并行）
        hidden_states = layer.forward(hidden_states, k, v)
    
    # 等待所有传输完成
    for handle in transfer_handles:
        await handle.wait()
```

节省的时间：

```
无流水线:  T_total = T_prefill + T_transfer(all_layers)
有流水线:  T_total = T_prefill + T_transfer(last_few_layers)
                   ≈ T_prefill + T_transfer(1_layer) × pipeline_depth
```

### 7.2 KV Cache 压缩传输

在传输前对 KV Cache 做量化压缩，减少传输数据量：

```python
# FP16 → INT4 压缩，数据量减少 75%
def compress_kv_cache(kv_fp16: torch.Tensor) -> tuple:
    """将 FP16 KV Cache 量化到 INT4"""
    scale = kv_fp16.abs().max(dim=-1, keepdim=True).values / 7.0
    kv_int4 = (kv_fp16 / scale).round().clamp(-8, 7).to(torch.int8)
    # 两个 INT4 打包到一个 INT8
    return kv_int4, scale

# 传输压缩后的数据（只有原来的 25%）
compressed_kv, scale = compress_kv_cache(kv_cache)
transfer(compressed_kv, scale)

# Decode 端解压
kv_fp16 = decompress_kv_cache(compressed_kv, scale)
```

**压缩传输的 trade-off：**
- 优势：传输数据量减少 2-4x，对网络带宽要求降低
- 代价：压缩/解压需要额外的 GPU 计算时间 + 精度损失

### 7.3 选择性传输

并非所有层的 KV Cache 都需要完整精度传输。研究表明，底层的 KV Cache 对精度更敏感，高层的可以做更激进的压缩：

```
Layer 0-20:   FP16 传输 (全精度, 底层重要)
Layer 20-60:  FP8 传输  (轻量压缩)
Layer 60-80:  INT4 传输 (激进压缩, 高层冗余度高)
```

## 8. 实际部署建议

### 8.1 同机多卡场景

```
推荐方案: NIXL 或 P2P NCCL
理由: NVLink 带宽足够高，传输开销可忽略
配置:
  - vLLM: --kv-transfer-config '{"kv_connector": "NixlConnector"}'
  - 或使用 P2P NCCL: --kv-transfer-config '{"kv_connector": "P2pNcclConnector"}'
```

### 8.2 跨节点 InfiniBand 场景

```
推荐方案: NIXL (GPUDirect RDMA) 或 Mooncake
理由: GPUDirect RDMA 避免 CPU 中转，延迟最低
配置:
  - 确保 MOFED 驱动安装正确
  - 确保 nvidia-peermem 模块加载
  - NIXL: --kv-transfer-config '{"kv_connector": "NixlConnector"}'
```

### 8.3 无 RDMA 的跨节点场景

```
推荐方案: P2P NCCL (TCP) 或 LMCache
理由: 无需 RDMA 硬件支持
注意: 性能会显著低于 RDMA 方案，需评估是否仍然划算
```

## 9. 小结

| 要点 | 内容 |
|------|------|
| 核心挑战 | KV Cache 数据量大（GB 级），传输延迟直接加到 TTFT |
| NIXL | NVIDIA 官方方案，通用性强，性能最优 |
| P2P NCCL | 最简单，适合同机多卡快速验证 |
| Mooncake | KV-centric 设计，多级缓存 + RDMA，适合大规模部署 |
| 关键优化 | 流水线传输、压缩传输、选择性精度 |

> **下一节**：[04-vllm-disagg.md](04-vllm-disagg.md) — vLLM Disaggregated Prefill 完整源码分析。
