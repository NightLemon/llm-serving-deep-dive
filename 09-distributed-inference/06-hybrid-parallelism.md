# 多维并行组合决策

> 本节提供一个系统化的决策框架，帮助你在实际部署中选择最优的并行策略组合。涵盖小/中/大/MoE/超长上下文等不同场景，并给出各硬件配置下的推荐策略和性能预估方法。

## 1. 并行策略全景

### 1.1 五种并行维度总结

```
并行方式      切分维度         通信模式         适用场景
──────────────────────────────────────────────────────────────
TP            Head/Hidden      AllReduce        节点内, 降延迟
PP            Layer            P2P Send/Recv    跨节点, 大模型
EP            Expert           All-to-All       MoE 模型
DP            Replica          无通信           提升吞吐
CP            Sequence         Ring/AllGather   超长上下文
```

### 1.2 各并行方式的 Trade-off

```
                延迟    吞吐    显存效率    通信开销    实现复杂度
TP              ↓↓      →       ↑          中          低
PP              →/↑     ↑       ↑          低          中
EP              →       ↑       ↑↑         高          高
DP              →       ↑↑      →          无          低
CP              →       →       ↑↑         中          高

↓ = 降低, ↑ = 提高, → = 不变
```

## 2. 决策框架

### 2.1 核心决策流程

```python
def choose_parallelism(
    model_params_B: float,          # 模型参数量 (B)
    model_type: str,                # "dense" or "moe"
    max_seq_len: int,               # 最大上下文长度
    target_tpot_ms: float,          # 目标 TPOT (ms)
    target_qps: float,             # 目标 QPS
    num_gpus: int,                  # 可用 GPU 总数
    gpus_per_node: int,             # 每节点 GPU 数
    gpu_memory_GB: float,           # 单卡显存 (GB)
    has_nvlink: bool,               # 是否有 NVLink
    inter_node_bw_GBs: float,      # 节点间带宽 (GB/s)
) -> dict:
    """并行策略选择决策框架"""
    
    num_nodes = num_gpus // gpus_per_node
    model_size_GB = model_params_B * 2  # bf16
    
    # ===== Step 1: 确定最小 TP =====
    # 模型权重必须能放入 TP 切分后的单卡 (留 40% 给 KV Cache)
    min_tp = 1
    while model_size_GB / min_tp > gpu_memory_GB * 0.6:
        min_tp *= 2
    
    # TP 不超过单节点 GPU 数 (跨节点 TP 通常不推荐)
    tp_size = min(min_tp, gpus_per_node)
    
    # ===== Step 2: 确定是否需要 PP =====
    pp_size = 1
    if model_size_GB / tp_size > gpu_memory_GB * 0.6:
        # 单节点 TP 放不下 → 需要 PP
        while model_size_GB / (tp_size * pp_size) > gpu_memory_GB * 0.6:
            pp_size += 1
    
    # ===== Step 3: MoE 特殊处理 =====
    ep_size = 1
    if model_type == "moe":
        # MoE 模型的 expert 权重额外占用显存
        # EP 可以分散 expert 到不同 GPU
        ep_size = determine_ep_size(model_params_B, tp_size, gpu_memory_GB)
    
    # ===== Step 4: 确定是否需要 CP =====
    cp_size = 1
    kv_cache_per_gpu = estimate_kv_cache(max_seq_len, tp_size)
    available_memory = gpu_memory_GB - model_size_GB / (tp_size * pp_size) - 3
    if kv_cache_per_gpu > available_memory * 0.8:
        while kv_cache_per_gpu / cp_size > available_memory * 0.8:
            cp_size *= 2
    
    # ===== Step 5: 确定 DP =====
    gpus_per_replica = tp_size * pp_size * ep_size * cp_size
    dp_size = num_gpus // gpus_per_replica
    
    # 检查是否满足吞吐要求
    estimated_qps = estimate_qps_per_replica(model_params_B, tp_size) * dp_size
    
    return {
        "tp": tp_size,
        "pp": pp_size,
        "ep": ep_size,
        "cp": cp_size,
        "dp": dp_size,
        "gpus_per_replica": gpus_per_replica,
        "estimated_qps": estimated_qps,
    }
```

### 2.2 决策树图示

```
                            开始
                             │
                     模型是否 MoE?
                    ╱              ╲
                  是                否
                  │                 │
            ┌─────▼─────┐    模型能放入单 GPU?
            │ MoE 路径   │   ╱              ╲
            │ (见 2.3)   │ 是                否
            └───────────┘  │                 │
                          DP=N          模型能放入单节点?
                          TP=1         ╱              ╲
                           │         是                否
                           │          │                 │
                      需要低延迟?   TP=gpus_per_node  TP=gpus_per_node
                     ╱       ╲      DP=剩余 GPU       PP=ceil(需求)
                   是         否                      DP=剩余 GPU
                    │          │
                  TP=2        DP=N
                  DP=N/2      TP=1
                                    
                                    超长上下文 (>128K)?
                                   ╱              ╲
                                 是                否
                                  │                 │
                           上述基础上              不变
                           加 CP=2/4
```

### 2.3 MoE 模型决策路径

```python
def moe_parallelism_decision(
    total_params_B: float,        # 总参数量
    active_params_B: float,       # 激活参数量
    num_experts: int,             # Expert 数量
    num_gpus: int,
    gpus_per_node: int,
    gpu_memory_GB: float,
):
    """MoE 模型并行策略"""
    
    num_nodes = num_gpus // gpus_per_node
    
    # Attention + Shared layers 用 TP
    shared_params_GB = active_params_B * 2  # 近似
    tp_size = 1
    while shared_params_GB / tp_size > gpu_memory_GB * 0.3:
        tp_size *= 2
    tp_size = min(tp_size, gpus_per_node)
    
    # Expert 用 EP
    expert_params_GB = (total_params_B - active_params_B) * 2
    experts_per_gpu_memory = (gpu_memory_GB * 0.5) / (expert_params_GB / num_experts)
    
    if experts_per_gpu_memory >= num_experts:
        # 所有 expert 放得下 → 不需要 EP
        ep_size = 1
    else:
        ep_size = math.ceil(num_experts / experts_per_gpu_memory)
        # 通常 EP 跨节点
        ep_size = min(ep_size, num_nodes)
    
    # 剩余 GPU 用于 DP
    gpus_per_replica = tp_size * ep_size
    dp_size = num_gpus // gpus_per_replica
    
    return {"tp": tp_size, "ep": ep_size, "dp": dp_size}
```

## 3. 场景化配置推荐

### 3.1 小模型 (7B-13B Dense)

```
模型: Llama-3-8B (bf16 ≈ 16 GB 权重)
硬件: 8×H100 80GB (1 节点)

配置 A: 延迟优先
  TP=1, DP=8
  每 replica 1 GPU, 8 个 replica 并行
  TPOT: ~5-8 ms
  QPS: ~200 req/s (短对话)
  
配置 B: 吞吐极致
  TP=1, DP=8 (与 A 相同)
  增加 batch size, 每 replica 处理更多并发
  TPOT: ~15-25 ms
  QPS: ~500 req/s (牺牲延迟换吞吐)

配置 C: 超低延迟
  TP=2, DP=4 (NVLink)
  TPOT: ~3-5 ms
  QPS: ~120 req/s
  适用: 实时语音交互等极低延迟场景

小模型建议: TP=1 + 最大化 DP
```

### 3.2 中模型 (70B Dense)

```
模型: Llama-3-70B (bf16 ≈ 140 GB 权重)
硬件: 8×H100 80GB (1 节点, NVSwitch)

配置 A: 延迟优先
  TP=4, DP=2
  每 replica 4 GPU, 2 个 replica
  每卡权重: 35 GB, KV Cache: ~40 GB
  TPOT: ~8-12 ms
  QPS: ~40 req/s

配置 B: 吞吐优先
  TP=8, DP=1
  每 replica 8 GPU
  每卡权重: 17.5 GB, KV Cache: ~55 GB
  TPOT: ~6-10 ms (单 replica 但 batch 更大)
  QPS: ~25-30 req/s (但支持更多并发, 更长序列)

配置 C: 多节点
  2 节点, 16 GPU
  TP=8, DP=2 (每节点一个 replica)
  QPS: ~50-60 req/s

70B 建议:
  - 单节点: TP=4 + DP=2 (平衡) 或 TP=8 (长序列)
  - 多节点: TP=8 per node + DP across nodes
```

### 3.3 大模型 (405B Dense)

```
模型: Llama-3-405B (bf16 ≈ 810 GB 权重)
硬件: 4×8×H100 80GB (4 节点, IB NDR)

配置 A: 基础部署
  TP=8, PP=2 (2 节点)
  每卡权重: 50.6 GB, KV Cache: ~25 GB
  剩余 2 节点 → 另一个 replica (DP=2)
  TPOT: ~15-25 ms
  QPS: ~15-20 req/s

配置 B: 最大 KV Cache
  TP=8, PP=4 (4 节点)
  每卡权重: 25.3 GB, KV Cache: ~50 GB
  DP=1 (所有 GPU 给一个 replica)
  TPOT: ~10-15 ms (PP pipeline 延迟, 但更大 batch)
  QPS: ~10-15 req/s (单 replica 但支持更长序列)

配置 C: FP8 量化
  TP=8, PP=1 (单节点!)
  FP8 权重: ~405 GB → 每卡 50.6 GB → 单节点可放下
  剩余 3 节点 → DP=4
  TPOT: ~8-12 ms
  QPS: ~60-80 req/s

405B 建议:
  - FP8 量化后单节点 TP=8 + 多节点 DP (最佳方案)
  - 无量化: TP=8 + PP=2 + DP
```

### 3.4 MoE 大模型 (DeepSeek-V3 671B)

```
模型: DeepSeek-V3 (671B, 256 experts, active ~37B)
硬件: 8×8×H100 80GB (8 节点, IB NDR)

配置 A: FP8 部署
  TP=8 (节点内)
  EP=8 (跨 8 节点, 每节点 32 experts)
  DP=1 (64 GPU 全部给一个 replica)
  每卡 expert 权重 (FP8): ~10 GB
  每卡 shared 权重 (FP8): ~5 GB
  每卡 KV Cache: ~60 GB (MLA 压缩后)
  TPOT: ~20-35 ms (受 All-to-All 延迟影响)

配置 B: BF16 部署
  TP=8 (节点内)
  EP=8 (跨节点)
  PP=1-2 (如果显存不够)
  每卡 expert 权重 (BF16): ~21 GB
  每卡 shared 权重 (BF16): ~10 GB
  每卡 KV Cache: ~40 GB

配置 C: 4 节点 FP8 (成本优化)
  TP=8 (节点内)
  EP=4 (跨 4 节点, 每节点 64 experts)
  每卡 expert 权重 (FP8): ~21 GB
  每卡 KV Cache: ~45 GB
  
MoE 建议:
  - TP 用于 attention 和 shared expert (节点内)
  - EP 用于 routed experts (跨节点)
  - FP8 量化大幅减少所需 GPU
  - EPLB 对生产至关重要
```

### 3.5 超长上下文场景

```
模型: Llama-3.1-70B, Context = 1M tokens
硬件: 2×8×H100 80GB (2 节点)

KV Cache 需求: ~41 GB/GPU (TP=8)
权重需求: ~17.5 GB/GPU (TP=8)
总计: ~58.5 GB → 单节点 TP=8 可能不够

配置:
  TP=8 (节点内)
  CP=2 (跨 2 节点)
  KV Cache/GPU: ~20.5 GB (CP=2 后减半)
  权重/GPU: ~17.5 GB
  总计: ~38 GB → 可行!

替代方案:
  TP=8, PP=2 (跨节点)
  但每卡仍需存 80 层的一半 = 40 层的 KV Cache
  KV Cache/GPU: 40/80 × 41 GB / 1 = 20.5 GB → 可行
  
CP vs PP 选择:
  CP: Prefill 更快 (Ring Attention overlap)
  PP: Decode 更简单 (无需跨节点 KV access)
  混合: CP for prefill, PP for decode (复杂但最优)
```

## 4. 各硬件配置下的最优策略

### 4.1 硬件配置速查表

```
GPU 型号      显存    HBM 带宽    NVLink     FP16 TFLOPS
─────────────────────────────────────────────────────────
A100 80GB     80 GB   2.0 TB/s    600 GB/s   312
H100 80GB     80 GB   3.35 TB/s   900 GB/s   990
H200 141GB    141 GB  4.8 TB/s    900 GB/s   990
B200 192GB    192 GB  8.0 TB/s    1800 GB/s  2250 (est.)
```

### 4.2 A100 推荐配置

```
模型        GPU 数    TP    PP    EP    DP    备注
──────────────────────────────────────────────────────────
8B          1-8       1     1     -     1-8   DP 最大化
70B (bf16)  8         8     1     -     1     NVSwitch 必须
70B (int8)  4-8       4     1     -     1-2   量化后 TP=4 足够
405B        32        8     4     -     1     4 节点
Mixtral     8         4-8   1     1-2   1     小型 MoE
```

### 4.3 H100 推荐配置

```
模型           GPU 数    TP    PP    EP    DP    备注
──────────────────────────────────────────────────────────
8B             1-8       1     1     -     1-8   单卡足够
70B (bf16)     4-8       4-8   1     -     1-2   TP=4 延迟最优
70B (FP8)      2-8       2     1     -     1-4   FP8 只需 TP=2
405B (bf16)    16        8     2     -     1     2 节点
405B (FP8)     8-16      8     1     -     1-2   FP8 单节点可放
DS-V3 (FP8)    32-64     8     1     4-8   1     4-8 节点 EP
```

### 4.4 H200/B200 推荐配置

```
H200 (141 GB 显存) 优势:
  - 70B bf16 权重 140GB ≈ 1 节点刚好放下
  - TP=8: 每卡 17.5 GB 权重 → 120+ GB 给 KV Cache!
  - 可以支持更长上下文和更大 batch
  
B200 (192 GB 显存) 优势:
  - 405B bf16: TP=8 每卡 ~101 GB → 单节点可放!
  - 无需 PP, 简化部署
  - ~90 GB/卡 给 KV Cache → 超长上下文无压力
  - NVLink 5 带宽翻倍 → TP 通信开销更低
```

## 5. 实际案例分析

### 5.1 案例: 在线对话服务

```
场景:
  模型: Llama-3-70B (bf16)
  上下文长度: 8K (平均)
  目标: TPOT < 30ms, QPS > 100
  硬件: 4 × DGX H100 (32 GPU)

分析:
  70B bf16 = 140 GB → 需要 TP≥2
  TPOT < 30ms → TP=4 足够 (实测 ~10ms)
  
  TP=4 → 每 replica 4 GPU
  32 / 4 = 8 replicas (DP=8)
  
  每 replica QPS ≈ 15 req/s
  总 QPS ≈ 120 req/s → 满足!

最终配置:
  TP=4, PP=1, DP=8
  每节点 2 个 replica
  使用 Cache-Aware 路由 (system prompt 亲和)
```

### 5.2 案例: 代码补全服务 (低延迟)

```
场景:
  模型: DeepSeek-Coder-V2-33B
  上下文长度: 16K (平均)
  目标: TPOT < 15ms (打字速度), QPS > 500
  硬件: 8 × DGX H100 (64 GPU)

分析:
  33B bf16 = 66 GB → TP=2 足够 (33 GB/卡)
  TPOT < 15ms → TP=2 实测 ~7ms → 满足
  
  TP=2 → 每 replica 2 GPU
  64 / 2 = 32 replicas (DP=32)
  
  每 replica QPS ≈ 20 req/s
  总 QPS ≈ 640 → 满足!
  
  或者使用 FP8:
  33B FP8 = 33 GB → TP=1!
  64 replicas (DP=64)
  QPS ≈ 1200+ req/s

最终配置:
  FP8: TP=1, DP=64
  使用 Least Load 路由 (代码补全请求长度差异大)
```

### 5.3 案例: 长文档分析 (高上下文)

```
场景:
  模型: Llama-3.1-70B
  上下文长度: 200K (法律文档)
  目标: TTFT < 30s (一次性 prefill 可以慢), TPOT < 50ms
  硬件: 2 × DGX H100 (16 GPU)

分析:
  200K KV Cache (TP=8): ~10 GB/卡 → 可以放下 (无需 CP)
  权重 (TP=8): 17.5 GB/卡
  总计: 27.5 GB/卡 → 没问题

  但 200K prefill 计算量巨大:
  Prefill 时间 (TP=8): ~10-15 秒 → 可接受
  
  TP=8, DP=2 (每节点一个 replica)
  
  如果上下文增至 500K:
  KV Cache (TP=8): ~25 GB/卡
  总计: 42.5 GB/卡 → 依然可以
  
  如果上下文增至 1M:
  KV Cache (TP=8): ~51 GB/卡
  总计: 68.5 GB/卡 → 需要 CP=2

最终配置:
  200K: TP=8, DP=2
  500K: TP=8, DP=2 (不调大 batch 即可)
  1M:   TP=8, CP=2 (跨 2 节点)
```

### 5.4 案例: DeepSeek-V3 生产部署

```
场景:
  模型: DeepSeek-V3 (671B, MoE)
  上下文长度: 32K (平均)
  目标: TPOT < 50ms, QPS > 50
  硬件: 8 × DGX H100 (64 GPU)

分析:
  FP8 权重: ~335 GB (总), active ~18 GB
  
  Attention + Shared Expert: TP=8 (节点内)
  Routed Experts: 256 experts
  
  FP8 每个 expert: ~2.6 GB
  EP=4: 每节点 64 experts → 64 × 2.6 ≈ 166 GB → 每卡 ~21 GB
  EP=8: 每节点 32 experts → 32 × 2.6 ≈ 83 GB → 每卡 ~10 GB
  
  选 EP=4 (节省节点, 更多显存给 KV Cache):
  每卡: 21 (experts) + 5 (shared, FP8) = 26 GB
  KV Cache: ~50 GB/卡 (MLA 压缩后更多)
  
  4 节点 × 8 GPU = 32 GPU 为一个 replica
  64 / 32 = DP=2

最终配置:
  TP=8, EP=4, DP=2
  EPLB 开启
  Shared Expert 计算与 All-to-All 通信 overlap
```

## 6. 性能预估方法

### 6.1 Decode 延迟预估

```python
def estimate_decode_latency(
    model_params_B: float,
    tp_size: int,
    pp_size: int,
    batch_size: int,
    seq_len: int,
    num_layers: int,
    hidden_size: int,
    dtype_bytes: int = 2,
    hbm_bw_TBs: float = 3.35,    # H100
    nvlink_bw_GBs: float = 450,
    ib_bw_GBs: float = 50,
):
    """Decode 延迟预估 (memory-bound 假设)"""
    
    # 权重读取时间 (memory-bound)
    weight_bytes = model_params_B * 1e9 * dtype_bytes / (tp_size * pp_size)
    weight_read_ms = weight_bytes / (hbm_bw_TBs * 1e12) * 1000
    
    # KV Cache 读取时间
    # 每层: batch_size × seq_len × 2 × num_kv_heads/tp_size × head_dim × dtype_bytes
    kv_read_per_layer = batch_size * seq_len * 2 * hidden_size / tp_size * dtype_bytes / 8
    kv_read_ms = kv_read_per_layer * (num_layers / pp_size) / (hbm_bw_TBs * 1e12) * 1000
    
    # TP 通信时间 (AllReduce)
    tp_comm_per_layer = batch_size * hidden_size * dtype_bytes  # 每次 AllReduce 数据量
    tp_comm_ms = 2 * (num_layers / pp_size) * (tp_comm_per_layer / (nvlink_bw_GBs * 1e9) + 5e-6) * 1000
    
    # PP 通信时间 (P2P)
    pp_comm_ms = 0
    if pp_size > 1:
        pp_msg_size = batch_size * hidden_size * dtype_bytes
        pp_comm_ms = (pp_size - 1) * (pp_msg_size / (ib_bw_GBs * 1e9) + 1e-6) * 1000
    
    total_ms = weight_read_ms + kv_read_ms + tp_comm_ms + pp_comm_ms
    
    return {
        "weight_read_ms": weight_read_ms,
        "kv_read_ms": kv_read_ms,
        "tp_comm_ms": tp_comm_ms,
        "pp_comm_ms": pp_comm_ms,
        "total_ms": total_ms,
    }
```

### 6.2 Prefill 吞吐预估

```python
def estimate_prefill_throughput(
    model_params_B: float,
    tp_size: int,
    prompt_len: int,
    num_layers: int,
    hidden_size: int,
    gpu_tflops: float = 990,     # H100 bf16
    efficiency: float = 0.5,     # MFU
):
    """Prefill 吞吐预估 (compute-bound)"""
    
    # 每 token 的 FLOPs (近似)
    flops_per_token = 2 * model_params_B * 1e9  # 2N FLOPs per token (前向)
    
    # Attention 额外 FLOPs (二次方)
    attn_flops_per_token = 2 * num_layers * prompt_len * hidden_size
    
    total_flops = prompt_len * (flops_per_token + attn_flops_per_token)
    
    # 实际算力
    effective_tflops = gpu_tflops * tp_size * efficiency
    
    prefill_time_ms = total_flops / (effective_tflops * 1e12) * 1000
    
    return {
        "prefill_time_ms": prefill_time_ms,
        "tokens_per_second": prompt_len / (prefill_time_ms / 1000),
    }
```

### 6.3 端到端 QPS 预估

```python
def estimate_max_qps(
    decode_latency_ms: float,     # 从 estimate_decode_latency 获得
    max_batch_size: int,          # 受 KV Cache 限制
    avg_output_tokens: int,       # 平均生成长度
    dp_size: int,
):
    """估算最大 QPS"""
    
    # 每秒生成的 token 数 (per replica)
    tokens_per_second = max_batch_size / (decode_latency_ms / 1000)
    
    # 每秒完成的请求数 (per replica)
    qps_per_replica = tokens_per_second / avg_output_tokens
    
    # 总 QPS
    total_qps = qps_per_replica * dp_size
    
    return {
        "tokens_per_second_per_replica": tokens_per_second,
        "qps_per_replica": qps_per_replica,
        "total_qps": total_qps,
    }
```

## 7. 常见错误与陷阱

```
1. 跨节点 TP:
   错误: 使用 TP=16 跨 2 节点
   后果: AllReduce 延迟暴增, 比 TP=8 + PP=2 慢 3-5x
   正确: 节点内 TP, 节点间 PP/EP

2. 过度 TP:
   错误: 小模型 (8B) 用 TP=8
   后果: 每卡计算量太小, TP 通信占比过高
   正确: TP=1-2 + DP

3. 忽略 KV Cache 显存:
   错误: 只看权重大小决定 TP
   后果: 运行时 KV Cache 不够用, 被迫缩小 batch
   正确: 预留 40-60% 显存给 KV Cache

4. MoE 用纯 TP:
   错误: DeepSeek-V3 用 TP=64
   后果: 每层 128 次 AllReduce, 通信灾难
   正确: TP=8 + EP, AllReduce 只用于 attention

5. 忽略 Decode 阶段:
   错误: 只优化 Prefill 性能
   后果: Decode 延迟 (TPOT) 超出 SLA
   正确: Decode 阶段是延迟瓶颈, 需要专门优化

6. 盲目追求低延迟:
   错误: 所有请求都用 TP=8 最低延迟
   后果: GPU 利用率低, 成本高
   正确: 根据 SLA 要求选择合适的 TP, 用 DP 提升吞吐
```

## 8. 总结

### 8.1 决策速查表

```
场景                     推荐配置                  优先级
─────────────────────────────────────────────────────────
8B, 高吞吐              TP=1, DP=max              吞吐 > 延迟
8B, 低延迟              TP=2, DP=max/2            延迟 > 吞吐
70B, 平衡               TP=4, DP=max/4            平衡
70B, 长上下文            TP=8                      KV Cache > 吞吐
405B, bf16              TP=8, PP=2+               必须多节点
405B, FP8               TP=8, DP=max/8            单节点可放
MoE 600B+              TP=8, EP=N                 EP 跨节点
超长上下文 (1M+)        TP=8, CP=2-4              显存 > 一切
```

### 8.2 核心原则

```
1. 先 TP (节点内) → 再 PP/EP (跨节点) → 最后 DP (多副本)
2. 量化 (FP8) 可以将 PP 变为 TP, 将 TP 变为 DP → 优先考虑量化
3. 节点间通信昂贵 → 尽量把通信密集操作 (TP) 放在节点内
4. Decode 延迟是大多数服务的 SLA 瓶颈 → 优化 TPOT
5. KV Cache 是隐藏的显存大户 → 预留充足空间
6. 没有一劳永逸的配置 → 根据负载特征持续调优
```

---

> **下一节**：[动手练习](exercises.md) —— 通过实践巩固分布式推理的核心概念
