# Preemption 策略

> 当 GPU 显存不足以容纳所有正在运行的请求时，推理系统必须做出取舍——抢占（preempt）某些请求以释放显存。本节深入分析 Swap 和 Recomputation 两种策略的原理、实现和选择框架。

## 1. 为什么需要 Preemption？

### 1.1 显存压力的来源

在持续接收请求的在线推理场景中，GPU 显存面临以下压力：

```
时间线：
t=0   Request A 到达，分配 5 blocks   [已用: 5 / 100]
t=1   Request B 到达，分配 8 blocks   [已用: 13 / 100]
t=2   Request C 到达，分配 6 blocks   [已用: 19 / 100]
...
t=50  Request A 生成中，已用 45 blocks
      Request B 生成中，已用 32 blocks
      Request C 生成中，已用 23 blocks  [已用: 100 / 100]
      
t=51  Request D 到达 → 没有空闲 block！
      同时 A/B/C 的 decode 也需要新 block → 死锁风险！
```

关键洞察：由于生成长度不可预知，即使初始分配成功的请求，在生成过程中也可能遇到显存不足。这不是"拒绝新请求"就能解决的——**已在运行的请求也可能需要被中断**。

### 1.2 Preemption 的目标

1. **释放足够显存**：让高优先级或更早到达的请求能继续运行
2. **最小化浪费**：被抢占的请求已经完成的计算不应该完全白费
3. **公平性**：避免某些请求被无限期抢占（饥饿问题）
4. **低开销**：preemption 本身的开销应该远小于节省的资源

## 2. Swap（换出到 CPU）

### 2.1 原理

Swap 策略将被抢占请求的 KV Cache 从 GPU 显存**复制到 CPU 内存**，释放 GPU block。当该请求被恢复时，再将 KV Cache 从 CPU **换入**回 GPU。

```
Preemption (Swap Out):
  GPU Block 7  ──copy──→  CPU Block 0
  GPU Block 3  ──copy──→  CPU Block 1
  GPU Block 12 ──copy──→  CPU Block 2
  释放 GPU Block 7, 3, 12 → 加入 free list

Resume (Swap In):
  CPU Block 0  ──copy──→  GPU Block 22  (可能是不同的物理 block)
  CPU Block 1  ──copy──→  GPU Block 8
  CPU Block 2  ──copy──→  GPU Block 15
  更新 Block Table 映射
```

### 2.2 实现要点

```python
# 简化的 Swap 实现逻辑
class SwapManager:
    def __init__(self, num_cpu_blocks: int, block_size: int, ...):
        # CPU 侧的 KV cache 存储
        # 使用 pinned memory 加速 GPU ↔ CPU 传输
        self.cpu_key_cache = torch.zeros(
            (num_cpu_blocks, block_size, num_kv_heads, head_dim),
            dtype=dtype,
            pin_memory=True,  # 关键：pinned memory
        )
        self.cpu_value_cache = torch.zeros_like(self.cpu_key_cache)
        
        # CPU 侧的 free list
        self.free_cpu_blocks = deque(range(num_cpu_blocks))
    
    def swap_out(self, gpu_block_ids: List[int]) -> Dict[int, int]:
        """将 GPU blocks 换出到 CPU"""
        gpu_to_cpu_mapping = {}
        
        for gpu_block_id in gpu_block_ids:
            cpu_block_id = self.free_cpu_blocks.popleft()
            
            # 异步 D2H 拷贝
            self.cpu_key_cache[cpu_block_id].copy_(
                gpu_key_cache[gpu_block_id], non_blocking=True
            )
            self.cpu_value_cache[cpu_block_id].copy_(
                gpu_value_cache[gpu_block_id], non_blocking=True
            )
            
            gpu_to_cpu_mapping[gpu_block_id] = cpu_block_id
        
        # 等待拷贝完成后再释放 GPU blocks
        torch.cuda.synchronize()
        return gpu_to_cpu_mapping
    
    def swap_in(self, cpu_block_ids: List[int]) -> List[int]:
        """将 CPU blocks 换入到 GPU"""
        new_gpu_blocks = gpu_block_pool.allocate(len(cpu_block_ids))
        
        for cpu_block_id, gpu_block_id in zip(cpu_block_ids, new_gpu_blocks):
            # 异步 H2D 拷贝
            gpu_key_cache[gpu_block_id].copy_(
                self.cpu_key_cache[cpu_block_id], non_blocking=True
            )
            gpu_value_cache[gpu_block_id].copy_(
                self.cpu_value_cache[cpu_block_id], non_blocking=True
            )
            
            # 释放 CPU block
            self.free_cpu_blocks.append(cpu_block_id)
        
        return new_gpu_blocks
```

### 2.3 Swap 的性能分析

**PCIe 带宽瓶颈**：

以 PCIe 4.0 x16 为例，理论带宽约 32 GB/s（实际 ~25 GB/s）：

```
假设模型配置：
  num_layers = 32
  num_kv_heads = 8
  head_dim = 128
  dtype = float16 (2 bytes)
  block_size = 16

单个 block 的 KV Cache 大小：
  = 2 (K+V) × 32 (layers) × 8 (heads) × 128 (dim) × 16 (tokens) × 2 (bytes)
  = 2 × 32 × 8 × 128 × 16 × 2
  = 2,097,152 bytes ≈ 2 MB

换出 10 个 block（~160 tokens 的 KV）：
  = 20 MB
  传输时间 ≈ 20 / 25000 ≈ 0.8 ms

换出 100 个 block（~1600 tokens 的 KV）：
  = 200 MB
  传输时间 ≈ 200 / 25000 ≈ 8 ms
```

对于长序列，swap 的传输延迟可能达到数毫秒到数十毫秒级别，影响其他请求的 decode latency。

### 2.4 优缺点

| 优点 | 缺点 |
|------|------|
| 恢复速度快（仅需数据传输，无需重新计算） | 需要额外的 CPU 内存 |
| 不浪费已完成的计算 | PCIe 带宽是瓶颈 |
| 对于长序列更高效 | swap in/out 期间可能阻塞其他操作 |
| 实现相对简单 | CPU 内存也有限制 |

## 3. Recomputation（重新计算）

### 3.1 原理

Recomputation 策略直接**丢弃**被抢占请求的 KV Cache。当请求被恢复时，需要重新执行 prefill 以重建 KV Cache。

```
Preemption (Recompute):
  直接释放 GPU Block 7, 3, 12
  保存请求的 token IDs（非常小，几 KB）
  不需要任何数据传输

Resume (Recompute):
  将请求重新放入 waiting 队列
  重新执行 prefill：
    input = original_prompt + already_generated_tokens
  重建完整的 KV Cache
  从之前中断的位置继续 decode
```

### 3.2 实现

Recomputation 的实现比 swap 简单得多：

```python
# 简化的 Recomputation 逻辑
class Scheduler:
    def _preempt_by_recompute(self, request: Request):
        # 1. 记录已生成的 tokens
        request.saved_output_tokens = request.output_token_ids.copy()
        
        # 2. 释放所有 GPU blocks
        self.kv_cache_manager.free(request.request_id)
        
        # 3. 将请求移回 waiting 队列
        self.running_queue.remove(request)
        self.waiting_queue.appendleft(request)  # 放在队首，优先恢复
        
        # 恢复时：
        # request 重新进入 running 队列
        # prefill 输入 = original_prompt + saved_output_tokens
        # 从 len(original_prompt + saved_output_tokens) 开始继续 decode
```

### 3.3 Recomputation 的性能分析

```
假设模型配置：
  LLaMA-3-8B on A100-80GB
  Prefill 吞吐量 ≈ 30,000 tokens/s

场景 1：短 prompt (256 tokens) + 已生成 50 tokens
  重算输入 = 306 tokens
  重算时间 ≈ 306 / 30000 ≈ 10 ms
  → 开销很小，recompute 可接受

场景 2：长 prompt (4096 tokens) + 已生成 200 tokens
  重算输入 = 4296 tokens
  重算时间 ≈ 4296 / 30000 ≈ 143 ms
  → 开销较大，但仍可能优于 swap（如果 swap 需要传输大量 KV）

场景 3：超长 prompt (32K tokens) + 已生成 1000 tokens
  重算输入 = 33,000 tokens
  重算时间 ≈ 33000 / 30000 ≈ 1100 ms
  → 开销很大，swap 通常更优
```

### 3.4 优缺点

| 优点 | 缺点 |
|------|------|
| 不需要额外 CPU 内存 | 需要重新执行 prefill（计算开销） |
| 无 PCIe 带宽压力 | 浪费已完成的计算 |
| 实现极简 | 对长 prompt 场景不友好 |
| 释放速度最快 | 恢复时占用 GPU 计算资源 |

## 4. vLLM 中的 Preemption 实现

### 4.1 Preemption 触发条件

在 vLLM 的 Scheduler 中，preemption 在以下情况被触发：

```python
# vllm/v1/core/scheduler.py (简化)
class Scheduler:
    def _schedule_running(self) -> List[Request]:
        """为 running 队列中的请求分配 decode 所需的 block"""
        requests_to_preempt = []
        
        for request in reversed(self.running_queue):
            # 尝试分配 1 个新 token 的 slot
            result = self.kv_cache_manager.allocate_slots(
                request, num_new_tokens=1, ...
            )
            
            if result is None:
                # 分配失败 → 该请求需要被抢占
                requests_to_preempt.append(request)
                
                # 释放该请求的 blocks 后，重试其他请求
                self.kv_cache_manager.free(request.request_id)
        
        return requests_to_preempt
```

**抢占顺序**：vLLM 采用 **LIFO**（Last In, First Out）策略——最后加入 running 队列的请求最先被抢占。这是因为：

1. 较新的请求生成的 token 较少，KV Cache 较小，重算代价低
2. 较旧的请求已经投入了大量计算，抢占它们的浪费更大
3. LIFO 避免了频繁抢占-恢复同一个请求的"乒乓"效应

### 4.2 `preemption_mode` 配置项

```bash
# vLLM 启动参数
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8B-Instruct \
    --preemption-mode swap   # 或 recompute
```

```python
# 配置定义 (vllm/config.py)
class SchedulerConfig:
    preemption_mode: Optional[str] = None
    # None → 自动选择（vLLM 默认逻辑）
    # "swap" → 强制使用 swap
    # "recompute" → 强制使用 recompute
```

### 4.3 自动选择逻辑

当 `preemption_mode=None` 时，vLLM 的自动选择逻辑（在 v0 架构中有明确实现）：

```python
def _get_preemption_mode(self, request: Request) -> PreemptionMode:
    """自动决定使用 swap 还是 recompute"""
    
    # 如果没有配置 CPU swap space，只能 recompute
    if self.swap_space == 0:
        return PreemptionMode.RECOMPUTE
    
    # 如果 CPU swap space 不足以容纳该请求的 KV Cache
    num_blocks = len(self.req_to_blocks[request.request_id])
    if num_blocks > self.available_cpu_blocks:
        return PreemptionMode.RECOMPUTE
    
    # 默认使用 recompute（vLLM 的默认偏好）
    # 在 v1 架构中，默认行为是 recompute
    return PreemptionMode.RECOMPUTE
```

注意：在 vLLM v1 架构中（0.7+），preemption 的默认行为倾向于 **recompute**。这是因为：
1. v1 架构初期简化了 swap 相关代码
2. 现代 GPU 的 prefill 吞吐量很高，recompute 的代价相对可控
3. 不需要管理 CPU 侧的 block allocator，降低复杂度

### 4.4 v0 架构中的完整 Swap 实现

在 v0 架构中，swap 有更完整的实现：

```python
# vllm/core/scheduler.py (v0, 简化)
class Scheduler:
    def _preempt(self, seq_group: SequenceGroup, 
                 preemption_mode: PreemptionMode):
        if preemption_mode == PreemptionMode.SWAP:
            self._swap_out(seq_group)
        else:
            self._recompute(seq_group)
    
    def _swap_out(self, seq_group: SequenceGroup):
        # 获取 GPU → CPU 的 block 映射
        mapping = self.block_manager.swap_out(seq_group)
        
        # 记录映射关系，用于后续 swap in
        seq_group.swap_mapping = mapping
        
        # 移入 swapped 队列（既不在 running 也不在 waiting）
        self.running.remove(seq_group)
        self.swapped.append(seq_group)
    
    def _schedule_swapped(self):
        """尝试将 swapped 队列中的请求换回 GPU"""
        for seq_group in self.swapped:
            # 检查是否有足够的 GPU blocks
            if self.block_manager.can_swap_in(seq_group):
                mapping = self.block_manager.swap_in(seq_group)
                self.swapped.remove(seq_group)
                self.running.append(seq_group)
```

## 5. 决策框架：何时选择 Swap vs Recompute？

### 5.1 量化比较模型

```
定义：
  T_swap_out  = swap out 时间 = KV_size / PCIe_bandwidth
  T_swap_in   = swap in 时间  = KV_size / PCIe_bandwidth
  T_recompute = 重算时间      = (prompt_len + generated_len) / prefill_throughput

  KV_size = num_blocks × block_size × num_layers × 2 × num_kv_heads × head_dim × dtype_bytes

选择 swap 当：
  T_swap_out + T_swap_in < T_recompute
  ⟺ 2 × KV_size / PCIe_BW < total_tokens / prefill_throughput
```

### 5.2 决策矩阵

| 场景 | Prompt 长度 | 生成长度 | KV 大小 | 推荐策略 | 原因 |
|------|------------|---------|---------|---------|------|
| 短 prompt + 长生成 | 256 | 2048 | 大 | **Swap** | 重算 2304 tokens 很贵 |
| 长 prompt + 短生成 | 8192 | 128 | 大 | **Swap** | KV 大但重算更贵 |
| 短 prompt + 短生成 | 256 | 64 | 小 | **Recompute** | KV 小，重算 320 tokens 很快 |
| 超长 prompt | 128K | 256 | 极大 | **取决于** | swap 传输量大，但重算更慢 |
| CPU 内存不足 | 任意 | 任意 | - | **Recompute** | 无法 swap |
| 高 PCIe 争用 | 任意 | 任意 | - | **Recompute** | swap 可能阻塞其他传输 |

### 5.3 具体场景分析

**场景 1：聊天机器人服务**

```
典型特征：
  - Prompt: 500-2000 tokens（含系统 prompt + 历史对话）
  - 生成: 100-500 tokens
  - 并发: 高
  - 抢占频率: 中等

推荐：Recompute
  - 重算 600-2500 tokens 仅需 20-80ms
  - 不需要管理 CPU 内存
  - 简化系统复杂度
  - 配合 prefix caching，系统 prompt 部分不需要重算
```

**场景 2：长文档摘要**

```
典型特征：
  - Prompt: 10K-100K tokens（长文档）
  - 生成: 200-1000 tokens
  - 并发: 低-中
  - 抢占频率: 低

推荐：Swap
  - 重算 10K+ tokens 需要 300ms+
  - swap 10K tokens 的 KV 约需 100-200ms（取决于模型大小）
  - 长 prompt 的 prefill 是 GPU 密集操作，重算浪费 GPU 算力
```

**场景 3：代码生成（长输出）**

```
典型特征：
  - Prompt: 500-3000 tokens
  - 生成: 2000-8000 tokens（生成完整代码文件）
  - 并发: 中
  - 抢占频率: 高（长生成导致 block 消耗大）

推荐：Swap
  - 被抢占时可能已经生成了数千 tokens
  - 重算代价随已生成 token 数线性增长
  - swap 可以保存已有成果
```

## 6. Preemption 的代价与优化

### 6.1 Preemption 是昂贵的

无论选择哪种策略，preemption 都是有代价的：

```
Recompute 的代价：
  - 被抢占请求的所有计算白费
  - 恢复时的 prefill 占用 GPU 计算资源
  - 影响其他请求的 TTFT 和 TBT

Swap 的代价：
  - D2H 和 H2D 传输占用 PCIe 带宽
  - 可能与模型权重加载、其他 swap 操作争用带宽
  - CPU 内存占用
  - 需要同步（cudaMemcpy 或 stream synchronize）
```

### 6.2 减少 Preemption 的策略

**1. 合理设置 `gpu_memory_utilization`**

```bash
# 保留更多显存余量，减少 preemption 触发频率
python -m vllm.entrypoints.openai.api_server \
    --gpu-memory-utilization 0.85  # 默认 0.9
```

**2. 限制 `max_num_seqs`**

```bash
# 限制同时运行的最大请求数
python -m vllm.entrypoints.openai.api_server \
    --max-num-seqs 128  # 减少并发以降低显存压力
```

**3. 设置 `max_model_len`**

```bash
# 限制最大序列长度，防止单个请求占用过多 block
python -m vllm.entrypoints.openai.api_server \
    --max-model-len 4096
```

**4. 启用 Prefix Caching**

```bash
# Prefix caching 减少重复 KV 的存储
python -m vllm.entrypoints.openai.api_server \
    --enable-prefix-caching
```

### 6.3 监控 Preemption

vLLM 提供 Prometheus metrics 来监控 preemption：

```python
# 关键指标
vllm:num_preemptions_total          # preemption 总次数
vllm:gpu_cache_usage_perc           # GPU KV cache 使用率
vllm:cpu_cache_usage_perc           # CPU swap space 使用率

# 监控建议
# - num_preemptions_total 持续增长 → 需要增加 GPU 或减少并发
# - gpu_cache_usage_perc > 95% → 接近 preemption 触发阈值
# - cpu_cache_usage_perc > 80% → swap space 即将耗尽
```

## 7. 前沿发展：超越 Swap 和 Recompute

### 7.1 分级 Preemption

一些研究提出更细粒度的 preemption 策略：

- **部分 Swap**：只换出部分 layer 的 KV Cache，而不是全部
- **渐进式 Recompute**：先释放最新的 block（重算代价低），不够再释放更早的 block
- **优先级感知**：结合请求优先级、SLO deadline 等因素决定抢占对象

### 7.2 KV Cache Offloading

与 preemption 不同，offloading 是**主动的**——在 block 不被当前 decode step 使用时，提前将其换出到 CPU/SSD，避免触发 preemption。详见 [Ch06: KV Cache 卸载](../06-kv-offloading/)。

### 7.3 Disaggregated Prefill

在 Prefill-Decode 分离架构（详见 [Ch05](../05-disaggregated-serving/)）中，preemption 的语义发生变化：
- Prefill 节点不持有长期 KV Cache，不需要 preemption
- Decode 节点的 preemption 可以通过"将 KV 发回 prefill 节点重建"来实现
- 整体系统的 preemption 频率降低，因为资源分配更精细

---

**核心总结：**
- **Swap** 适合 KV Cache 大、重算代价高的场景
- **Recompute** 适合 KV Cache 小、CPU 内存有限的场景
- 减少 preemption 的发生比选择更好的 preemption 策略更重要
- 监控 `num_preemptions_total` 是生产环境必须做的事情
