# 动手练习

> 通过实践加深对 KV Cache offloading 各个方面的理解。

---

## 练习 1：实现一个 KV Cache Offload Manager

**目标：** 从零实现一个简化版的 KV Cache Offload Manager，理解 GPU↔CPU 数据搬运的核心机制。

### 1.1 基础要求

实现一个 `SimpleOffloadManager` 类，支持以下功能：
- 预分配 CPU 端的 pinned memory 作为 offload 存储
- 将指定的 KV block 从 GPU 异步传输到 CPU
- 将指定的 KV block 从 CPU 异步传输回 GPU
- 使用 LRU 策略管理 CPU 存储空间

```python
import torch
from collections import OrderedDict
from typing import List, Dict, Optional, Tuple

class SimpleOffloadManager:
    """
    练习：实现一个简化版的 KV Cache Offload Manager。
    
    参数：
        num_gpu_blocks: GPU 上的 KV block 数量
        num_cpu_blocks: CPU 上预分配的 KV block 数量
        block_size: 每个 block 包含的 token 数
        num_layers: Transformer 层数
        num_kv_heads: KV head 数量
        head_dim: head 维度
        dtype: 数据类型
    """
    
    def __init__(
        self,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        block_size: int = 16,
        num_layers: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        dtype: torch.dtype = torch.float16,
    ):
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks
        self.block_size = block_size
        
        # TODO: 1. 分配 GPU KV Cache 存储
        # 形状: [num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim]
        # 提示: 使用 torch.randn(..., device='cuda')
        self.gpu_kv_cache = None  # 替换为你的实现
        
        # TODO: 2. 分配 CPU KV Cache 存储（必须使用 pinned memory）
        # 提示: 使用 torch.empty(..., pin_memory=True)
        self.cpu_kv_cache = None  # 替换为你的实现
        
        # TODO: 3. 初始化空闲 CPU block 池
        self.free_cpu_blocks: List[int] = []  # 替换为你的实现
        
        # TODO: 4. 初始化 GPU→CPU block 映射表
        self.gpu_to_cpu_map: Dict[int, int] = {}
        
        # TODO: 5. 创建传输用的 CUDA stream
        self.transfer_stream = None  # 替换为你的实现
        
        # TODO: 6. 初始化 LRU 追踪器
        self.lru_tracker = OrderedDict()
    
    def offload_blocks(self, gpu_block_ids: List[int]) -> None:
        """
        将指定的 GPU blocks 异步 offload 到 CPU。
        
        步骤：
        1. 为每个 GPU block 分配一个空闲的 CPU block
        2. 在 transfer stream 中执行异步拷贝
        3. 更新映射表和 LRU 追踪器
        
        注意：
        - 如果 CPU 空间不足，应该先驱逐最久未使用的 CPU block
        - 使用 non_blocking=True 实现异步传输
        """
        # TODO: 实现 offload 逻辑
        pass
    
    def reload_blocks(self, gpu_block_ids: List[int]) -> None:
        """
        将指定的 blocks 从 CPU 异步 reload 回 GPU。
        
        步骤：
        1. 查找每个 GPU block 对应的 CPU block
        2. 在 transfer stream 中执行异步拷贝
        3. 归还 CPU block，更新映射表
        """
        # TODO: 实现 reload 逻辑
        pass
    
    def sync(self) -> None:
        """等待所有异步传输完成。"""
        # TODO: 同步 transfer stream
        pass
    
    def access_block(self, gpu_block_id: int) -> None:
        """
        标记一个 GPU block 被访问（用于 LRU 追踪）。
        """
        # TODO: 更新 LRU 追踪器
        pass
    
    def get_stats(self) -> dict:
        """
        返回当前状态统计。
        
        包括：
        - GPU blocks 使用情况
        - CPU blocks 使用情况
        - 映射表大小
        """
        # TODO: 返回统计信息
        pass


# ========== 测试代码 ==========

def test_basic_offload_reload():
    """测试基本的 offload 和 reload 功能。"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU test.")
        print("请使用 CPU 模拟版本（见下方 bonus 部分）。")
        return
    
    manager = SimpleOffloadManager(
        num_gpu_blocks=64,
        num_cpu_blocks=128,
        block_size=16,
        num_layers=4,      # 为了测试方便，使用较小的参数
        num_kv_heads=4,
        head_dim=64,
    )
    
    # 1. 在 GPU 上填充一些数据
    test_block_ids = [0, 1, 2, 3]
    for bid in test_block_ids:
        manager.gpu_kv_cache[:, :, bid] = torch.randn_like(
            manager.gpu_kv_cache[:, :, bid]
        )
        manager.access_block(bid)
    
    # 2. 保存原始数据用于验证
    original_data = {
        bid: manager.gpu_kv_cache[:, :, bid].clone()
        for bid in test_block_ids
    }
    
    # 3. Offload 到 CPU
    manager.offload_blocks(test_block_ids)
    manager.sync()
    print("Offload completed.")
    
    # 4. 清空 GPU 上的数据（模拟这些 block 被分配给其他请求）
    for bid in test_block_ids:
        manager.gpu_kv_cache[:, :, bid].zero_()
    
    # 5. Reload 回 GPU
    manager.reload_blocks(test_block_ids)
    manager.sync()
    print("Reload completed.")
    
    # 6. 验证数据一致性
    for bid in test_block_ids:
        if torch.allclose(
            manager.gpu_kv_cache[:, :, bid],
            original_data[bid],
            atol=1e-6
        ):
            print(f"  Block {bid}: PASS ✓")
        else:
            print(f"  Block {bid}: FAIL ✗")
            diff = (manager.gpu_kv_cache[:, :, bid] - original_data[bid]).abs().max()
            print(f"    Max diff: {diff.item()}")
    
    print(f"\nStats: {manager.get_stats()}")


def test_lru_eviction():
    """测试 LRU 驱逐策略。"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU test.")
        return
    
    manager = SimpleOffloadManager(
        num_gpu_blocks=32,
        num_cpu_blocks=4,   # CPU 只有 4 个 block！
        block_size=16,
        num_layers=2,
        num_kv_heads=2,
        head_dim=32,
    )
    
    # Offload 6 个 blocks 到只有 4 个 slot 的 CPU 存储
    # 前 4 个应该正常存储，第 5、6 个应该触发 LRU 驱逐
    
    for bid in range(6):
        manager.gpu_kv_cache[:, :, bid] = bid * 1.0  # 用简单值填充
        manager.access_block(bid)
    
    # 先 offload 0-3
    manager.offload_blocks([0, 1, 2, 3])
    manager.sync()
    print(f"After offload [0,1,2,3]: {manager.get_stats()}")
    
    # 再 offload 4-5（应该触发驱逐 block 0 和 1）
    manager.offload_blocks([4, 5])
    manager.sync()
    print(f"After offload [4,5]: {manager.get_stats()}")
    
    # block 0 和 1 应该已被驱逐，无法 reload
    # block 2,3,4,5 应该可以 reload
    try:
        manager.reload_blocks([2, 3])
        manager.sync()
        print("Reload [2,3]: PASS ✓")
    except Exception as e:
        print(f"Reload [2,3]: FAIL ✗ - {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Basic Offload and Reload")
    print("=" * 60)
    test_basic_offload_reload()
    
    print("\n" + "=" * 60)
    print("Test 2: LRU Eviction")
    print("=" * 60)
    test_lru_eviction()
```

### 1.2 进阶要求

在基础实现之上，添加以下功能：

1. **双 Stream 支持**：offload 和 reload 使用不同的 CUDA stream，可以并行执行
2. **批量传输优化**：将多个小 block 的传输合并为一个大的 `copy_` 操作
3. **传输带宽统计**：记录每次传输的数据量和耗时，计算实际带宽

### 1.3 思考题

1. 为什么 CPU 端必须使用 pinned memory？如果使用普通 pageable memory 会发生什么？
2. 在什么情况下异步传输无法完全隐藏延迟？
3. 如果 GPU 有 80 GB HBM，CPU 有 512 GB DRAM，PCIe 5.0 带宽 50 GB/s，decode step 耗时 10 ms，那么每个 decode step 最多可以预取多少 KV Cache 数据？

---

## 练习 2：ARC 驱逐策略实现与基准测试

**目标：** 实现 ARC (Adaptive Replacement Cache) 策略，并与 LRU 在模拟 KV Cache 负载下进行性能对比。

### 2.1 实现 ARC

```python
from collections import OrderedDict
from typing import List, Optional

class ARCCache:
    """
    练习：实现 ARC (Adaptive Replacement Cache)。
    
    ARC 维护四个列表：
    - T1: 最近访问过一次的条目
    - T2: 最近访问过多次的条目
    - B1: T1 的 ghost 条目（只记录 key，不存数据）
    - B2: T2 的 ghost 条目
    
    自适应参数 p 控制 T1 和 T2 之间的大小平衡。
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.p = 0  # 自适应目标
        
        self.t1 = OrderedDict()  # 访问一次
        self.t2 = OrderedDict()  # 访问多次
        self.b1 = OrderedDict()  # T1 的 ghost
        self.b2 = OrderedDict()  # T2 的 ghost
    
    def access(self, key: int) -> bool:
        """
        访问一个 key。返回 True 表示 cache hit，False 表示 miss。
        
        TODO: 实现 ARC 的五种情况：
        1. key 在 T1 中 → 移到 T2
        2. key 在 T2 中 → 更新位置
        3. key 在 B1 中 → 增大 p，调整并移到 T2
        4. key 在 B2 中 → 减小 p，调整并移到 T2
        5. key 不在任何列表中 → 新条目
        """
        # TODO: 实现
        pass
    
    def _replace(self, key: int, in_b2: bool):
        """ARC 的内部替换操作。"""
        # TODO: 实现
        pass
    
    def get_cached_keys(self) -> set:
        """返回当前 cache 中的所有 keys（T1 + T2）。"""
        return set(self.t1.keys()) | set(self.t2.keys())
    
    def size(self) -> int:
        """返回当前 cache 大小。"""
        return len(self.t1) + len(self.t2)


class LRUCache:
    """LRU Cache（用于对比）。"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def access(self, key: int) -> bool:
        """访问一个 key。返回 True 表示 hit，False 表示 miss。"""
        if key in self.cache:
            self.cache.move_to_end(key)
            return True
        
        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        
        self.cache[key] = True
        return False
    
    def get_cached_keys(self) -> set:
        return set(self.cache.keys())
    
    def size(self) -> int:
        return len(self.cache)
```

### 2.2 模拟 KV Cache 负载

```python
import random

def generate_workload(
    num_requests: int = 1000,
    num_prefix_groups: int = 5,    # 5 个不同的 system prompt
    prefix_length: int = 50,       # 每个 prefix 50 个 blocks
    private_length: int = 20,      # 每个请求私有 20 个 blocks
    scan_probability: float = 0.1, # 10% 概率出现长 context 扫描
    scan_length: int = 200,        # 扫描请求长度 200 blocks
) -> List[List[int]]:
    """
    生成模拟的 KV Cache 访问序列。
    
    模拟场景：
    - 多个请求共享若干 system prompt（高频 prefix）
    - 每个请求有自己的私有 context
    - 偶尔出现长 context 扫描（一次性访问大量不同 block）
    
    返回：每个请求访问的 block ID 序列
    """
    block_id_counter = 0
    
    # 为每个 prefix group 分配固定的 block IDs
    prefix_blocks = {}
    for g in range(num_prefix_groups):
        prefix_blocks[g] = list(range(
            block_id_counter, 
            block_id_counter + prefix_length
        ))
        block_id_counter += prefix_length
    
    workload = []
    for _ in range(num_requests):
        if random.random() < scan_probability:
            # 长 context 扫描请求
            scan_blocks = list(range(
                block_id_counter, 
                block_id_counter + scan_length
            ))
            block_id_counter += scan_length
            workload.append(scan_blocks)
        else:
            # 正常请求：prefix + private
            group = random.randint(0, num_prefix_groups - 1)
            private = list(range(
                block_id_counter, 
                block_id_counter + private_length
            ))
            block_id_counter += private_length
            workload.append(prefix_blocks[group] + private)
    
    return workload


def benchmark_policies(
    workload: List[List[int]],
    cache_capacity: int = 200,
):
    """
    对比 LRU 和 ARC 在给定负载下的命中率。
    
    TODO: 
    1. 创建 LRU 和 ARC cache 实例
    2. 对每个请求的每个 block 依次访问
    3. 统计命中率
    4. 打印对比结果
    """
    lru = LRUCache(cache_capacity)
    arc = ARCCache(cache_capacity)
    
    lru_hits, lru_total = 0, 0
    arc_hits, arc_total = 0, 0
    
    for request_blocks in workload:
        for block_id in request_blocks:
            lru_total += 1
            arc_total += 1
            
            if lru.access(block_id):
                lru_hits += 1
            if arc.access(block_id):
                arc_hits += 1
    
    print(f"\nCache Capacity: {cache_capacity} blocks")
    print(f"Total accesses: {lru_total}")
    print(f"LRU hit rate: {lru_hits/lru_total*100:.2f}%")
    print(f"ARC hit rate: {arc_hits/arc_total*100:.2f}%")
    print(f"ARC advantage: {(arc_hits-lru_hits)/lru_total*100:+.2f}%")


if __name__ == "__main__":
    print("Generating workload...")
    workload = generate_workload(
        num_requests=500,
        num_prefix_groups=5,
        prefix_length=50,
        private_length=20,
        scan_probability=0.15,  # 15% 扫描请求
        scan_length=200,
    )
    print(f"Generated {len(workload)} requests")
    
    # 测试不同 cache 容量
    for capacity in [100, 200, 500]:
        benchmark_policies(workload, cache_capacity=capacity)
```

### 2.3 思考题

1. 在什么样的访问模式下 ARC 明显优于 LRU？在什么模式下两者相近？
2. ARC 的 ghost 列表 (B1, B2) 起到了什么作用？如果去掉 ghost 列表，ARC 退化成什么？
3. 在 KV Cache 场景中，`scan_probability` 对应现实中的什么情况？

---

## 练习 3：传输延迟隐藏效果分析

**目标：** 通过基准测试量化 CUDA stream overlap 对传输延迟的隐藏效果。

### 3.1 实验代码

```python
import torch
import time
from typing import List, Tuple

def measure_sync_transfer(
    data_size_mb: float,
    num_transfers: int = 10,
) -> float:
    """
    测量同步传输的延迟（GPU → CPU → GPU roundtrip）。
    
    TODO:
    1. 在 GPU 上创建指定大小的 tensor
    2. 在 CPU 上创建 pinned memory 目标 tensor
    3. 执行同步 copy (不使用 non_blocking)
    4. 返回平均 roundtrip 时间 (ms)
    """
    if not torch.cuda.is_available():
        print("CUDA not available, returning simulated result.")
        return data_size_mb / 50.0 * 1000 * 2  # 模拟 50 GB/s, roundtrip
    
    size = int(data_size_mb * 1024 * 1024 / 2)  # FP16, 2 bytes per element
    gpu_tensor = torch.randn(size, dtype=torch.float16, device='cuda')
    cpu_tensor = torch.empty(size, dtype=torch.float16, pin_memory=True)
    
    # Warm up
    cpu_tensor.copy_(gpu_tensor)
    gpu_tensor.copy_(cpu_tensor)
    torch.cuda.synchronize()
    
    total_time = 0
    for _ in range(num_transfers):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # 同步传输: GPU → CPU
        cpu_tensor.copy_(gpu_tensor)
        torch.cuda.synchronize()
        
        # 同步传输: CPU → GPU
        gpu_tensor.copy_(cpu_tensor)
        torch.cuda.synchronize()
        
        total_time += time.perf_counter() - start
    
    return (total_time / num_transfers) * 1000  # ms


def measure_async_transfer_with_compute(
    data_size_mb: float,
    compute_time_ms: float,
    num_iterations: int = 10,
) -> Tuple[float, float]:
    """
    测量异步传输 + 计算重叠的效果。
    
    模拟一个 decode step：
    - 主 stream：执行 attention 计算（用 sleep 模拟）
    - 传输 stream：同时进行 KV Cache offload/reload
    
    返回：(总时间 ms, 纯传输时间 ms)
    
    TODO:
    1. 创建独立的传输 stream
    2. 在主 stream 中执行 "计算"（矩阵乘法模拟）
    3. 同时在传输 stream 中执行 GPU→CPU 拷贝
    4. 测量总时间和对比
    """
    if not torch.cuda.is_available():
        pure_transfer = data_size_mb / 50.0 * 1000  # 模拟 50 GB/s
        overlap_time = max(compute_time_ms, pure_transfer)
        return overlap_time, pure_transfer
    
    size = int(data_size_mb * 1024 * 1024 / 2)
    gpu_tensor = torch.randn(size, dtype=torch.float16, device='cuda')
    cpu_tensor = torch.empty(size, dtype=torch.float16, pin_memory=True)
    
    # 计算用的矩阵（模拟 attention 计算）
    compute_size = 2048
    a = torch.randn(compute_size, compute_size, device='cuda', dtype=torch.float16)
    b = torch.randn(compute_size, compute_size, device='cuda', dtype=torch.float16)
    
    transfer_stream = torch.cuda.Stream()
    
    # 测量纯传输时间
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    cpu_tensor.copy_(gpu_tensor)
    torch.cuda.synchronize()
    pure_transfer_ms = (time.perf_counter() - t0) * 1000
    
    # 测量重叠执行时间
    total_time = 0
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # 主 stream: 计算
        for _ in range(max(1, int(compute_time_ms / 0.5))):
            c = torch.mm(a, b)
        
        # 传输 stream: 同时传输
        with torch.cuda.stream(transfer_stream):
            cpu_tensor.copy_(gpu_tensor, non_blocking=True)
        
        torch.cuda.synchronize()
        total_time += time.perf_counter() - start
    
    overlap_ms = (total_time / num_iterations) * 1000
    return overlap_ms, pure_transfer_ms


def run_latency_hiding_analysis():
    """
    运行完整的延迟隐藏效果分析。
    
    生成一个表格，展示不同数据大小和计算时间下的延迟隐藏效果。
    """
    print("\n" + "=" * 80)
    print("延迟隐藏效果分析")
    print("=" * 80)
    
    data_sizes = [0.1, 0.5, 1.0, 2.0]  # MB
    compute_times = [5, 10, 20]          # ms
    
    print(f"\n{'数据大小':>10} | {'纯传输时间':>12} | ", end="")
    for ct in compute_times:
        print(f"{'计算'+str(ct)+'ms 重叠':>16} | ", end="")
    print()
    print("-" * 80)
    
    for ds in data_sizes:
        sync_time = measure_sync_transfer(ds, num_transfers=5)
        print(f"{ds:>8.1f} MB | {sync_time/2:>10.2f} ms | ", end="")
        
        for ct in compute_times:
            overlap_time, pure_transfer = measure_async_transfer_with_compute(
                ds, ct, num_iterations=5
            )
            hiding_ratio = max(0, 1 - (overlap_time - ct) / pure_transfer) * 100
            print(f"{overlap_time:>7.2f}ms ({hiding_ratio:>3.0f}%) | ", end="")
        
        print()
    
    print("\n说明：百分比表示传输延迟被隐藏的比例。100% = 完全隐藏。")


if __name__ == "__main__":
    run_latency_hiding_analysis()
```

### 3.2 分析问题

完成实验后，回答以下问题：

1. 当传输数据量为 1 GB 时，需要多长的计算时间才能完全隐藏传输延迟？
2. Pinned memory 和 pageable memory 的传输带宽差异有多大？（修改代码，将 `pin_memory=True` 改为 `False` 重新测量）
3. 双向同时传输（一个 stream offload，一个 stream reload）的总带宽是否等于 PCIe 理论双向带宽？

---

## 练习 4：OpenAI Extended Caching 成本分析

**目标：** 编写一个成本分析工具，帮助决定是否应该使用 Extended Caching。

### 4.1 实现成本计算器

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class CachingScenario:
    """定义一个使用场景"""
    name: str
    system_prompt_tokens: int       # System prompt 长度
    avg_user_message_tokens: int    # 平均用户消息长度
    avg_history_tokens: int         # 平均对话历史长度
    queries_per_hour: float         # 每小时查询数
    hours_per_day: float            # 每天活跃小时数
    inter_query_interval_min: float # 平均查询间隔（分钟）
    num_unique_prefixes: int        # 不同前缀的数量


@dataclass
class PricingModel:
    """OpenAI 定价模型"""
    input_price_per_m: float = 2.50         # $/M tokens
    cached_price_per_m: float = 1.25        # $/M tokens (50% discount)
    output_price_per_m: float = 10.00       # $/M tokens
    extended_storage_per_m_per_hour: float = 0.0  # $/M cached tokens/hour
    avg_output_tokens: int = 500


def analyze_caching_benefit(
    scenario: CachingScenario,
    pricing: PricingModel,
) -> dict:
    """
    分析 Extended Caching 在给定场景下的成本效益。
    
    TODO: 实现以下计算：
    1. 无缓存时的每日成本
    2. 标准缓存（5-10 分钟 TTL）的预期命中率和每日成本
    3. Extended Caching（24h TTL）的预期命中率和每日成本
    4. 年度节省金额
    
    提示：
    - 标准缓存的命中率取决于 inter_query_interval 和 5min TTL
    - Extended Caching 的命中率取决于 inter_query_interval 和 24h TTL
    - 只有 system_prompt + 历史的共同前缀部分可以被缓存
    """
    
    total_input_tokens = (
        scenario.system_prompt_tokens + 
        scenario.avg_history_tokens + 
        scenario.avg_user_message_tokens
    )
    cacheable_tokens = scenario.system_prompt_tokens  # 可缓存的前缀部分
    
    daily_queries = scenario.queries_per_hour * scenario.hours_per_day
    
    # 1. 无缓存成本
    daily_input_cost_no_cache = (
        total_input_tokens * daily_queries * pricing.input_price_per_m / 1_000_000
    )
    daily_output_cost = (
        pricing.avg_output_tokens * daily_queries * pricing.output_price_per_m / 1_000_000
    )
    
    # 2. 标准缓存命中率估算
    # 如果查询间隔 < 5 分钟且前缀相同，则命中
    standard_ttl_min = 7  # 平均 7 分钟
    if scenario.inter_query_interval_min <= standard_ttl_min:
        standard_hit_rate = min(0.95, 1.0 - scenario.inter_query_interval_min / (standard_ttl_min * 3))
    else:
        # 间隔超过 TTL，命中率随间隔指数衰减
        standard_hit_rate = max(0.0, 0.5 ** (scenario.inter_query_interval_min / standard_ttl_min))
    
    # 考虑多个不同前缀的影响
    standard_hit_rate *= min(1.0, 1.0 / scenario.num_unique_prefixes * 3)
    
    # 3. Extended Caching 命中率估算
    extended_ttl_min = 24 * 60  # 24 小时
    if scenario.inter_query_interval_min <= extended_ttl_min:
        extended_hit_rate = min(0.95, 1.0 - scenario.inter_query_interval_min / (extended_ttl_min * 3))
    else:
        extended_hit_rate = 0.0
    extended_hit_rate *= min(1.0, 1.0 / scenario.num_unique_prefixes * 3)
    
    # 4. 计算各种场景的成本
    def calc_cost(hit_rate, storage_cost=0):
        cached = cacheable_tokens * hit_rate
        uncached = total_input_tokens - cached
        input_cost = (
            (cached * pricing.cached_price_per_m + uncached * pricing.input_price_per_m)
            * daily_queries / 1_000_000
        )
        return input_cost + daily_output_cost + storage_cost
    
    # Extended caching 存储成本
    extended_storage_cost = (
        cacheable_tokens * pricing.extended_storage_per_m_per_hour 
        * scenario.hours_per_day / 1_000_000
    )
    
    no_cache_daily = daily_input_cost_no_cache + daily_output_cost
    standard_daily = calc_cost(standard_hit_rate)
    extended_daily = calc_cost(extended_hit_rate, extended_storage_cost)
    
    return {
        "scenario": scenario.name,
        "daily_queries": daily_queries,
        "cacheable_tokens": cacheable_tokens,
        "total_input_tokens": total_input_tokens,
        "standard_hit_rate": standard_hit_rate,
        "extended_hit_rate": extended_hit_rate,
        "no_cache_daily_cost": no_cache_daily,
        "standard_cache_daily_cost": standard_daily,
        "extended_cache_daily_cost": extended_daily,
        "standard_daily_savings": no_cache_daily - standard_daily,
        "extended_daily_savings": no_cache_daily - extended_daily,
        "extended_annual_savings": (no_cache_daily - extended_daily) * 365,
    }


# ========== 预定义场景 ==========

scenarios = [
    CachingScenario(
        name="客服系统",
        system_prompt_tokens=5000,
        avg_user_message_tokens=200,
        avg_history_tokens=2000,
        queries_per_hour=100,
        hours_per_day=16,
        inter_query_interval_min=3,     # 同一前缀平均 3 分钟一次
        num_unique_prefixes=3,          # 3 种 system prompt
    ),
    CachingScenario(
        name="代码助手",
        system_prompt_tokens=8000,
        avg_user_message_tokens=500,
        avg_history_tokens=5000,
        queries_per_hour=30,
        hours_per_day=10,
        inter_query_interval_min=15,    # 15 分钟一次
        num_unique_prefixes=10,         # 10 个不同项目
    ),
    CachingScenario(
        name="文档 QA",
        system_prompt_tokens=20000,
        avg_user_message_tokens=100,
        avg_history_tokens=1000,
        queries_per_hour=50,
        hours_per_day=24,
        inter_query_interval_min=5,
        num_unique_prefixes=5,
    ),
    CachingScenario(
        name="批量分析",
        system_prompt_tokens=3000,
        avg_user_message_tokens=2000,
        avg_history_tokens=0,
        queries_per_hour=500,
        hours_per_day=8,
        inter_query_interval_min=0.5,   # 30 秒一次
        num_unique_prefixes=1,
    ),
]


def run_cost_analysis():
    """运行所有场景的成本分析。"""
    pricing = PricingModel()
    
    print("\n" + "=" * 90)
    print("OpenAI Extended Caching 成本分析")
    print("=" * 90)
    
    for scenario in scenarios:
        result = analyze_caching_benefit(scenario, pricing)
        
        print(f"\n{'─' * 60}")
        print(f"场景: {result['scenario']}")
        print(f"{'─' * 60}")
        print(f"  每日查询量: {result['daily_queries']:.0f}")
        print(f"  可缓存 tokens: {result['cacheable_tokens']:,}")
        print(f"  总 input tokens: {result['total_input_tokens']:,}")
        print(f"")
        print(f"  标准缓存命中率: {result['standard_hit_rate']*100:.1f}%")
        print(f"  Extended 命中率: {result['extended_hit_rate']*100:.1f}%")
        print(f"")
        print(f"  无缓存每日成本:     ${result['no_cache_daily_cost']:.2f}")
        print(f"  标准缓存每日成本:   ${result['standard_cache_daily_cost']:.2f} "
              f"(节省 ${result['standard_daily_savings']:.2f}/天)")
        print(f"  Extended 每日成本:  ${result['extended_cache_daily_cost']:.2f} "
              f"(节省 ${result['extended_daily_savings']:.2f}/天)")
        print(f"  Extended 年度节省:  ${result['extended_annual_savings']:.0f}")


if __name__ == "__main__":
    run_cost_analysis()
```

### 4.2 扩展任务

1. 添加 Anthropic 和 Google 的定价模型进行对比
2. 绘制 "缓存命中率 vs 查询间隔" 的曲线图
3. 计算盈亏平衡点：在什么查询频率下 Extended Caching 开始产生正收益？

---

## 练习 5：设计一个多级 KV Cache 系统

**目标：** 在纸面上设计一个完整的多级 KV Cache 系统，综合运用本章所学的所有知识。

### 5.1 设计要求

为一个 LLM 推理服务设计 KV Cache 管理系统，满足以下约束：

**硬件环境：**
- 4 台服务器，每台 8×H100 (80GB)，512 GB CPU DRAM，2×3.84TB NVMe
- InfiniBand HDR (200 Gbps) 连接
- 一个 Redis Cluster (6 节点，共 384 GB 内存)

**业务需求：**
- 模型：Llama-3.1-70B (FP8 部署，权重 ~70 GB)
- 日均 100 万次请求
- 平均 context 长度：8K tokens，最长 128K tokens
- 60% 的请求共享 10 种 system prompt 之一
- p99 TTFT < 2 秒，p99 TPOT < 100 ms

### 5.2 需要回答的问题

请设计并回答以下问题（不需要写代码，写设计文档即可）：

```markdown
## 设计方案

### 1. 存储层级划分
- L0 (GPU HBM): 用于什么？分配多少？
- L1 (CPU DRAM): 用于什么？分配多少？
- L2 (NVMe SSD): 用于什么？分配多少？
- L3 (Redis Cluster): 用于什么？

### 2. 驱逐策略选择
- 选择 LRU 还是 ARC？为什么？
- 高水位和低水位设为多少？

### 3. 路由策略
- 如何将请求路由到合适的服务器？
- 是否使用 hash-based routing？hash key 是什么？

### 4. 预取策略
- 何时触发预取？
- 预取窗口多大？

### 5. 容量规划
- 每台服务器能同时服务多少个请求？
- GPU HBM 中能放多少 KV blocks？
- CPU DRAM 中能放多少？
- 整个集群的 KV Cache 总容量是多少？

### 6. 性能分析
- 最坏情况下的 TTFT 是多少？
- 预取能隐藏多少传输延迟？
- 缓存命中率的目标是多少？

### 7. 成本分析
- 与不使用 offloading 的方案相比，需要多少额外的服务器？
- 投资回报率 (ROI) 是多少？
```

### 5.3 参考计算

以下是一些有用的计算，帮助你完成设计：

```python
# Llama-3.1-70B 的 KV Cache 参数
num_layers = 80
num_kv_heads = 8    # GQA
head_dim = 128
bytes_per_element = 2  # FP16

# 每个 token 的 KV Cache 大小
kv_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_element
print(f"KV per token: {kv_per_token / 1024:.1f} KB")
# → 320 KB

# 8K context 请求的 KV Cache
kv_8k = 8192 * kv_per_token / 1024**3
print(f"KV for 8K context: {kv_8k:.2f} GB")
# → 2.56 GB

# 128K context 请求的 KV Cache
kv_128k = 131072 * kv_per_token / 1024**3
print(f"KV for 128K context: {kv_128k:.2f} GB")
# → 40.96 GB

# H100 可用 HBM (扣除模型权重)
available_hbm = 80 - 70 / 8  # TP=8, 权重约 8.75 GB/卡
print(f"Available HBM per GPU: {available_hbm:.2f} GB")
# → 71.25 GB

# 每张 GPU 能容纳多少个 8K 请求的 KV Cache
requests_per_gpu = available_hbm / (kv_8k / 8)  # TP=8, KV 也分布在 8 卡
print(f"8K requests per GPU (TP=8): {requests_per_gpu:.0f}")
```

---

## 提交与检验

完成以上练习后，确保：

1. **练习 1**：所有测试通过（`test_basic_offload_reload` 和 `test_lru_eviction`）
2. **练习 2**：ARC 实现正确，benchmark 结果合理（ARC 在有扫描负载时应优于 LRU）
3. **练习 3**：生成延迟隐藏效果分析表格，能解释结果
4. **练习 4**：成本分析结果合理，能解释哪些场景适合 Extended Caching
5. **练习 5**：设计文档完整，所有问题都有合理的答案和计算支撑
