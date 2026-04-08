# 动手练习

> 本节包含 5 个实践练习，帮助你深入理解 PagedAttention 的内存管理机制。建议按顺序完成。

---

## 练习 1：Block 分配模拟器

### 目标

实现一个简化版的 Block Pool 和 Block Table，模拟 PagedAttention 的内存管理过程。

### 要求

实现以下类：

```python
from collections import deque
from typing import Dict, List, Optional, Tuple
import numpy as np


class BlockPool:
    """简化版 Block Pool，管理物理 block 的分配和释放。"""
    
    def __init__(self, num_blocks: int, block_size: int):
        """
        Args:
            num_blocks: 物理 block 总数
            block_size: 每个 block 可容纳的 token 数
        """
        self.num_blocks = num_blocks
        self.block_size = block_size
        # TODO: 初始化 free list 和 ref count 数组
    
    def allocate(self, num_blocks: int) -> Optional[List[int]]:
        """分配指定数量的 block，返回 block ID 列表。
        如果空间不足返回 None。"""
        # TODO
        pass
    
    def free(self, block_id: int) -> None:
        """释放一个 block（引用计数 -1）。"""
        # TODO
        pass
    
    def increase_ref_count(self, block_id: int) -> None:
        """增加 block 的引用计数。"""
        # TODO
        pass
    
    def get_num_free_blocks(self) -> int:
        """返回当前空闲 block 数量。"""
        # TODO
        pass


class KVCacheManager:
    """简化版 KV Cache Manager，管理请求级别的 block 分配。"""
    
    def __init__(self, block_pool: BlockPool):
        self.block_pool = block_pool
        self.block_size = block_pool.block_size
        # request_id → List[block_id] 的映射
        self.req_to_blocks: Dict[str, List[int]] = {}
    
    def allocate_slots(self, request_id: str, 
                       num_tokens: int) -> Optional[List[int]]:
        """为请求分配 KV cache 存储空间。
        
        Args:
            request_id: 请求 ID
            num_tokens: 当前请求的总 token 数
        
        Returns:
            新分配的 block ID 列表，或 None 如果分配失败
        """
        # TODO: 
        # 1. 计算需要的总 block 数
        # 2. 减去已有的 block 数
        # 3. 分配差量
        pass
    
    def free(self, request_id: str) -> None:
        """释放请求的所有 block。"""
        # TODO
        pass
    
    def get_block_table(self, request_id: str) -> List[int]:
        """返回请求的 block table。"""
        return self.req_to_blocks.get(request_id, [])
```

### 测试用例

```python
def test_block_allocation():
    pool = BlockPool(num_blocks=10, block_size=16)
    manager = KVCacheManager(pool)
    
    # 1. 请求 A: 42 tokens → 需要 3 个 block
    result = manager.allocate_slots("req_A", num_tokens=42)
    assert result is not None and len(result) == 3
    assert pool.get_num_free_blocks() == 7
    
    # 2. 请求 B: 10 tokens → 需要 1 个 block
    result = manager.allocate_slots("req_B", num_tokens=10)
    assert result is not None and len(result) == 1
    assert pool.get_num_free_blocks() == 6
    
    # 3. 请求 A 继续生成，现在有 50 tokens → 需要 4 个 block (多分配 1 个)
    result = manager.allocate_slots("req_A", num_tokens=50)
    assert result is not None and len(result) == 1  # 新增 1 个 block
    assert pool.get_num_free_blocks() == 5
    
    # 4. 释放请求 B
    manager.free("req_B")
    assert pool.get_num_free_blocks() == 6
    
    # 5. 尝试分配超过容量
    result = manager.allocate_slots("req_C", num_tokens=200)  # 需要 13 个 block
    assert result is None  # 应该失败
    
    print("All tests passed!")

test_block_allocation()
```

### 进阶挑战

- 添加 Copy-on-Write 支持：实现 `fork_request(src_id, dst_id)` 方法，让两个请求共享 block（增加引用计数），并在写入时执行 copy
- 添加碎片率统计：实现 `get_internal_fragmentation()` 方法，计算所有活跃请求的平均内部碎片率

---

## 练习 2：碎片率分析

### 目标

通过模拟不同的工作负载，分析 block_size 对内部碎片率的影响。

### 要求

编写一个模拟脚本，生成不同分布的序列长度，计算各种 block_size 下的碎片率。

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


def calculate_fragmentation(
    seq_lengths: List[int], 
    block_size: int
) -> Tuple[float, float, float]:
    """计算给定序列长度分布下的碎片统计。
    
    Args:
        seq_lengths: 序列长度列表
        block_size: block 大小
    
    Returns:
        (avg_fragmentation_rate, total_wasted_slots, total_allocated_slots)
        avg_fragmentation_rate: 平均碎片率 (wasted / allocated)
        total_wasted_slots: 总浪费 slot 数
        total_allocated_slots: 总分配 slot 数
    """
    # TODO: 实现碎片计算
    pass


def simulate_workloads():
    """模拟不同工作负载，对比碎片率。"""
    
    block_sizes = [1, 4, 8, 16, 32, 64, 128]
    
    # 工作负载 1：聊天场景（短-中等长度）
    chatbot_lengths = np.random.exponential(scale=200, size=1000).astype(int) + 10
    
    # 工作负载 2：文档摘要（长 prompt）
    summary_lengths = np.random.normal(loc=4000, scale=1000, size=1000).astype(int)
    summary_lengths = np.clip(summary_lengths, 500, 8000)
    
    # 工作负载 3：代码生成（变化大）
    code_lengths = np.concatenate([
        np.random.exponential(scale=100, size=500).astype(int) + 20,  # 短请求
        np.random.normal(loc=2000, scale=500, size=500).astype(int),  # 长请求
    ])
    code_lengths = np.clip(code_lengths, 10, 8000)
    
    # 工作负载 4：embedding / 分类（极短）
    embed_lengths = np.random.randint(5, 50, size=1000)
    
    workloads = {
        "Chatbot (avg~200)": chatbot_lengths,
        "Summarization (avg~4000)": summary_lengths,
        "Code Gen (bimodal)": code_lengths,
        "Embedding (avg~25)": embed_lengths,
    }
    
    # TODO: 对每个 workload 和 block_size 计算碎片率
    # TODO: 绘制对比图
    #   X 轴: block_size
    #   Y 轴: 碎片率 (%)
    #   每条线: 一个 workload
    
    pass

simulate_workloads()
```

### 预期观察

1. Embedding 场景下，大 block_size 的碎片率可能高达 50%+
2. 长序列场景下，即使 block_size=64，碎片率也 < 2%
3. block_size=16 在大多数场景下是一个良好的平衡点

---

## 练习 3：Preemption 策略对比实验

### 目标

使用 vLLM 实际测量 swap 和 recompute 两种 preemption 模式的性能差异。

### 环境要求

- 一块 GPU（建议 24GB+）
- 安装 vLLM (`pip install vllm`)
- 一个中等大小的模型（如 `meta-llama/Llama-3.2-1B-Instruct` 或 `Qwen/Qwen2.5-1.5B-Instruct`）

### 实验步骤

**步骤 1：构造显存压力**

```bash
# 启动 vLLM，故意设置较低的 gpu_memory_utilization 以触发 preemption
# 终端 1：使用 recompute 模式
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --gpu-memory-utilization 0.5 \
    --max-model-len 4096 \
    --max-num-seqs 32 \
    --preemption-mode recompute \
    --port 8000

# 终端 2（另一个实验）：使用 swap 模式
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --gpu-memory-utilization 0.5 \
    --max-model-len 4096 \
    --max-num-seqs 32 \
    --preemption-mode swap \
    --swap-space 4 \
    --port 8001
```

**步骤 2：发送并发请求**

```python
import asyncio
import aiohttp
import time
import json
from typing import List, Dict


async def send_request(session: aiohttp.ClientSession, 
                       url: str, prompt: str, 
                       max_tokens: int) -> Dict:
    """发送单个请求并记录延迟。"""
    payload = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    
    start = time.time()
    async with session.post(
        f"{url}/v1/chat/completions", 
        json=payload
    ) as resp:
        result = await resp.json()
        elapsed = time.time() - start
        
    return {
        "elapsed": elapsed,
        "prompt_tokens": result.get("usage", {}).get("prompt_tokens", 0),
        "completion_tokens": result.get("usage", {}).get("completion_tokens", 0),
    }


async def benchmark(url: str, num_requests: int = 50):
    """发送一批并发请求。"""
    # 混合短长请求，制造显存压力
    prompts = []
    for i in range(num_requests):
        if i % 3 == 0:
            # 长 prompt
            prompts.append(("Tell me a very long story about " + "AI " * 200, 512))
        elif i % 3 == 1:
            # 中等 prompt
            prompts.append(("Explain quantum computing in detail.", 256))
        else:
            # 短 prompt
            prompts.append(("Hi!", 128))
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            send_request(session, url, prompt, max_tokens)
            for prompt, max_tokens in prompts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 统计
    successful = [r for r in results if isinstance(r, dict)]
    latencies = [r["elapsed"] for r in successful]
    
    print(f"Successful: {len(successful)}/{num_requests}")
    print(f"Avg latency: {sum(latencies)/len(latencies):.2f}s")
    print(f"P50 latency: {sorted(latencies)[len(latencies)//2]:.2f}s")
    print(f"P99 latency: {sorted(latencies)[int(len(latencies)*0.99)]:.2f}s")
    print(f"Max latency: {max(latencies):.2f}s")


# 运行 benchmark
# asyncio.run(benchmark("http://localhost:8000"))  # recompute
# asyncio.run(benchmark("http://localhost:8001"))  # swap
```

**步骤 3：收集 Prometheus 指标**

```bash
# 检查 preemption 次数
curl -s http://localhost:8000/metrics | grep preemption
curl -s http://localhost:8001/metrics | grep preemption

# 检查 KV cache 使用率
curl -s http://localhost:8000/metrics | grep cache_usage
```

### 分析要点

1. 比较两种模式下的 P50/P99 延迟差异
2. 观察 preemption 发生的频率
3. 思考：在什么条件下 swap 模式明显优于 recompute？

---

## 练习 4：vLLM 源码阅读 Checklist

### 目标

通过阅读 vLLM 源码，回答以下问题。每个问题标注了对应的源码文件。

### 问题清单

请 clone vLLM 仓库 (`git clone https://github.com/vllm-project/vllm.git`) 并回答以下问题：

**BlockPool 相关** (`vllm/v1/core/block_pool.py`)：

- [ ] Q1: `BlockPool` 使用什么数据结构存储引用计数？为什么不用 Python dict？
- [ ] Q2: 当 `enable_caching=True` 时，`free()` 方法中 block 被释放后会去哪里？与 `enable_caching=False` 有什么区别？
- [ ] Q3: Block 的 hash 值是如何计算的？它用于什么目的？

**KVCacheManager 相关** (`vllm/v1/core/kv_cache_manager.py`)：

- [ ] Q4: `allocate_slots()` 方法中，如何判断是否需要分配新的 block（而不是复用最后一个 block 的空 slot）？
- [ ] Q5: `get_computed_blocks()` 返回的 `num_computed_tokens` 是如何计算的？它与 prefix caching 的关系是什么？
- [ ] Q6: 当一个请求完成时，它的 block 立即被释放还是保留在 cache 中？在什么条件下保留？

**BlockTable 相关** (`vllm/v1/worker/block_table.py`)：

- [ ] Q7: `BlockTable` 使用什么格式存储 block 映射？它如何同步到 GPU？
- [ ] Q8: `block_table_np` 和 `block_table_gpu` 之间的同步频率是怎样的？每次 decode step 都同步还是增量同步？

**Scheduler 相关** (`vllm/v1/core/scheduler.py`)：

- [ ] Q9: Scheduler 按什么顺序决定哪个请求被 preempt？为什么选择这个顺序？
- [ ] Q10: 当所有 running 请求都被 preempt 后仍然没有足够的 block，会发生什么？

### 记录格式

建议用以下格式记录答案：

```markdown
## Q1: BlockPool 引用计数的数据结构
**文件**: vllm/v1/core/block_pool.py, 第 XX 行
**答案**: ...
**代码片段**:
    ```python
    # 相关代码
    ```
**为什么**: ...
```

---

## 练习 5：自定义 Block 驱逐策略

### 目标

在练习 1 的 BlockPool 基础上，实现一个支持 Prefix Caching 的 Block Pool，包含可插拔的驱逐策略。

### 要求

```python
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional, List, Dict, Tuple


class EvictionPolicy(ABC):
    """驱逐策略抽象基类。"""
    
    @abstractmethod
    def on_access(self, block_id: int) -> None:
        """block 被访问时调用。"""
        pass
    
    @abstractmethod
    def on_add(self, block_id: int, block_hash: int) -> None:
        """block 加入 cache 时调用。"""
        pass
    
    @abstractmethod
    def evict(self) -> Optional[int]:
        """选择一个 block 驱逐，返回 block_id。
        如果没有可驱逐的 block，返回 None。"""
        pass
    
    @abstractmethod
    def remove(self, block_id: int) -> None:
        """从驱逐策略中移除 block。"""
        pass


class LRUEvictionPolicy(EvictionPolicy):
    """LRU (Least Recently Used) 驱逐策略。"""
    
    def __init__(self):
        # TODO: 使用 OrderedDict 实现 LRU
        pass
    
    def on_access(self, block_id: int) -> None:
        # TODO: 移动到最近使用端
        pass
    
    def on_add(self, block_id: int, block_hash: int) -> None:
        # TODO: 添加到 cache
        pass
    
    def evict(self) -> Optional[int]:
        # TODO: 驱逐最久未使用的 block
        pass
    
    def remove(self, block_id: int) -> None:
        # TODO: 从 LRU 中移除
        pass


class LFUEvictionPolicy(EvictionPolicy):
    """LFU (Least Frequently Used) 驱逐策略。
    
    进阶挑战：实现此策略并与 LRU 对比。
    """
    
    def __init__(self):
        # TODO
        pass
    
    # TODO: 实现所有抽象方法


class CachingBlockPool:
    """支持 Prefix Caching 的 Block Pool。"""
    
    def __init__(self, num_blocks: int, block_size: int,
                 eviction_policy: EvictionPolicy):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.eviction_policy = eviction_policy
        
        # 物理 block 管理
        self.ref_cnts = np.zeros(num_blocks, dtype=np.int32)
        self.free_blocks = deque(range(num_blocks))
        
        # Prefix cache: hash → block_id
        self.hash_to_block: Dict[int, int] = {}
        # block_id → hash（反向映射）
        self.block_to_hash: Dict[int, int] = {}
    
    def allocate(self, num_blocks: int) -> Optional[List[int]]:
        """分配 block，必要时驱逐 cached block。"""
        # TODO:
        # 1. 先尝试从 free list 分配
        # 2. 不够则通过 eviction_policy 驱逐
        # 3. 仍不够则返回 None
        pass
    
    def free(self, block_id: int) -> None:
        """释放 block。ref_count 为 0 时加入 cache 而非 free list。"""
        # TODO
        pass
    
    def get_cached_block(self, block_hash: int) -> Optional[int]:
        """查找 prefix cache，命中则增加引用计数。"""
        # TODO
        pass
    
    def add_to_cache(self, block_id: int, block_hash: int) -> None:
        """将 block 标记为 cached（用于 prefix caching）。"""
        # TODO
        pass
    
    def get_stats(self) -> Dict:
        """返回统计信息。"""
        return {
            "total_blocks": self.num_blocks,
            "free_blocks": len(self.free_blocks),
            "cached_blocks": len(self.hash_to_block),
            "active_blocks": int(np.sum(self.ref_cnts > 0)),
        }
```

### 测试场景

```python
def test_prefix_caching():
    """模拟 prefix caching 场景。"""
    policy = LRUEvictionPolicy()
    pool = CachingBlockPool(num_blocks=20, block_size=16, 
                            eviction_policy=policy)
    
    # 系统 prompt 的 hash（所有请求共享）
    system_prompt_hashes = [hash(f"system_block_{i}") for i in range(3)]
    
    # 模拟多轮对话
    for round_idx in range(10):
        request_id = f"req_{round_idx}"
        
        # 1. 检查 prefix cache 命中
        cached_blocks = []
        for h in system_prompt_hashes:
            block = pool.get_cached_block(h)
            if block is not None:
                cached_blocks.append(block)
                print(f"  Round {round_idx}: Cache HIT for hash {h}")
            else:
                print(f"  Round {round_idx}: Cache MISS for hash {h}")
                break
        
        # 2. 分配新 block
        num_new = 3 - len(cached_blocks) + 2  # 3 system + 2 user
        new_blocks = pool.allocate(num_new)
        assert new_blocks is not None, f"Allocation failed at round {round_idx}"
        
        all_blocks = cached_blocks + new_blocks
        
        # 3. 标记 system prompt blocks 为 cached
        for i, h in enumerate(system_prompt_hashes):
            if i < len(new_blocks):
                pool.add_to_cache(new_blocks[i], h)
        
        # 4. 模拟请求完成，释放 block
        for block_id in all_blocks:
            pool.free(block_id)
        
        print(f"  Round {round_idx}: Stats = {pool.get_stats()}")
    
    print("\nPrefix caching test completed!")

test_prefix_caching()
```

### 进阶挑战

1. 实现 LFU 驱逐策略并对比 LRU，在什么工作负载下 LFU 更优？
2. 实现一个 `SizeAwareLRU` 策略，优先驱逐包含较少 computed token 的 block
3. 添加 cache hit rate 的统计和可视化

---

## 附录：推荐的实验环境

| 组件 | 推荐配置 |
|------|---------|
| GPU | NVIDIA GPU 24GB+（练习 3 需要） |
| Python | 3.10+ |
| vLLM | 0.8.x+ |
| PyTorch | 2.4+ |
| 可视化 | matplotlib, seaborn |

对于没有 GPU 的同学，练习 1、2、4、5 可以纯 CPU 完成（不涉及实际的 GPU 操作）。练习 3 需要至少一块 GPU。
