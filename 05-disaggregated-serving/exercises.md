# 动手练习

> 通过实践加深对 Prefill-Decode 分离架构的理解。

## 练习 1：计算 KV Cache 传输开销

### 目标

学会估算不同模型、不同序列长度、不同网络条件下的 KV Cache 传输开销，建立量化直觉。

### 任务

编写一个 Python 脚本，计算并可视化 KV Cache 传输开销：

```python
"""
练习 1：KV Cache 传输开销计算器

要求：
1. 实现 kv_cache_size() 函数，计算 KV Cache 大小
2. 实现 transfer_time() 函数，计算传输时间
3. 对比不同场景下的传输开销
4. 生成结果表格
"""

from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype_bytes: int = 2  # FP16

@dataclass
class NetworkConfig:
    name: str
    bandwidth_gbps: float  # 有效带宽, GB/s

# 预定义模型配置
MODELS = {
    "Llama-3-8B": ModelConfig("Llama-3-8B", 32, 8, 128),
    "Llama-3-70B": ModelConfig("Llama-3-70B", 80, 8, 128),
    "Llama-3-405B": ModelConfig("Llama-3-405B", 126, 16, 128),
    "Qwen-2.5-72B": ModelConfig("Qwen-2.5-72B", 80, 8, 128),
    "DeepSeek-V3": ModelConfig("DeepSeek-V3", 61, 1, 512, 2),  # MLA: 特殊 KV head 配置
}

NETWORKS = {
    "NVLink-H100": NetworkConfig("NVLink-H100", 450),
    "NVLink-A100": NetworkConfig("NVLink-A100", 300),
    "IB-NDR-400G": NetworkConfig("IB-NDR-400G", 46),
    "IB-HDR-200G": NetworkConfig("IB-HDR-200G", 23),
    "RoCE-100G": NetworkConfig("RoCE-100G", 11),
    "TCP-25G": NetworkConfig("TCP-25G", 2.8),
}

def kv_cache_size(model: ModelConfig, seq_len: int) -> float:
    """
    计算 KV Cache 大小 (bytes)
    
    公式: size = 2 (K+V) × num_layers × num_kv_heads × head_dim × seq_len × dtype_bytes
    
    TODO: 实现此函数
    """
    pass

def transfer_time_ms(cache_size_bytes: float, network: NetworkConfig) -> float:
    """
    计算传输时间 (毫秒)
    
    TODO: 实现此函数
    """
    pass

def main():
    seq_lengths = [512, 1024, 4096, 16384, 32768, 131072]
    
    print("=" * 80)
    print("KV Cache Size (MB)")
    print("=" * 80)
    print(f"{'Model':<20}", end="")
    for sl in seq_lengths:
        print(f"  {sl:>8}", end="")
    print()
    print("-" * 80)
    
    for name, model in MODELS.items():
        print(f"{name:<20}", end="")
        for sl in seq_lengths:
            size = kv_cache_size(model, sl)
            print(f"  {size / 1e6:>8.1f}", end="")
        print()
    
    print()
    print("=" * 80)
    print("Transfer Time (ms) — Llama-3-70B")
    print("=" * 80)
    model = MODELS["Llama-3-70B"]
    print(f"{'Network':<20}", end="")
    for sl in seq_lengths:
        print(f"  {sl:>8}", end="")
    print()
    print("-" * 80)
    
    for net_name, network in NETWORKS.items():
        print(f"{net_name:<20}", end="")
        for sl in seq_lengths:
            size = kv_cache_size(model, sl)
            time = transfer_time_ms(size, network)
            print(f"  {time:>8.1f}", end="")
        print()
    
    # 扩展练习: 
    # 1. 添加流水线传输的时间计算（假设计算和传输可重叠）
    # 2. 添加 FP8 量化后的传输时间对比
    # 3. 计算"分离值得"的临界 prompt 长度

if __name__ == "__main__":
    main()
```

### 预期输出

运行脚本后应该看到类似以下的表格：

```
KV Cache Size (MB)
Model                    512     1024     4096    16384    32768   131072
Llama-3-8B               8.0     16.0     64.0    256.0    512.0   2048.0
Llama-3-70B             20.0     40.0    160.0    640.0   1280.0   5120.0
...

Transfer Time (ms) — Llama-3-70B
Network                  512     1024     4096    16384    32768   131072
NVLink-H100              0.0      0.1      0.4      1.4      2.8     11.4
IB-NDR-400G              0.4      0.9      3.5     13.9     27.8    111.3
...
```

### 思考题

1. DeepSeek-V3 使用 MLA（Multi-head Latent Attention），它的 KV Cache 大小与 Llama-3-70B 相比如何？为什么？
2. 如果使用 FP8 量化 KV Cache 再传输，传输时间减少多少？这对 decode 质量有什么影响？
3. 对于 128K context 的场景，哪种网络条件下分离架构才值得使用？

---

## 练习 2：模拟 Prefill-Decode 调度

### 目标

通过编写一个简化的调度模拟器，理解分离架构中 prefill 和 decode 的调度交互，以及 P:D 比例对吞吐和延迟的影响。

### 任务

```python
"""
练习 2：Prefill-Decode 分离调度模拟器

模拟一个简化的分离架构，分析不同 P:D 比例下的性能表现。

要求：
1. 实现请求到达（泊松过程）
2. 实现 Prefill Queue 和 Decode Queue
3. 实现 KV Transfer 延迟
4. 统计 TTFT, TPOT, 吞吐, GPU 利用率
5. 对比不同 P:D 比例的结果
"""

import heapq
import random
from dataclasses import dataclass, field

@dataclass
class Request:
    id: int
    arrival_time: float
    prompt_length: int
    output_length: int
    # 时间戳记录
    prefill_start: float = 0
    prefill_end: float = 0
    transfer_end: float = 0
    decode_start: float = 0
    decode_end: float = 0

@dataclass(order=True)
class Event:
    time: float
    event_type: str = field(compare=False)
    data: dict = field(compare=False, default_factory=dict)

class DisaggSimulator:
    """分离架构调度模拟器"""
    
    def __init__(
        self,
        num_prefill_gpus: int,
        num_decode_gpus: int,
        prefill_speed: float = 40000,     # tokens/s per GPU
        decode_speed: float = 100,         # tokens/s per GPU per sequence
        max_decode_batch: int = 64,        # 每 GPU 最大并发 decode 序列
        kv_transfer_bw: float = 46.0,      # GB/s (IB NDR)
        model_kv_bytes_per_token: float = 40960,  # bytes per token (70B model)
    ):
        self.num_prefill = num_prefill_gpus
        self.num_decode = num_decode_gpus
        self.prefill_speed = prefill_speed
        self.decode_speed = decode_speed
        self.max_decode_batch = max_decode_batch
        self.kv_transfer_bw = kv_transfer_bw
        self.kv_bytes_per_token = model_kv_bytes_per_token
        
        # 状态
        self.prefill_busy = [False] * num_prefill_gpus
        self.decode_active = [0] * num_decode_gpus  # 每 GPU 活跃序列数
        
        # 队列
        self.event_queue: list[Event] = []
        self.prefill_wait_queue: list[Request] = []
        self.decode_wait_queue: list[Request] = []
        
        # 结果
        self.completed_requests: list[Request] = []
    
    def generate_workload(
        self,
        num_requests: int,
        arrival_rate: float,        # requests per second
        prompt_len_range: tuple,    # (min, max)
        output_len_range: tuple,    # (min, max)
    ) -> list[Request]:
        """生成工作负载"""
        requests = []
        current_time = 0
        
        for i in range(num_requests):
            # TODO: 使用指数分布生成请求到达间隔
            inter_arrival = 0  # 替换为泊松过程
            current_time += inter_arrival
            
            req = Request(
                id=i,
                arrival_time=current_time,
                prompt_length=random.randint(*prompt_len_range),
                output_length=random.randint(*output_len_range),
            )
            requests.append(req)
        
        return requests
    
    def simulate(self, requests: list[Request]) -> dict:
        """
        运行模拟
        
        TODO: 实现以下逻辑:
        1. 将请求到达事件加入 event_queue
        2. 事件驱动循环:
           - REQUEST_ARRIVE: 请求到达，加入 prefill_wait_queue
           - PREFILL_START: 开始 prefill（如果有空闲 prefill GPU）
           - PREFILL_END: prefill 完成，开始 KV transfer
           - TRANSFER_END: 传输完成，加入 decode_wait_queue
           - DECODE_START: 开始 decode（如果有容量）
           - DECODE_END: decode 完成，释放资源
        3. 统计指标
        """
        pass
    
    def compute_metrics(self) -> dict:
        """计算性能指标"""
        ttfts = []
        tpots = []
        
        for req in self.completed_requests:
            # TTFT = 从到达到第一个 token 生成
            ttft = req.decode_start - req.arrival_time
            ttfts.append(ttft)
            
            # TPOT = decode 总时间 / output token 数
            decode_time = req.decode_end - req.decode_start
            tpot = decode_time / req.output_length if req.output_length > 0 else 0
            tpots.append(tpot)
        
        total_time = max(r.decode_end for r in self.completed_requests) - min(r.arrival_time for r in self.completed_requests)
        throughput = len(self.completed_requests) / total_time
        
        return {
            "avg_ttft_ms": sum(ttfts) / len(ttfts) * 1000,
            "p99_ttft_ms": sorted(ttfts)[int(len(ttfts) * 0.99)] * 1000,
            "avg_tpot_ms": sum(tpots) / len(tpots) * 1000,
            "p99_tpot_ms": sorted(tpots)[int(len(tpots) * 0.99)] * 1000,
            "throughput_rps": throughput,
        }

def compare_ratios():
    """对比不同 P:D 比例"""
    total_gpus = 8
    
    print(f"{'P:D':<8} {'Avg TTFT(ms)':<14} {'P99 TTFT(ms)':<14} "
          f"{'Avg TPOT(ms)':<14} {'Throughput':<12}")
    print("-" * 70)
    
    for n_prefill in range(1, total_gpus):
        n_decode = total_gpus - n_prefill
        
        sim = DisaggSimulator(
            num_prefill_gpus=n_prefill,
            num_decode_gpus=n_decode,
        )
        
        requests = sim.generate_workload(
            num_requests=500,
            arrival_rate=20,
            prompt_len_range=(4000, 12000),
            output_len_range=(100, 300),
        )
        
        sim.simulate(requests)
        metrics = sim.compute_metrics()
        
        print(f"{n_prefill}:{n_decode:<6} "
              f"{metrics['avg_ttft_ms']:<14.1f} "
              f"{metrics['p99_ttft_ms']:<14.1f} "
              f"{metrics['avg_tpot_ms']:<14.1f} "
              f"{metrics['throughput_rps']:<12.1f}")

if __name__ == "__main__":
    compare_ratios()
```

### 扩展挑战

1. 添加 **混合部署** 模拟器作为 baseline，对比分离 vs 混合的性能差异
2. 实现 **动态 P:D 调整**：当 prefill 队列积压时，将 decode GPU 临时切换为 prefill
3. 添加 **流水线 KV Transfer**：传输与 prefill 计算重叠
4. 模拟 **Prefill Worker 故障**：一个 prefill GPU 突然不可用，观察系统行为

---

## 练习 3：搭建 vLLM Disaggregated Serving

### 目标

在本地或云环境中实际搭建一个 vLLM 分离部署，体验完整的部署和测试流程。

### 前置条件

- 至少 2 个 GPU（可以是同一台机器上的）
- vLLM >= 0.8.0
- Python >= 3.10

### Step 1：安装 vLLM

```bash
pip install vllm>=0.8.0
```

### Step 2：启动 Prefill 节点

```bash
# 终端 1: Prefill Worker (GPU 0)
CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8100 \
    --kv-transfer-config '{
        "kv_connector": "P2pNcclConnector",
        "kv_role": "kv_producer",
        "kv_parallel_size": 1,
        "kv_port": 14579
    }' \
    --max-model-len 8192 \
    --enforce-eager
```

### Step 3：启动 Decode 节点

```bash
# 终端 2: Decode Worker (GPU 1)
CUDA_VISIBLE_DEVICES=1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8200 \
    --kv-transfer-config '{
        "kv_connector": "P2pNcclConnector",
        "kv_role": "kv_consumer",
        "kv_parallel_size": 1,
        "kv_port": 14579
    }' \
    --max-model-len 8192 \
    --enforce-eager
```

### Step 4：启动 Router

```bash
# 终端 3: Disaggregated Router
vllm serve \
    --disagg-config '{
        "prefill_urls": ["http://localhost:8100"],
        "decode_urls": ["http://localhost:8200"]
    }' \
    --port 8000
```

### Step 5：发送请求测试

```python
"""测试 disaggregated serving"""
import openai
import time

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",
)

# 测试 1: 基本功能验证
print("=== 功能测试 ===")
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello, explain what is KV cache in LLM inference."}],
    max_tokens=200,
)
print(response.choices[0].message.content)

# 测试 2: 流式输出
print("\n=== 流式测试 ===")
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Write a haiku about AI."}],
    max_tokens=50,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()

# 测试 3: 长 prompt 性能测试
print("\n=== 长 Prompt 性能测试 ===")
long_prompt = "Explain the following concept in detail: " + "token " * 2000
start = time.time()
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": long_prompt}],
    max_tokens=100,
)
elapsed = time.time() - start
print(f"长 prompt (2000 tokens) 响应时间: {elapsed:.2f}s")
print(f"TTFT (估算): {elapsed - len(response.choices[0].message.content.split()) * 0.03:.2f}s")
```

### Step 6：性能对比测试

```bash
# 对比混合部署 vs 分离部署

# 混合部署 baseline (2 GPU)
CUDA_VISIBLE_DEVICES=0,1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 8192

# 使用 benchmark 脚本对比
python -m vllm.entrypoints.openai.run_batch \
    --input-file benchmark_requests.jsonl \
    --output-file results.jsonl
```

### 验收标准

- [ ] Prefill 和 Decode 节点均启动成功，日志中看到 KV connector 初始化信息
- [ ] 通过 Router 发送请求能正常获得响应
- [ ] 流式输出正常工作
- [ ] 对比混合部署，在长 prompt 场景下 TTFT 有改善

---

## 练习 4：实现自定义 KV Connector

### 目标

通过实现一个简化的 KV Connector，深入理解 vLLM 的 KV Transfer 框架。

### 任务

实现一个基于共享内存 (Shared Memory) 的 KV Connector，适用于同机多进程场景：

```python
"""
练习 4：实现一个基于共享内存的 KV Connector

这个 connector 使用 Python 的 multiprocessing.shared_memory
在同机的 prefill 和 decode 进程间传输 KV Cache。

虽然性能不如 NIXL 或 NCCL，但可以帮助理解 connector 的接口设计。
"""

import torch
import numpy as np
from multiprocessing import shared_memory
from typing import Optional
from dataclasses import dataclass

@dataclass
class ShmKVBlock:
    """共享内存中的一个 KV Cache block"""
    shm_name: str
    shape: tuple
    dtype: str
    ready: bool = False

class SharedMemoryKVConnector:
    """
    基于共享内存的 KV Cache Connector (简化版)
    
    实现提示:
    1. 注册 KV Cache: 记录每层 KV Cache 的 shape 和 dtype
    2. 发送: 将 GPU tensor 拷贝到 CPU，再写入共享内存
    3. 接收: 从共享内存读取数据，拷贝回 GPU
    4. 使用文件锁或信号量做同步
    """
    
    def __init__(self, role: str, num_layers: int, block_size: int):
        """
        Args:
            role: "producer" (prefill) 或 "consumer" (decode)
            num_layers: 模型层数
            block_size: 每个 KV Cache block 的大小
        """
        self.role = role
        self.num_layers = num_layers
        self.block_size = block_size
        self.shm_blocks: dict[str, shared_memory.SharedMemory] = {}
    
    def register_kv_caches(self, kv_caches: list[tuple[torch.Tensor, torch.Tensor]]):
        """
        注册 KV Cache
        
        TODO: 
        - 记录每层 KV Cache 的元信息
        - producer: 创建共享内存段
        - consumer: 连接到已有的共享内存段
        """
        pass
    
    def send_kv_layer(
        self,
        layer_idx: int,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_ids: list[int],
        request_id: str,
    ):
        """
        发送一层的 KV Cache 到共享内存
        
        TODO:
        1. 从 GPU tensor 中提取指定 block_ids 的数据
        2. 将数据拷贝到 CPU (tensor.cpu())
        3. 写入共享内存
        4. 设置 ready 标志
        """
        pass
    
    def recv_kv_layer(
        self,
        layer_idx: int,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_ids: list[int],
        request_id: str,
    ):
        """
        从共享内存接收一层的 KV Cache
        
        TODO:
        1. 等待 ready 标志
        2. 从共享内存读取数据
        3. 将数据拷贝回 GPU (tensor.cuda())
        4. 写入到指定的 block_ids 位置
        """
        pass
    
    def cleanup(self):
        """释放共享内存"""
        for name, shm in self.shm_blocks.items():
            shm.close()
            if self.role == "producer":
                shm.unlink()

# 测试代码
def test_shm_connector():
    """测试共享内存 connector"""
    import multiprocessing as mp
    
    num_layers = 4
    num_heads = 8
    head_dim = 64
    block_size = 16
    num_blocks = 10
    
    def producer_fn():
        connector = SharedMemoryKVConnector("producer", num_layers, block_size)
        
        # 模拟 KV Cache
        kv_caches = []
        for _ in range(num_layers):
            k = torch.randn(num_blocks, num_heads, block_size, head_dim, device="cuda:0")
            v = torch.randn(num_blocks, num_heads, block_size, head_dim, device="cuda:0")
            kv_caches.append((k, v))
        
        connector.register_kv_caches(kv_caches)
        
        # 发送 block 0, 1, 2
        for layer_idx in range(num_layers):
            k, v = kv_caches[layer_idx]
            connector.send_kv_layer(layer_idx, k, v, [0, 1, 2], "req_001")
        
        print("[Producer] 所有层的 KV Cache 已发送")
    
    def consumer_fn():
        import time
        time.sleep(1)  # 等 producer 先启动
        
        connector = SharedMemoryKVConnector("consumer", num_layers, block_size)
        
        # 分配接收 buffer
        kv_caches = []
        for _ in range(num_layers):
            k = torch.zeros(num_blocks, num_heads, block_size, head_dim, device="cuda:0")
            v = torch.zeros(num_blocks, num_heads, block_size, head_dim, device="cuda:0")
            kv_caches.append((k, v))
        
        connector.register_kv_caches(kv_caches)
        
        # 接收
        for layer_idx in range(num_layers):
            k, v = kv_caches[layer_idx]
            connector.recv_kv_layer(layer_idx, k, v, [0, 1, 2], "req_001")
        
        print("[Consumer] 所有层的 KV Cache 已接收")
        
        # 验证数据正确性
        # TODO: 与 producer 端对比数据是否一致
    
    p1 = mp.Process(target=producer_fn)
    p2 = mp.Process(target=consumer_fn)
    p1.start()
    p2.start()
    p1.join()
    p2.join()

if __name__ == "__main__":
    test_shm_connector()
```

### 扩展挑战

1. 添加**传输带宽统计**：记录每次传输的数据量和耗时，计算有效带宽
2. 实现**流水线传输**：producer 逐层发送，consumer 逐层接收，传输和计算重叠
3. 对比 Shared Memory vs CUDA IPC（`torch.multiprocessing` 的 GPU 直接共享）的性能
4. 将你的 connector 集成到 vLLM 的 connector 框架中（继承 `KVConnectorBase_V1`）

---

## 练习 5：分析生产工作负载并做出分离决策

### 目标

给定一组真实的（模拟的）生产流量数据，分析是否应该使用分离架构，并确定最优的 P:D 比例。

### 任务

```python
"""
练习 5：生产工作负载分析与分离决策

给定一天的请求日志，分析工作负载特征，做出分离架构决策。
"""

import json
import random
import math
from collections import defaultdict

def generate_production_log(num_requests: int = 10000) -> list[dict]:
    """
    生成模拟的生产请求日志
    
    模拟一个混合工作负载：
    - 60% RAG 请求（长 prompt, 短 output）
    - 25% 对话请求（中等 prompt, 中等 output）
    - 15% 创意写作（短 prompt, 长 output）
    """
    log = []
    
    for i in range(num_requests):
        r = random.random()
        
        if r < 0.6:
            # RAG
            entry = {
                "request_id": f"req_{i:06d}",
                "type": "rag",
                "prompt_tokens": random.randint(4000, 16000),
                "output_tokens": random.randint(50, 300),
                "timestamp": i * 0.05 + random.gauss(0, 0.01),
            }
        elif r < 0.85:
            # 对话
            entry = {
                "request_id": f"req_{i:06d}",
                "type": "chat",
                "prompt_tokens": random.randint(200, 2000),
                "output_tokens": random.randint(200, 1000),
                "timestamp": i * 0.05 + random.gauss(0, 0.01),
            }
        else:
            # 创意写作
            entry = {
                "request_id": f"req_{i:06d}",
                "type": "creative",
                "prompt_tokens": random.randint(50, 300),
                "output_tokens": random.randint(1000, 4000),
                "timestamp": i * 0.05 + random.gauss(0, 0.01),
            }
        
        log.append(entry)
    
    return log

def analyze_workload(log: list[dict]) -> dict:
    """
    分析工作负载特征
    
    TODO: 计算以下指标:
    1. 各类请求的占比
    2. 平均/P50/P99 prompt 长度
    3. 平均/P50/P99 output 长度
    4. 平均 POR (Prompt-Output Ratio)
    5. 请求到达率 (QPS)
    6. 预估的 prefill 计算量和 decode 计算量
    """
    pass

def make_decision(analysis: dict, infra: dict) -> dict:
    """
    基于分析结果做出分离决策
    
    TODO:
    1. 计算 POR，判断是否适合分离
    2. 估算传输开销
    3. 估算最优 P:D 比例
    4. 计算预期的吞吐提升
    5. 给出最终建议
    
    Args:
        analysis: analyze_workload 的结果
        infra: 基础设施信息，如:
            {
                "total_gpus": 16,
                "gpu_type": "H100",
                "network": "IB-NDR-400G",
                "model": "Llama-3-70B",
            }
    """
    pass

def main():
    # 生成日志
    log = generate_production_log(10000)
    
    # 分析工作负载
    analysis = analyze_workload(log)
    
    print("=" * 60)
    print("工作负载分析报告")
    print("=" * 60)
    # TODO: 打印分析结果
    
    # 基础设施信息
    infra = {
        "total_gpus": 16,
        "gpu_type": "H100",
        "network": "IB-NDR-400G",
        "model": "Llama-3-70B",
        "model_layers": 80,
        "model_kv_heads": 8,
        "model_head_dim": 128,
    }
    
    # 做出决策
    decision = make_decision(analysis, infra)
    
    print("\n" + "=" * 60)
    print("分离架构决策")
    print("=" * 60)
    # TODO: 打印决策结果
    
    # 扩展:
    # 1. 按小时分析流量模式，看是否需要动态调整 P:D 比例
    # 2. 分别分析 RAG、对话、创意写作三类请求，
    #    看是否需要对不同请求类型采用不同的路由策略
    # 3. 计算分离部署的 TCO 并与混合部署对比

if __name__ == "__main__":
    main()
```

### 预期交付

1. **分析报告**：包含工作负载特征、POR 分布、QPS 分析
2. **决策结论**：是否应该分离，推荐的 P:D 比例
3. **预期收益**：吞吐提升百分比、延迟改善、成本节省

### 思考题

1. 如果工作负载中 RAG 占比从 60% 变成 30%，决策会改变吗？
2. 如果网络从 IB NDR 降级到 RoCE 100G，分离还值得吗？
3. 如何设计一个系统，自动根据流量特征决定是否启用分离架构？

---

## 总结

| 练习 | 核心技能 | 难度 |
|------|---------|------|
| 练习 1 | 量化分析 KV Cache 传输开销 | ⭐⭐ |
| 练习 2 | 理解分离调度的交互逻辑 | ⭐⭐⭐ |
| 练习 3 | 实际搭建 vLLM 分离部署 | ⭐⭐⭐ |
| 练习 4 | 深入理解 KV Connector 框架 | ⭐⭐⭐⭐ |
| 练习 5 | 生产决策分析能力 | ⭐⭐⭐ |

完成这些练习后，你应该能够：
- 量化评估分离架构在特定场景下的收益
- 搭建和调试 vLLM 的分离部署
- 理解 KV Transfer 框架的设计并能扩展
- 基于真实工作负载做出合理的架构决策
