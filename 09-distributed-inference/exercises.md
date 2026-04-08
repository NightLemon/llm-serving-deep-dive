# 分布式推理动手练习

> 通过以下练习巩固 Tensor Parallel、Pipeline Parallel、Expert Parallel、Data Parallel、Context Parallel 以及多维并行组合的核心概念。

---

## 练习 1: TP 通信开销分析与性能建模

**目标**：理解 Tensor Parallel 的通信开销如何随配置变化，建立性能预估直觉。

### 任务

编写一个 Python 脚本，计算并可视化不同 TP 配置下的 decode 延迟分解（权重读取、KV Cache 读取、TP 通信）。

```python
"""
TP 通信开销分析工具

要求:
1. 实现 decode 延迟分解函数，输入模型参数和硬件参数，
   输出权重读取时间、KV Cache 读取时间、TP AllReduce 时间
2. 对比以下场景:
   - Llama-3-8B:  TP=1,2
   - Llama-3-70B: TP=2,4,8
   - Llama-3-405B: TP=8 + PP=1,2,4
3. 输出表格和图表，展示:
   - 各组件延迟占比 (stacked bar chart)
   - TP 效率 (relative speedup vs ideal)
   - 通信/计算比随 batch_size 的变化曲线
"""

import dataclasses

@dataclasses.dataclass
class ModelConfig:
    name: str
    params_B: float          # 参数量 (B)
    num_layers: int
    hidden_size: int
    num_kv_heads: int
    head_dim: int
    
@dataclasses.dataclass
class HardwareConfig:
    name: str
    hbm_bw_TBs: float       # HBM 带宽 (TB/s)
    nvlink_bw_GBs: float    # NVLink 单向带宽 (GB/s)
    allreduce_latency_us: float  # AllReduce 固定延迟 (μs)


# TODO: 实现以下函数

def decode_latency_breakdown(
    model: ModelConfig,
    hw: HardwareConfig,
    tp_size: int,
    pp_size: int,
    batch_size: int,
    seq_len: int,
    dtype_bytes: int = 2,
) -> dict:
    """
    计算 decode 阶段单步延迟分解
    
    返回:
      {
        "weight_read_ms": float,     # 权重从 HBM 读取的时间
        "kv_read_ms": float,         # KV Cache 从 HBM 读取的时间
        "tp_comm_ms": float,         # TP AllReduce 通信时间
        "pp_comm_ms": float,         # PP P2P 通信时间
        "total_ms": float,           # 总延迟
        "tp_efficiency": float,      # TP 效率 (%)
        "comm_compute_ratio": float, # 通信/计算比
      }
    """
    pass


def run_analysis():
    """运行分析并输出结果"""
    
    models = [
        ModelConfig("Llama-3-8B", 8, 32, 4096, 8, 128),
        ModelConfig("Llama-3-70B", 70, 80, 8192, 8, 128),
        ModelConfig("Llama-3-405B", 405, 126, 16384, 8, 128),
    ]
    
    h100 = HardwareConfig("H100", 3.35, 450, 5)
    a100 = HardwareConfig("A100", 2.0, 300, 8)
    
    # TODO:
    # 1. 对每个模型, 遍历 TP=1,2,4,8, PP=1,2,4
    # 2. 对 batch_size=1,8,32,128,256 分别计算
    # 3. 输出格式化表格
    # 4. (可选) 用 matplotlib 画图
    pass


if __name__ == "__main__":
    run_analysis()
```

### 思考题

1. 对于 Llama-3-70B，在 H100 上 TP=4 和 TP=8 哪个 decode 延迟更低？在什么 batch size 下结论会反转？
2. 为什么 decode 阶段 TP=8 的 PCIe 系统效率只有 50-65%？写出具体的计算过程。
3. 如果用 FP8 量化（dtype_bytes=1），TP 的最优选择会如何变化？

---

## 练习 2: Pipeline Parallel Bubble 模拟

**目标**：通过模拟理解 PP 在推理中的 bubble 行为和 interleaved scheduling 的效果。

### 任务

```python
"""
PP 推理调度模拟器

要求:
1. 实现一个简单的 PP 推理模拟器
2. 模拟 continuous batching 场景下的 pipeline 行为
3. 计算 bubble 率和 GPU 利用率
4. 对比 naive 调度和 interleaved 调度的效果
"""

from typing import List, Tuple
import heapq

@dataclasses.dataclass
class Request:
    id: int
    prompt_len: int       # prompt token 数
    output_len: int       # 生成 token 数
    arrival_time_ms: float

@dataclasses.dataclass 
class StageEvent:
    """一个 stage 上的计算事件"""
    start_ms: float
    end_ms: float
    request_id: int
    event_type: str       # "prefill" or "decode"
    stage_id: int


class PPSimulator:
    """Pipeline Parallel 推理模拟器"""
    
    def __init__(
        self,
        pp_size: int,
        per_layer_compute_ms: float,   # 每层每 token 的计算时间
        num_layers: int,
        inter_stage_comm_ms: float,    # stage 间通信延迟
    ):
        self.pp_size = pp_size
        self.layers_per_stage = num_layers // pp_size
        self.per_stage_compute_ms = per_layer_compute_ms * self.layers_per_stage
        self.inter_stage_comm_ms = inter_stage_comm_ms
        self.events: List[StageEvent] = []
    
    def simulate_naive(self, requests: List[Request]) -> dict:
        """
        Naive 调度: 每个 request 串行经过所有 stage
        
        TODO: 实现模拟逻辑
        返回:
          {
            "events": List[StageEvent],
            "total_time_ms": float,
            "bubble_ratio": float,       # 所有 stage 的空闲时间占比
            "gpu_utilization": float,    # GPU 利用率
            "avg_ttft_ms": float,        # 平均 TTFT
            "avg_tpot_ms": float,        # 平均 TPOT
          }
        """
        pass
    
    def simulate_interleaved(self, requests: List[Request]) -> dict:
        """
        Interleaved 调度: 多个 request 交错执行
        Stage i 在等待上游时可以处理其他 request
        
        TODO: 实现模拟逻辑
        """
        pass
    
    def compute_bubble_ratio(self, events: List[StageEvent], total_time_ms: float) -> float:
        """
        计算 bubble 率
        bubble = stage 空闲时间 / stage 总时间
        """
        pass


def run_simulation():
    """运行模拟"""
    
    # 场景: Llama-3-70B, PP=4, 80 层 → 每 stage 20 层
    sim = PPSimulator(
        pp_size=4,
        per_layer_compute_ms=0.1,      # 100 μs/layer/token (decode)
        num_layers=80,
        inter_stage_comm_ms=0.05,      # 50 μs inter-stage
    )
    
    # 生成模拟请求
    requests = [
        Request(i, prompt_len=500, output_len=200, arrival_time_ms=i*10)
        for i in range(20)
    ]
    
    # TODO:
    # 1. 运行 naive 和 interleaved 两种调度
    # 2. 对比 bubble ratio, GPU utilization, 平均延迟
    # 3. 画 Gantt 图展示 pipeline 行为
    pass
```

### 思考题

1. PP=2 vs PP=4 下，bubble ratio 分别是多少？假设有 10 个请求同时到达。
2. Interleaved 调度相比 naive 调度，在 decode 阶段的 TPOT 有什么变化？吞吐呢？
3. 如果 prefill 和 decode 的计算时间差异很大（比如 prefill 是 decode 的 100 倍），PP 的 bubble 行为会如何变化？

---

## 练习 3: MoE Expert 负载均衡模拟

**目标**：理解 MoE 推理中 Expert 负载不均衡的影响和 EPLB 的效果。

### 任务

```python
"""
MoE Expert 负载均衡模拟

要求:
1. 模拟不同 routing 分布下的 Expert 负载
2. 实现三种负载均衡策略: Static EP, Random Replication, EPLB
3. 计算各策略下的:
   - 最大负载 GPU 的等待时间
   - 总体 GPU 利用率
   - All-to-All 通信量
"""

import numpy as np

class MoELoadSimulator:
    """MoE Expert 负载模拟器"""
    
    def __init__(
        self,
        num_experts: int,       # Expert 数量
        ep_size: int,           # EP size
        top_k: int,             # 每 token 选择的 expert 数
        num_tokens: int,        # Batch 中的 token 数
    ):
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.top_k = top_k
        self.num_tokens = num_tokens
        self.experts_per_gpu = num_experts // ep_size
    
    def generate_routing(self, distribution: str = "uniform") -> np.ndarray:
        """
        生成 routing 矩阵
        
        返回: routing[i][j] = token i 的第 j 个选择的 expert id
        形状: [num_tokens, top_k]
        
        distribution:
        - "uniform": 均匀分布
        - "zipf": Zipf 分布 (少数 expert 非常热门)
        - "clustered": 聚类分布 (某些 expert 组合经常一起出现)
        
        TODO: 实现三种分布
        """
        pass
    
    def compute_load_static(self, routing: np.ndarray) -> dict:
        """
        Static EP: 固定分配 expert 到 GPU
        Expert 0~(E/N-1) → GPU 0, Expert (E/N)~(2E/N-1) → GPU 1, ...
        
        返回:
          {
            "load_per_gpu": List[int],    # 每个 GPU 处理的 token 数
            "max_load": int,
            "min_load": int,
            "imbalance_ratio": float,     # max / mean
            "gpu_utilization": float,     # mean / max
          }
        
        TODO: 实现
        """
        pass
    
    def compute_load_eplb(self, routing: np.ndarray, history: list) -> dict:
        """
        EPLB: 基于历史负载重新分配 expert
        
        策略:
        1. 统计过去 N 步的 expert 负载
        2. 将高负载 expert 复制到多个 GPU (冗余)
        3. 合并低负载 expert
        
        TODO: 实现
        """
        pass
    
    def compute_all_to_all_volume(self, routing: np.ndarray) -> dict:
        """
        计算 All-to-All 通信量
        
        返回:
          {
            "total_bytes": int,
            "cross_node_bytes": int,       # 假设每节点 8 GPU
            "max_per_pair_bytes": int,      # 最大单对 GPU 间通信量
            "comm_matrix": np.ndarray,      # [ep_size, ep_size] 通信矩阵
          }
        
        TODO: 实现
        """
        pass


def run_experiment():
    """运行实验"""
    
    sim = MoELoadSimulator(
        num_experts=256,
        ep_size=32,
        top_k=8,
        num_tokens=4096,
    )
    
    # TODO:
    # 1. 对三种分布分别生成 routing
    # 2. 计算 Static EP, EPLB 的负载分布
    # 3. 输出对比表格:
    #    - 分布类型 × 策略 → imbalance ratio, GPU utilization
    # 4. 画负载分布直方图
    # 5. 画 All-to-All 通信矩阵热力图
    pass
```

### 思考题

1. 当 routing 分布是 Zipf 分布（α=1.5）时，Static EP 的 GPU 利用率大约是多少？EPLB 能提升多少？
2. 如果我们允许最多 2 倍的 expert 冗余（最多复制每个 expert 2 份），理论上能将 Zipf 分布的 imbalance ratio 从多少降到多少？
3. All-to-All 通信矩阵在 Zipf 分布下是什么特征？这对网络设计有什么启示？

---

## 练习 4: 并行策略选择器

**目标**：构建一个完整的并行策略推荐工具，给定模型和硬件配置，输出最优的 TP/PP/EP/DP/CP 组合。

### 任务

```python
"""
并行策略选择器

要求:
1. 实现完整的策略选择逻辑
2. 支持 Dense 和 MoE 模型
3. 支持多种硬件配置
4. 输出推荐配置、预估延迟、预估吞吐
5. 输出多个候选方案供用户选择
"""

@dataclasses.dataclass
class DeploymentConfig:
    """部署配置"""
    tp: int
    pp: int
    ep: int
    dp: int
    cp: int
    
    @property
    def total_gpus(self) -> int:
        return self.tp * self.pp * self.ep * self.dp * self.cp
    
    @property
    def gpus_per_replica(self) -> int:
        return self.tp * self.pp * self.ep * self.cp


@dataclasses.dataclass
class PerformanceEstimate:
    """性能预估"""
    ttft_ms: float           # Time to First Token
    tpot_ms: float           # Time Per Output Token
    max_batch_size: int      # 最大并发请求数
    max_qps: float           # 最大 QPS
    kv_cache_per_gpu_GB: float
    weight_per_gpu_GB: float


class ParallelismAdvisor:
    """并行策略顾问"""
    
    def __init__(self, model: ModelConfig, hw: HardwareConfig):
        self.model = model
        self.hw = hw
    
    def enumerate_configs(
        self,
        num_gpus: int,
        gpus_per_node: int,
        max_seq_len: int,
        target_tpot_ms: float = None,
        target_qps: float = None,
    ) -> List[Tuple[DeploymentConfig, PerformanceEstimate]]:
        """
        枚举所有可行的并行配置并预估性能
        
        TODO: 实现
        
        步骤:
        1. 枚举所有 (TP, PP, EP, DP, CP) 组合
        2. 过滤不可行的组合 (显存不够, GPU 数不够, etc.)
        3. 对可行配置预估性能
        4. 按照目标排序 (延迟优先 or 吞吐优先)
        5. 返回 top-5 候选
        """
        pass
    
    def estimate_performance(
        self,
        config: DeploymentConfig,
        max_seq_len: int,
    ) -> PerformanceEstimate:
        """
        预估给定配置的性能
        
        TODO: 实现
        """
        pass
    
    def recommend(
        self,
        num_gpus: int,
        gpus_per_node: int,
        max_seq_len: int,
        priority: str = "balanced",  # "latency", "throughput", "balanced"
    ) -> DeploymentConfig:
        """
        推荐最优配置
        
        TODO: 实现
        """
        pass


def main():
    """交互式策略推荐"""
    
    # 定义几个测试场景
    scenarios = [
        {
            "name": "场景 1: 8B 模型, 8 GPU, 短对话",
            "model": ModelConfig("Llama-3-8B", 8, 32, 4096, 8, 128),
            "num_gpus": 8, "gpus_per_node": 8,
            "max_seq_len": 4096,
        },
        {
            "name": "场景 2: 70B 模型, 16 GPU, 中等上下文",
            "model": ModelConfig("Llama-3-70B", 70, 80, 8192, 8, 128),
            "num_gpus": 16, "gpus_per_node": 8,
            "max_seq_len": 32768,
        },
        {
            "name": "场景 3: 70B 模型, 16 GPU, 超长上下文",
            "model": ModelConfig("Llama-3.1-70B", 70, 80, 8192, 8, 128),
            "num_gpus": 16, "gpus_per_node": 8,
            "max_seq_len": 1_000_000,
        },
        {
            "name": "场景 4: 405B 模型, 32 GPU",
            "model": ModelConfig("Llama-3-405B", 405, 126, 16384, 8, 128),
            "num_gpus": 32, "gpus_per_node": 8,
            "max_seq_len": 8192,
        },
    ]
    
    hw = HardwareConfig("H100", 3.35, 450, 5)
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"  {scenario['name']}")
        print(f"{'='*60}")
        
        advisor = ParallelismAdvisor(scenario["model"], hw)
        
        # TODO:
        # 1. 调用 enumerate_configs 获取候选方案
        # 2. 打印 top-5 候选方案的配置和预估性能
        # 3. 调用 recommend 获取推荐方案
        # 4. 输出推荐理由
        pass
```

### 预期输出示例

```
============================================================
  场景 2: 70B 模型, 16 GPU, 中等上下文
============================================================

候选方案:
┌────┬────┬────┬────┬────┬──────────┬──────────┬──────────┐
│ TP │ PP │ EP │ DP │ CP │ TPOT(ms) │ Max QPS  │ KV(GB)   │
├────┼────┼────┼────┼────┼──────────┼──────────┼──────────┤
│  4 │  1 │  1 │  4 │  1 │   10.2   │   62.0   │  42.5    │
│  8 │  1 │  1 │  2 │  1 │    7.5   │   38.0   │  55.0    │
│  4 │  2 │  1 │  2 │  1 │   11.8   │   45.0   │  42.5    │
│  2 │  1 │  1 │  8 │  1 │   18.5   │   95.0   │  30.0    │
│  8 │  2 │  1 │  1 │  1 │    9.0   │   20.0   │  55.0    │
└────┴────┴────┴────┴────┴──────────┴──────────┴──────────┘

推荐: TP=4, PP=1, DP=4
理由: 在满足延迟要求 (TPOT<30ms) 的前提下, 最大化吞吐 (QPS=62)
```

### 思考题

1. 为什么有时 TP=4+DP=4 比 TP=8+DP=2 的吞吐更高，即使 TP=8 的单请求延迟更低？
2. 在场景 3（1M 上下文）中，CP 是必须的吗？有没有其他替代方案？
3. 你的选择器是否考虑了量化（FP8/INT8）的影响？如何扩展以支持量化？

---

## 练习 5: vLLM 分布式推理实战

**目标**：在实际环境中部署分布式推理并测量性能。

> 注意：此练习需要多 GPU 环境。如果没有多 GPU 环境，可以使用单 GPU 完成部分任务，或使用云服务（如 RunPod, Lambda Labs, 或 AWS）临时租用。

### 任务

#### 5.1 基础 TP 部署与 Benchmark

```bash
# 1. 安装 vLLM (需要 CUDA 12+)
pip install vllm

# 2. 启动 TP=2 的推理服务
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --port 8000

# 3. 使用 vLLM benchmark 工具测试
python -m vllm.entrypoints.openai.run_batch \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tp 2 \
    --input-len 512 \
    --output-len 128 \
    --num-prompts 100
```

#### 5.2 TP 对比实验

```bash
# 对比 TP=1 和 TP=2 的性能差异

# TP=1
python benchmark.py --tp 1 --batch-sizes 1,4,16,64,128

# TP=2
python benchmark.py --tp 2 --batch-sizes 1,4,16,64,128

# 记录并对比:
# 1. 不同 batch size 下的 decode latency
# 2. 不同 batch size 下的 throughput (tokens/s)
# 3. TP=2 相对 TP=1 的加速比
# 4. 在什么 batch size 下 TP=2 的加速比最大？最小？
```

#### 5.3 编写 Benchmark 脚本

```python
"""
编写一个完整的分布式推理 benchmark 脚本

要求:
1. 支持不同的 TP/PP/DP 配置
2. 测量 TTFT, TPOT, 总吞吐
3. 测量不同 prompt 长度的性能
4. 输出 CSV 格式的结果供后续分析
"""

import asyncio
import aiohttp
import time
import csv
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    config: str              # "tp2_pp1_dp1"
    prompt_len: int
    output_len: int
    batch_size: int
    ttft_ms: float           # Time to First Token
    tpot_ms: float           # Time Per Output Token
    total_time_ms: float
    throughput_tps: float    # tokens per second


async def benchmark_single_request(
    url: str,
    prompt: str,
    max_tokens: int,
) -> BenchmarkResult:
    """
    发送单个请求并测量延迟
    
    TODO: 实现
    - 使用 streaming API 测量 TTFT 和 TPOT
    - 记录完整的延迟分解
    """
    pass


async def benchmark_concurrent(
    url: str,
    prompts: list,
    max_tokens: int,
    concurrency: int,
) -> list:
    """
    并发发送多个请求并测量吞吐
    
    TODO: 实现
    - 控制并发度
    - 测量总吞吐
    """
    pass


async def run_benchmark_suite(
    url: str = "http://localhost:8000/v1/completions",
    config_name: str = "tp2_pp1_dp1",
):
    """
    运行完整的 benchmark 套件
    
    TODO:
    1. 不同 prompt 长度: 128, 512, 2048, 8192
    2. 不同 output 长度: 64, 256, 1024
    3. 不同并发度: 1, 4, 16, 64
    4. 将结果保存到 CSV
    """
    pass
```

### 思考题

1. 在你的实验中，TP=2 对 decode 的加速比是多少？与理论预估（~1.8-1.9x）是否吻合？差异来自哪里？
2. 增大 batch size 时，TP 的加速比有什么变化趋势？为什么？
3. 如果你有 4 张 GPU，部署 Llama-3-8B 时，TP=2+DP=2 和 TP=1+DP=4 哪个 QPS 更高？为什么？

---

## 提交检查清单

- [ ] 练习 1: 实现延迟分解函数，输出对比表格
- [ ] 练习 2: 实现 PP 模拟器，对比 naive vs interleaved 调度
- [ ] 练习 3: 实现 MoE 负载模拟，对比 Static EP vs EPLB
- [ ] 练习 4: 实现并行策略选择器，输出推荐配置
- [ ] 练习 5: (需要多 GPU) 完成实际部署和 benchmark
- [ ] 所有思考题已作答

---

> **返回**：[分布式推理概览](README.md)
