# 动手练习：调度参数调优实验

> 通过 5 个递进式实验，深入理解 vLLM 调度器的行为和参数影响

## 前置准备

### 环境要求

```bash
# Python 3.10+, CUDA 12.x, 至少一张 A100/L40S/H100 GPU
pip install vllm openai aiohttp matplotlib pandas

# 下载测试模型（推荐使用较小的模型以便快速实验）
# 以下实验以 Qwen/Qwen2.5-7B-Instruct 为例
# 你也可以使用 meta-llama/Llama-3.1-8B-Instruct 或其他 7-8B 模型
```

### 通用 benchmark 脚本

后续实验将反复使用以下 benchmark 脚本：

```python
# benchmark_scheduling.py
import asyncio
import time
import json
import argparse
from typing import Optional
import aiohttp
import numpy as np

async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    max_tokens: int,
    priority: Optional[int] = None,
) -> dict:
    """发送单个请求并记录延迟指标"""
    payload = {
        "model": "model",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    if priority is not None:
        payload["priority"] = priority

    t_start = time.perf_counter()
    t_first_token = None
    token_times = []
    token_count = 0

    async with session.post(f"{url}/v1/chat/completions", json=payload) as resp:
        async for line in resp.content:
            line = line.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            data = json.loads(line[6:])
            if data["choices"][0]["delta"].get("content"):
                now = time.perf_counter()
                if t_first_token is None:
                    t_first_token = now
                else:
                    token_times.append(now)
                token_count += 1

    t_end = time.perf_counter()
    ttft = (t_first_token - t_start) * 1000 if t_first_token else None
    tbts = []
    if len(token_times) > 0 and t_first_token:
        prev = t_first_token
        for t in token_times:
            tbts.append((t - prev) * 1000)
            prev = t

    return {
        "ttft_ms": ttft,
        "tbt_mean_ms": np.mean(tbts) if tbts else None,
        "tbt_p99_ms": np.percentile(tbts, 99) if tbts else None,
        "total_ms": (t_end - t_start) * 1000,
        "tokens_generated": token_count,
    }

async def run_benchmark(
    url: str,
    prompts: list[dict],
    concurrency: int,
    request_rate: float = float("inf"),
) -> list[dict]:
    """运行 benchmark，返回所有请求的指标"""
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async with aiohttp.ClientSession() as session:
        async def bounded_request(prompt_config):
            async with semaphore:
                if request_rate != float("inf"):
                    await asyncio.sleep(np.random.exponential(1 / request_rate))
                return await send_request(
                    session, url,
                    prompt_config["prompt"],
                    prompt_config["max_tokens"],
                    prompt_config.get("priority"),
                )

        tasks = [bounded_request(p) for p in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    return [r for r in results if isinstance(r, dict)]

def print_summary(results: list[dict], label: str = ""):
    """打印 benchmark 汇总"""
    ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"] is not None]
    tbts = [r["tbt_mean_ms"] for r in results if r["tbt_mean_ms"] is not None]
    tbt_p99s = [r["tbt_p99_ms"] for r in results if r["tbt_p99_ms"] is not None]
    total_tokens = sum(r["tokens_generated"] for r in results)
    total_time = max(r["total_ms"] for r in results) / 1000

    print(f"\n{'='*60}")
    print(f" {label}")
    print(f"{'='*60}")
    print(f" 请求数: {len(results)}")
    print(f" TTFT P50: {np.percentile(ttfts, 50):.1f} ms")
    print(f" TTFT P99: {np.percentile(ttfts, 99):.1f} ms")
    print(f" TBT mean: {np.mean(tbts):.1f} ms")
    print(f" TBT P99:  {np.percentile(tbt_p99s, 99):.1f} ms")
    print(f" 吞吐量:  {total_tokens / total_time:.1f} tokens/s")
    print(f"{'='*60}\n")
```

---

## 练习 1：Static vs Continuous Batching 吞吐量对比

### 目标
量化 continuous batching 相比 static batching 的吞吐量优势，验证 `throughput_ratio ≈ L_max / L_mean`。

### 步骤

**Step 1：准备不同长度分布的测试数据**

```python
# exercise1_data.py
import random
import json

def generate_prompts(distribution: str, num_prompts: int = 200):
    """生成不同输出长度分布的 prompt"""
    prompts = []
    if distribution == "uniform":
        lengths = [random.randint(50, 300) for _ in range(num_prompts)]
    elif distribution == "exponential":
        lengths = [int(random.expovariate(1/150)) + 10 for _ in range(num_prompts)]
        lengths = [min(l, 500) for l in lengths]
    elif distribution == "bimodal":
        lengths = []
        for _ in range(num_prompts):
            if random.random() < 0.5:
                lengths.append(random.randint(20, 60))
            else:
                lengths.append(random.randint(200, 400))
    elif distribution == "fixed":
        lengths = [150] * num_prompts
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    for i, length in enumerate(lengths):
        prompts.append({
            "prompt": f"写一篇关于人工智能发展的文章，用{length}个字左右。请不要太短也不要太长。",
            "max_tokens": length,
        })
    return prompts, lengths

if __name__ == "__main__":
    for dist in ["fixed", "uniform", "exponential", "bimodal"]:
        prompts, lengths = generate_prompts(dist)
        print(f"{dist}: mean={sum(lengths)/len(lengths):.0f}, "
              f"max={max(lengths)}, "
              f"ratio={max(lengths)/(sum(lengths)/len(lengths)):.2f}")
```

**Step 2：启动 vLLM 并运行 benchmark**

```bash
# 启动 vLLM
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --max-num-seqs 64 \
    --max-num-batched-tokens 4096 \
    --enable-chunked-prefill \
    --port 8000
```

```python
# exercise1_run.py
import asyncio
from exercise1_data import generate_prompts
from benchmark_scheduling import run_benchmark, print_summary

async def main():
    url = "http://localhost:8000"

    for dist in ["fixed", "uniform", "exponential", "bimodal"]:
        prompts, lengths = generate_prompts(dist, num_prompts=200)
        results = await run_benchmark(url, prompts, concurrency=32, request_rate=20)
        print_summary(results, label=f"Distribution: {dist}")

asyncio.run(main())
```

### 思考题

1. 哪种输出长度分布下，continuous batching 的优势最明显？为什么？
2. 当所有请求的输出长度完全相同（fixed 分布）时，continuous batching 相比 static batching 还有优势吗？
3. 计算 `L_max / L_mean`，与实际观测到的吞吐量比例是否一致？

---

## 练习 2：Chunked Prefill 参数调优

### 目标
观察不同 `max_num_batched_tokens` 对 TTFT 和 TBT 的影响，找到最优 chunk size。

### 步骤

**Step 1：准备包含长短 prompt 混合的测试数据**

```python
# exercise2_data.py
import random

def generate_mixed_prompts(num_prompts: int = 100):
    """生成混合长短 prompt 的测试数据"""
    long_text = "人工智能是一种模拟人类智能的技术。" * 200  # ~4000 chars → ~2000 tokens
    short_text = "解释量子计算的基本原理。"

    prompts = []
    for i in range(num_prompts):
        if i % 5 == 0:
            # 20% 长 prompt（模拟长上下文请求）
            prompts.append({
                "prompt": f"请对以下文本进行详细分析：\n{long_text}",
                "max_tokens": 100,
                "type": "long_prefill",
            })
        else:
            # 80% 短 prompt（模拟交互式聊天）
            prompts.append({
                "prompt": short_text,
                "max_tokens": 200,
                "type": "short_chat",
            })
    return prompts
```

**Step 2：分别用不同 budget 启动 vLLM 并测试**

```bash
# 测试不同的 max_num_batched_tokens 配置
for BUDGET in 512 1024 2048 4096 8192; do
    echo "Testing budget=$BUDGET"
    vllm serve Qwen/Qwen2.5-7B-Instruct \
        --max-num-seqs 128 \
        --max-num-batched-tokens $BUDGET \
        --enable-chunked-prefill \
        --port 8000 &
    SERVER_PID=$!
    sleep 30  # 等待服务启动

    python exercise2_run.py --budget $BUDGET

    kill $SERVER_PID
    sleep 5
done
```

```python
# exercise2_run.py
import asyncio
import argparse
from exercise2_data import generate_mixed_prompts
from benchmark_scheduling import run_benchmark, print_summary

async def main(budget: int):
    url = "http://localhost:8000"
    prompts = generate_mixed_prompts(num_prompts=100)
    results = await run_benchmark(url, prompts, concurrency=32, request_rate=15)

    # 分别统计长 prompt 和短 prompt 的指标
    long_results = [r for r, p in zip(results, prompts)
                    if p.get("type") == "long_prefill" and isinstance(r, dict)]
    short_results = [r for r, p in zip(results, prompts)
                     if p.get("type") == "short_chat" and isinstance(r, dict)]

    print_summary(long_results, f"Budget={budget} | 长 Prompt 请求")
    print_summary(short_results, f"Budget={budget} | 短 Chat 请求")
    print_summary(results, f"Budget={budget} | 整体")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=2048)
    args = parser.parse_args()
    asyncio.run(main(args.budget))
```

### 思考题

1. 绘制 TTFT-P99 和 TBT-P99 随 `max_num_batched_tokens` 变化的曲线。两者是否呈反向关系？
2. 对于短 chat 请求，哪个 budget 值能同时满足 TTFT < 500ms 和 TBT < 100ms？
3. 如果你的 SLA 是 TBT P99 < 50ms，最大可用的 budget 是多少？

---

## 练习 3：优先级调度效果验证

### 目标
验证优先级调度在高负载下对高优先级请求的 SLA 保护效果。

### 步骤

**Step 1：启动带优先级调度的 vLLM**

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --max-num-seqs 64 \
    --max-num-batched-tokens 2048 \
    --enable-chunked-prefill \
    --scheduling-policy priority \
    --port 8000
```

**Step 2：同时发送高优先级和低优先级请求**

```python
# exercise3_run.py
import asyncio
import random
from benchmark_scheduling import run_benchmark, print_summary

async def main():
    url = "http://localhost:8000"

    prompts = []
    # 30 个高优先级请求（模拟付费用户）
    for i in range(30):
        prompts.append({
            "prompt": "简要解释什么是机器学习。",
            "max_tokens": 150,
            "priority": 0,
            "label": "high",
        })

    # 70 个低优先级请求（模拟免费用户）
    for i in range(70):
        prompts.append({
            "prompt": "详细介绍人工智能的历史和发展趋势。请尽可能全面。",
            "max_tokens": 300,
            "priority": 10,
            "label": "low",
        })

    # 打乱顺序
    random.shuffle(prompts)

    # 高并发发送（超过系统容量，迫使调度器做取舍）
    results = await run_benchmark(url, prompts, concurrency=100, request_rate=50)

    # 分开统计
    high_results = [r for r, p in zip(results, prompts)
                    if p.get("label") == "high" and isinstance(r, dict)]
    low_results = [r for r, p in zip(results, prompts)
                   if p.get("label") == "low" and isinstance(r, dict)]

    print_summary(high_results, "高优先级请求 (priority=0)")
    print_summary(low_results, "低优先级请求 (priority=10)")
    print_summary(results, "所有请求")

asyncio.run(main())
```

**Step 3：对比 FCFS 调度**

```bash
# 重新启动 vLLM 使用 FCFS
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --max-num-seqs 64 \
    --max-num-batched-tokens 2048 \
    --enable-chunked-prefill \
    --scheduling-policy fcfs \
    --port 8000

# 运行相同的 benchmark（去掉 priority 参数）
python exercise3_run.py
```

### 思考题

1. 优先级调度下，高优先级请求的 TTFT 和 TBT 相比 FCFS 有多大改善？
2. 低优先级请求的延迟恶化了多少？系统总吞吐量有变化吗？
3. 如果高优先级请求的比例从 30% 增加到 70%，优先级调度的效果会如何变化？

---

## 练习 4：Preemption 行为观察

### 目标
触发并观察 vLLM 的 preemption 行为，理解其对延迟和吞吐量的影响。

### 步骤

**Step 1：配置受限的 KV Cache 以触发 preemption**

```bash
# 限制 GPU 内存使用，迫使 KV Cache 不足以容纳所有并发请求
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --max-num-seqs 128 \
    --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.5 \
    --enable-chunked-prefill \
    --port 8000
```

**Step 2：发送超过 KV Cache 容量的请求**

```python
# exercise4_run.py
import asyncio
import aiohttp

async def main():
    url = "http://localhost:8000"

    # 发送大量长输出请求，超过 KV Cache 容量
    prompts = []
    for i in range(200):
        prompts.append({
            "prompt": "写一篇关于太空探索的长文章。包含历史、现状和未来展望。请尽可能详细。",
            "max_tokens": 500,
        })

    from benchmark_scheduling import run_benchmark, print_summary
    results = await run_benchmark(url, prompts, concurrency=128, request_rate=30)
    print_summary(results, "高负载（预期触发 preemption）")

asyncio.run(main())
```

**Step 3：观察 Prometheus 指标**

```bash
# 在另一个终端查看 preemption 指标
curl -s http://localhost:8000/metrics | grep -E "(preempt|cache_usage|waiting|running)"
```

```python
# exercise4_monitor.py
import time
import requests

def monitor_metrics(duration_sec: int = 60):
    """每秒采集一次关键指标"""
    metrics_history = []

    for _ in range(duration_sec):
        try:
            resp = requests.get("http://localhost:8000/metrics")
            text = resp.text
            metrics = {}
            for line in text.split("\n"):
                if line.startswith("#"):
                    continue
                for key in ["num_requests_running", "num_requests_waiting",
                            "num_preemptions_total", "gpu_cache_usage_perc"]:
                    if key in line:
                        try:
                            metrics[key] = float(line.split()[-1])
                        except ValueError:
                            pass
            metrics["timestamp"] = time.time()
            metrics_history.append(metrics)
            print(f"Running: {metrics.get('num_requests_running', '?'):>5} | "
                  f"Waiting: {metrics.get('num_requests_waiting', '?'):>5} | "
                  f"Preemptions: {metrics.get('num_preemptions_total', '?'):>5} | "
                  f"Cache: {metrics.get('gpu_cache_usage_perc', '?'):>6.1%}")
        except Exception:
            pass
        time.sleep(1)

    return metrics_history

if __name__ == "__main__":
    monitor_metrics(120)
```

### 思考题

1. 在什么 KV Cache 使用率水平下开始出现 preemption？
2. Preemption 发生时，`num_requests_running` 和 `num_requests_waiting` 如何变化？
3. 如果将 `--gpu-memory-utilization` 从 0.5 提高到 0.8，preemption 频率如何变化？
4. 被 preempt 的请求重新调度后，其 TTFT 是否包含了重新 prefill 的时间？

---

## 练习 5：调度参数综合调优

### 目标
给定一个具体的 SLA 目标和工作负载特征，找到最优的调度参数组合。

### 场景设定

```
业务场景：在线客服系统
SLA 要求：
  - TTFT P99 < 1000ms
  - TBT P99 < 100ms
  - 吞吐量 > 500 tokens/s

工作负载特征：
  - 平均 prompt 长度: 500 tokens
  - 最大 prompt 长度: 4000 tokens
  - 平均输出长度: 200 tokens
  - 并发用户数: 50-100
  - 请求到达率: ~20 req/s
```

### 步骤

**Step 1：生成模拟工作负载**

```python
# exercise5_workload.py
import random
import numpy as np

def generate_customer_service_workload(num_requests: int = 500):
    """生成模拟客服工作负载"""
    prompts = []
    # 模拟不同长度的客户消息
    base_text = "你好，我是客户。我有一个关于产品使用的问题。"
    long_context = "以下是我与客服的历史对话记录：\n" + "客户：请问如何使用这个功能？\n客服：您可以按照以下步骤操作...\n" * 50

    for i in range(num_requests):
        r = random.random()
        if r < 0.1:
            # 10% 长上下文请求
            prompt = long_context + f"\n新问题 #{i}: 基于上面的对话，请给出总结和建议。"
            max_tokens = int(np.random.normal(200, 50))
        elif r < 0.5:
            # 40% 中等长度请求
            prompt = base_text * random.randint(3, 10) + f"\n请回答问题 #{i}"
            max_tokens = int(np.random.normal(150, 40))
        else:
            # 50% 短请求
            prompt = f"简短回答：产品问题 #{i}"
            max_tokens = int(np.random.normal(100, 30))

        max_tokens = max(20, min(max_tokens, 500))
        prompts.append({
            "prompt": prompt,
            "max_tokens": max_tokens,
        })
    return prompts
```

**Step 2：参数网格搜索**

```python
# exercise5_search.py
import asyncio
import subprocess
import time
import signal
from exercise5_workload import generate_customer_service_workload
from benchmark_scheduling import run_benchmark, print_summary
import numpy as np

PARAM_GRID = {
    "max_num_seqs": [64, 128, 256],
    "max_num_batched_tokens": [1024, 2048, 4096],
}

async def test_config(max_seqs: int, budget: int):
    """测试一组参数配置"""
    prompts = generate_customer_service_workload(200)
    url = "http://localhost:8000"

    results = await run_benchmark(
        url, prompts, concurrency=80, request_rate=20
    )

    ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"] is not None]
    tbts = [r["tbt_p99_ms"] for r in results if r["tbt_p99_ms"] is not None]
    total_tokens = sum(r["tokens_generated"] for r in results)
    total_time = max(r["total_ms"] for r in results) / 1000

    ttft_p99 = np.percentile(ttfts, 99) if ttfts else float("inf")
    tbt_p99 = np.percentile(tbts, 99) if tbts else float("inf")
    throughput = total_tokens / total_time if total_time > 0 else 0

    # 检查 SLA
    sla_met = ttft_p99 < 1000 and tbt_p99 < 100 and throughput > 500

    return {
        "max_num_seqs": max_seqs,
        "budget": budget,
        "ttft_p99": ttft_p99,
        "tbt_p99": tbt_p99,
        "throughput": throughput,
        "sla_met": sla_met,
    }

# 手动执行每组参数:
# 1. 启动 vLLM 使用指定参数
# 2. 运行 test_config
# 3. 记录结果
# 4. 停止 vLLM，更换参数，重复
```

**Step 3：结果分析模板**

```python
# exercise5_analyze.py
import pandas as pd

# 将所有实验结果填入此表
results = [
    # {"max_num_seqs": 64, "budget": 1024, "ttft_p99": ?, "tbt_p99": ?, "throughput": ?, "sla_met": ?},
    # {"max_num_seqs": 64, "budget": 2048, ...},
    # ...
]

df = pd.DataFrame(results)
print("\n所有实验结果:")
print(df.to_string(index=False))

print("\n满足 SLA 的配置:")
sla_configs = df[df["sla_met"] == True]
if len(sla_configs) > 0:
    # 在满足 SLA 的配置中，选择吞吐量最高的
    best = sla_configs.loc[sla_configs["throughput"].idxmax()]
    print(f"\n最优配置:")
    print(f"  max_num_seqs = {best['max_num_seqs']}")
    print(f"  max_num_batched_tokens = {best['budget']}")
    print(f"  TTFT P99 = {best['ttft_p99']:.1f} ms")
    print(f"  TBT P99 = {best['tbt_p99']:.1f} ms")
    print(f"  吞吐量 = {best['throughput']:.1f} tokens/s")
else:
    print("没有配置满足 SLA，考虑:")
    print("  - 使用更强的 GPU")
    print("  - 降低 SLA 要求")
    print("  - 减少并发量")
```

### 思考题

1. 在 3x3 的参数网格中，哪组参数最优？为什么？
2. `max_num_seqs` 和 `max_num_batched_tokens` 哪个参数对 TBT 影响更大？哪个对吞吐量影响更大？
3. 如果 SLA 放宽到 TBT P99 < 200ms，最优配置会如何变化？吞吐量能提升多少？
4. 如果并发用户数从 80 增加到 200，当前最优配置还能满足 SLA 吗？你会如何调整？

---

## 实验记录模板

建议用以下格式记录每次实验的结果：

```markdown
## 实验记录

### 实验日期: YYYY-MM-DD
### 硬件环境: GPU 型号, 显存大小
### 模型: 模型名称
### vLLM 版本: x.y.z

| 实验 | 参数配置 | TTFT P50 | TTFT P99 | TBT Mean | TBT P99 | 吞吐量 | 备注 |
|------|---------|----------|----------|----------|---------|--------|------|
| 1-1  | budget=2048, seqs=64 | | | | | | |
| 1-2  | budget=4096, seqs=64 | | | | | | |
| ...  | ... | | | | | | |

### 关键发现
1. ...
2. ...

### 调优结论
- 最优配置: ...
- 原因: ...
```

## 扩展挑战

如果你完成了以上 5 个练习，可以尝试以下进阶挑战：

1. **多模型调度**：在同一台机器上部署两个不同大小的模型（如 7B 和 1.5B），观察它们如何共享 GPU 资源。

2. **动态负载适应**：编写一个控制器，根据实时监控的 TTFT/TBT 指标动态调整 `max_num_seqs` 和 `max_num_batched_tokens`。

3. **Prefix Caching + Chunked Prefill 交互**：在大量请求共享相同 system prompt 的场景下，观察 prefix caching 如何影响 chunked prefill 的行为（chunk 数量是否减少）。

4. **Preemption 策略对比**：对比不同 `--gpu-memory-utilization` 设置下 preemption 频率和整体性能的关系，绘制 Pareto 曲线。
