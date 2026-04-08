# 动手练习

> 本章的练习侧重工程实践，目标是将前面学到的知识应用到实际场景中。

## 练习 1：搭建 vLLM 监控 Dashboard

### 目标

为 vLLM 推理服务搭建完整的 Prometheus + Grafana 监控系统。

### 前置条件

- Docker 和 Docker Compose
- 一个可运行 vLLM 的 GPU 环境（至少一张支持的 GPU）
- 基本的 Prometheus/Grafana 使用经验

### 步骤

**Step 1: 创建项目结构**

```bash
mkdir -p vllm-monitoring/{grafana/provisioning/datasources,grafana/provisioning/dashboards,grafana/dashboards}
cd vllm-monitoring
```

**Step 2: 编写 docker-compose.yml**

```yaml
# docker-compose.yml
version: "3.8"

services:
  # vLLM 推理服务 (如果有 GPU)
  # 如果没有 GPU，可以用 mock exporter 代替
  vllm:
    image: vllm/vllm-openai:latest
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
              count: 1
    ports:
      - "8000:8000"
    command: >
      --model Qwen/Qwen2.5-1.5B-Instruct
      --max-model-len 2048
      --gpu-memory-utilization 0.8

  prometheus:
    image: prom/prometheus:v2.50.0
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:10.3.0
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning
      - ./grafana/dashboards:/var/lib/grafana/dashboards
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_AUTH_ANONYMOUS_ENABLED=true
      - GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer
```

**Step 3: 配置 Prometheus**

```yaml
# prometheus.yml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: "vllm"
    static_configs:
      - targets: ["vllm:8000"]
```

**Step 4: 配置 Grafana 数据源**

```yaml
# grafana/provisioning/datasources/prometheus.yml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

**Step 5: 启动并验证**

```bash
docker compose up -d

# 等待 vLLM 加载模型
sleep 120

# 验证 metrics endpoint
curl http://localhost:8000/metrics

# 验证 Prometheus
curl http://localhost:9090/api/v1/targets

# 打开 Grafana
# 浏览器访问 http://localhost:3000 (admin/admin)
```

### 练习任务

1. **创建 4 个面板的 Dashboard：**
   - 请求速率 (req/s) 和成功率
   - TTFT P50/P90/P99 折线图
   - GPU KV Cache 使用率仪表盘
   - 吞吐量 (input tokens/s + output tokens/s)

2. **发送负载并观察指标变化：**
   ```python
   # load_test.py
   import httpx
   import asyncio
   import time
   
   async def send_request(client, prompt):
       resp = await client.post(
           "http://localhost:8000/v1/chat/completions",
           json={
               "model": "Qwen/Qwen2.5-1.5B-Instruct",
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 100,
           },
           timeout=60,
       )
       return resp.json()
   
   async def load_test(qps=5, duration=120):
       """发送固定 QPS 的请求"""
       async with httpx.AsyncClient() as client:
           start = time.time()
           prompts = [
               "Explain quantum computing in simple terms.",
               "Write a Python function to sort a list.",
               "What is the capital of France?",
               "Summarize the theory of relativity.",
               "How does a neural network work?",
           ]
           
           request_count = 0
           while time.time() - start < duration:
               tasks = []
               for _ in range(qps):
                   prompt = prompts[request_count % len(prompts)]
                   tasks.append(send_request(client, prompt))
                   request_count += 1
               
               results = await asyncio.gather(*tasks, return_exceptions=True)
               successes = sum(1 for r in results if not isinstance(r, Exception))
               print(f"[{time.time()-start:.0f}s] Sent {len(tasks)}, "
                     f"Success: {successes}, Total: {request_count}")
               
               await asyncio.sleep(1)
   
   asyncio.run(load_test(qps=5, duration=120))
   ```

3. **回答以下问题：**
   - 在 5 QPS 负载下，GPU KV Cache 使用率是多少？
   - TTFT P99 是多少？是否满足 2 秒 SLA？
   - 如果将 QPS 提高到 20，哪个指标先告警？

### 验收标准

- [ ] Grafana Dashboard 包含至少 4 个面板
- [ ] 能够在 Dashboard 上看到负载测试期间的指标变化
- [ ] 能够解释各指标的含义及其变化趋势

---

## 练习 2：实现 Cache-Aware 路由器

### 目标

实现一个简单的 cache-aware 路由器，对比它与 round-robin 路由的 prefix cache 命中率差异。

### 步骤

**Step 1: 启动两个 vLLM 实例**

```bash
# 实例 1
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --port 8000 \
    --enable-prefix-caching &

# 实例 2
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --port 8001 \
    --enable-prefix-caching &
```

**Step 2: 实现路由器**

```python
# router.py
import hashlib
import httpx
import asyncio
import time
from fastapi import FastAPI, Request
from collections import defaultdict

app = FastAPI()

BACKENDS = ["http://localhost:8000", "http://localhost:8001"]

# 统计信息
stats = defaultdict(int)
round_robin_counter = 0

def get_prefix_hash(messages: list) -> str:
    """提取 system message 作为路由 key"""
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            return hashlib.sha256(content[:200].encode()).hexdigest()
    # 没有 system message, 用第一条 user message
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            return hashlib.sha256(content[:100].encode()).hexdigest()
    return "default"

@app.post("/v1/chat/completions")
async def route_request(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    
    # 根据 header 选择路由策略
    strategy = request.headers.get("x-routing-strategy", "cache-aware")
    
    if strategy == "round-robin":
        global round_robin_counter
        backend_idx = round_robin_counter % len(BACKENDS)
        round_robin_counter += 1
    else:
        prefix_hash = get_prefix_hash(messages)
        backend_idx = int(prefix_hash, 16) % len(BACKENDS)
    
    backend = BACKENDS[backend_idx]
    stats[f"{strategy}_to_{backend_idx}"] += 1
    
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(f"{backend}/v1/chat/completions", json=body)
        return resp.json()

@app.get("/stats")
async def get_stats():
    """获取路由统计"""
    # 同时获取各实例的 cache hit rate
    cache_stats = {}
    async with httpx.AsyncClient() as client:
        for i, backend in enumerate(BACKENDS):
            try:
                resp = await client.get(f"{backend}/metrics")
                # 解析 prefix cache hit rate
                for line in resp.text.split("\n"):
                    if "prefix_cache_hit_rate" in line and not line.startswith("#"):
                        cache_stats[f"backend_{i}_cache_hit_rate"] = line.split()[-1]
            except Exception:
                pass
    
    return {"routing_stats": dict(stats), "cache_stats": cache_stats}
```

### 练习任务

1. 启动路由器: `uvicorn router:app --port 9000`

2. 准备 3 种不同的 system prompt，发送 200 个请求：
   - 100 个请求使用 `round-robin` 策略
   - 100 个请求使用 `cache-aware` 策略

3. 对比两种策略下各 backend 的 prefix cache hit rate

4. **思考：** 如果有 10 种 system prompt 但 2 个 backend，cache-aware 路由应该如何分配？

### 验收标准

- [ ] cache-aware 路由的 prefix cache hit rate 显著高于 round-robin
- [ ] 能够解释 cache hit rate 差异的原因
- [ ] 代码能正确运行并输出统计信息

---

## 练习 3：性能瓶颈诊断

### 目标

使用 profiling 工具诊断不同配置下的性能瓶颈。

### 步骤

**Step 1: 准备 benchmark 脚本**

```python
# benchmark.py
import httpx
import asyncio
import time
import statistics

async def benchmark(
    endpoint: str,
    prompt_length: int,
    max_tokens: int,
    num_requests: int,
    concurrent: int,
):
    """基准测试"""
    
    prompt = "Hello " * (prompt_length // 2)  # 粗略估算 token 数
    
    ttfts = []
    tbts = []
    e2e_latencies = []
    
    semaphore = asyncio.Semaphore(concurrent)
    
    async def single_request():
        async with semaphore:
            start = time.perf_counter()
            
            async with httpx.AsyncClient(timeout=120) as client:
                # 使用 streaming 来测量 TTFT
                async with client.stream(
                    "POST",
                    f"{endpoint}/v1/chat/completions",
                    json={
                        "model": "Qwen/Qwen2.5-1.5B-Instruct",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "stream": True,
                    }
                ) as response:
                    first_token_time = None
                    token_times = []
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: ") and line != "data: [DONE]":
                            now = time.perf_counter()
                            if first_token_time is None:
                                first_token_time = now
                                ttfts.append(now - start)
                            else:
                                token_times.append(now)
                    
                    end = time.perf_counter()
                    e2e_latencies.append(end - start)
                    
                    # 计算 TBT
                    if len(token_times) > 1:
                        intervals = [
                            token_times[i] - token_times[i-1] 
                            for i in range(1, len(token_times))
                        ]
                        tbts.extend(intervals)
    
    tasks = [single_request() for _ in range(num_requests)]
    await asyncio.gather(*tasks, return_exceptions=True)
    
    def percentile(data, p):
        if not data:
            return 0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        return sorted_data[min(idx, len(sorted_data) - 1)]
    
    print(f"\n=== Benchmark Results ===")
    print(f"Prompt length: ~{prompt_length} tokens")
    print(f"Max output tokens: {max_tokens}")
    print(f"Concurrent: {concurrent}")
    print(f"Requests: {num_requests}")
    print(f"\nTTFT:")
    print(f"  P50: {percentile(ttfts, 50)*1000:.1f}ms")
    print(f"  P90: {percentile(ttfts, 90)*1000:.1f}ms")
    print(f"  P99: {percentile(ttfts, 99)*1000:.1f}ms")
    print(f"\nTBT:")
    print(f"  P50: {percentile(tbts, 50)*1000:.1f}ms")
    print(f"  P90: {percentile(tbts, 90)*1000:.1f}ms")
    print(f"  P99: {percentile(tbts, 99)*1000:.1f}ms")
    print(f"\nE2E Latency:")
    print(f"  P50: {percentile(e2e_latencies, 50)*1000:.1f}ms")
    print(f"  P99: {percentile(e2e_latencies, 99)*1000:.1f}ms")
    print(f"\nThroughput: {num_requests / max(e2e_latencies):.1f} req/s")

# 测试不同场景
asyncio.run(benchmark("http://localhost:8000", 
    prompt_length=100, max_tokens=50, num_requests=50, concurrent=1))
asyncio.run(benchmark("http://localhost:8000", 
    prompt_length=100, max_tokens=50, num_requests=50, concurrent=10))
asyncio.run(benchmark("http://localhost:8000", 
    prompt_length=2000, max_tokens=50, num_requests=20, concurrent=5))
```

### 练习任务

1. 运行三组 benchmark：
   - 低并发 + 短 prompt (concurrent=1, prompt=100)
   - 高并发 + 短 prompt (concurrent=10, prompt=100)
   - 中并发 + 长 prompt (concurrent=5, prompt=2000)

2. 使用 `nvidia-smi dmon` 观察各场景下的 GPU 利用率和显存带宽

3. 分析：
   - 哪个场景是 compute-bound？哪个是 memory-bound？
   - 高并发对 TTFT 的影响是什么？为什么？
   - 长 prompt 对 TBT 的影响是什么？（提示：chunked prefill）

### 验收标准

- [ ] 完成三组 benchmark 并记录结果
- [ ] 能够正确判断各场景的瓶颈类型
- [ ] 提出至少一个配置调优建议（例如调整 `max-num-batched-tokens`）

---

## 练习 4：构建简易灰度发布系统

### 目标

实现一个支持金丝雀发布的简单路由系统，并模拟一次完整的模型更新流程。

### 步骤

```python
# canary_router.py
import httpx
import asyncio
import random
import time
from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI()

class RouterConfig(BaseModel):
    stable_weight: int = 100
    canary_weight: int = 0
    stable_endpoints: list[str] = []
    canary_endpoints: list[str] = []

config = RouterConfig(
    stable_endpoints=["http://localhost:8000"],
    canary_endpoints=["http://localhost:8001"],
)

request_log = []  # 记录每个请求的路由决策

@app.post("/v1/chat/completions")
async def route(request: Request):
    body = await request.json()
    
    # 加权随机选择
    total = config.stable_weight + config.canary_weight
    if total == 0:
        return {"error": "No backends configured"}
    
    r = random.randint(1, total)
    if r <= config.canary_weight and config.canary_endpoints:
        group = "canary"
        endpoint = random.choice(config.canary_endpoints)
    else:
        group = "stable"
        endpoint = random.choice(config.stable_endpoints)
    
    start = time.time()
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(f"{endpoint}/v1/chat/completions", json=body)
        latency = time.time() - start
    
    request_log.append({
        "timestamp": time.time(),
        "group": group,
        "endpoint": endpoint,
        "latency": latency,
        "status": resp.status_code,
    })
    
    return resp.json()

@app.post("/admin/set_weights")
async def set_weights(stable: int, canary: int):
    config.stable_weight = stable
    config.canary_weight = canary
    return {"stable_weight": stable, "canary_weight": canary}

@app.get("/admin/stats")
async def get_stats():
    """获取各组的请求统计"""
    stable_reqs = [r for r in request_log if r["group"] == "stable"]
    canary_reqs = [r for r in request_log if r["group"] == "canary"]
    
    def summarize(reqs):
        if not reqs:
            return {"count": 0}
        latencies = [r["latency"] for r in reqs]
        errors = sum(1 for r in reqs if r["status"] != 200)
        return {
            "count": len(reqs),
            "avg_latency": sum(latencies) / len(latencies),
            "p99_latency": sorted(latencies)[int(len(latencies) * 0.99)],
            "error_rate": errors / len(reqs),
        }
    
    return {
        "stable": summarize(stable_reqs),
        "canary": summarize(canary_reqs),
    }
```

### 练习任务

1. 启动两个 vLLM 实例（可以用同一模型模拟不同版本）
2. 启动金丝雀路由器
3. 模拟一次完整的金丝雀发布：
   - 初始：100% stable, 0% canary
   - 阶段 1：95% stable, 5% canary → 观察 1 分钟
   - 阶段 2：75% stable, 25% canary → 观察 1 分钟
   - 阶段 3：50% stable, 50% canary → 观察 1 分钟
   - 完成：0% stable, 100% canary
4. 在每个阶段检查 `/admin/stats`，对比两组的延迟和错误率

### 验收标准

- [ ] 成功完成完整的金丝雀发布流程
- [ ] 每个阶段都有统计数据
- [ ] 能解释在实际场景中，什么指标会触发自动回滚

---

## 练习 5：成本优化分析

### 目标

为一个给定的业务场景做成本优化分析，给出具体的优化建议和预期节省。

### 场景描述

```
你在运营一个 AI 编程助手服务:
- 模型: CodeLlama-70B (或类似 70B 代码模型)
- 当前部署: 8x A100 80GB, TP=4, 2 replicas
- 日均请求量: 50,000
- 平均 prompt 长度: 3,000 tokens (包含代码上下文)
- 平均输出长度: 800 tokens
- 高峰时段: 10:00-18:00 (工作时间)
- 低谷时段: 22:00-06:00 (夜间)
- 高峰流量是低谷的 5 倍
- 所有请求使用相同的 system prompt (约 500 tokens)
- 当前 GPU 租赁价格: $2.20/GPU/hour
- 当前 prefix cache 未启用
```

### 练习任务

1. **计算当前月度成本**
   ```
   8 GPU × $2.20/h × 24h × 30d = ?
   ```

2. **分析优化机会**（至少给出 3 种）
   - Prompt Caching 的潜在收益
   - 按时段扩缩容的潜在收益
   - FP8 量化后的硬件需求变化
   - 其他你能想到的优化

3. **为每种优化计算预期节省**

4. **给出优先级排序：** 哪个优化应该先做？为什么？

5. **写一份简短的优化提案（200 字以内）**

### 验收标准

- [ ] 成本计算正确
- [ ] 至少给出 3 种优化策略及其量化收益
- [ ] 优先级排序有合理的依据
- [ ] 优化提案简洁且可执行
