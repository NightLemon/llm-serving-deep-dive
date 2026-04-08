# 动手练习：测量 Cache Hit Rate

> 通过实际操作和代码实验，深入理解 Prefix Caching 的行为和性能影响。

## 练习 1：Anthropic Prompt Caching 行为观测

### 目标

通过 Anthropic API 观察 `cache_creation_input_tokens` 和 `cache_read_input_tokens` 的变化，理解 cache 的建立和命中过程。

### 准备

```bash
pip install anthropic
export ANTHROPIC_API_KEY="your-api-key"
```

### 实验代码

```python
"""
实验 1：观察 Anthropic Prompt Caching 的 cache 建立与命中行为。
"""
import anthropic
import time
import json

client = anthropic.Anthropic()

# 构造一个足够长的 system prompt（> 1024 tokens）
# 使用重复内容确保超过最小 token 数
system_prompt = """You are an expert software engineer specializing in 
distributed systems, machine learning infrastructure, and cloud-native 
architectures. You have deep knowledge of:

1. Distributed consensus protocols (Raft, Paxos, PBFT)
2. Stream processing frameworks (Apache Kafka, Apache Flink, Apache Spark)
3. Container orchestration (Kubernetes, Docker Swarm, Nomad)
4. Machine learning serving (vLLM, TensorRT-LLM, Triton Inference Server)
5. Database internals (B-trees, LSM-trees, MVCC, WAL)
6. Network protocols (TCP/IP, gRPC, HTTP/2, QUIC)
7. Caching systems (Redis, Memcached, CDN caching strategies)
8. Observability (OpenTelemetry, Prometheus, Grafana, distributed tracing)

When answering questions, you should:
- Provide detailed technical explanations with concrete examples
- Reference specific algorithms, data structures, and design patterns
- Consider trade-offs between different approaches
- Include relevant performance characteristics and complexity analysis
- Cite relevant papers or documentation when applicable

Your responses should be structured with clear headings, bullet points, 
and code examples where appropriate. Always consider both theoretical 
foundations and practical engineering implications.
""" * 3  # 重复 3 次确保超过 1024 tokens

questions = [
    "Explain how Raft consensus works in 3 sentences.",
    "What is the difference between TCP and UDP?",
    "How does a B-tree handle node splits?",
    "What is the CAP theorem?",
    "Explain eventual consistency.",
]

print("=" * 60)
print("实验 1：Anthropic Prompt Caching 行为观测")
print("=" * 60)

results = []

for i, question in enumerate(questions):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": question}],
    )
    
    usage = response.usage
    result = {
        "request": i + 1,
        "question": question[:50],
        "input_tokens": usage.input_tokens,
        "cache_creation": getattr(usage, "cache_creation_input_tokens", 0),
        "cache_read": getattr(usage, "cache_read_input_tokens", 0),
        "output_tokens": usage.output_tokens,
    }
    results.append(result)
    
    print(f"\n--- 请求 {i+1}: {question[:50]}... ---")
    print(f"  input_tokens:                {result['input_tokens']}")
    print(f"  cache_creation_input_tokens: {result['cache_creation']}")
    print(f"  cache_read_input_tokens:     {result['cache_read']}")
    print(f"  output_tokens:               {result['output_tokens']}")
    
    # 短暂等待，观察 TTL 行为
    if i < len(questions) - 1:
        time.sleep(2)

print("\n" + "=" * 60)
print("结果分析")
print("=" * 60)

# 第一次请求应该是 cache_creation > 0, cache_read == 0
# 后续请求应该是 cache_creation == 0, cache_read > 0
first = results[0]
if first["cache_creation"] > 0 and first["cache_read"] == 0:
    print("✓ 第一次请求：正确触发了 cache creation")
else:
    print("✗ 第一次请求：未触发 cache creation（检查 system prompt 长度）")

subsequent_hits = sum(
    1 for r in results[1:] if r["cache_read"] > 0
)
print(f"✓ 后续 {len(results)-1} 次请求中，{subsequent_hits} 次命中 cache")

# 计算节省
total_normal_cost = sum(
    (r["cache_creation"] + r["cache_read"] + r["input_tokens"])
    for r in results
) * 3 / 1_000_000  # $3/M tokens

total_cached_cost = sum(
    r["cache_creation"] * 3.75 / 1_000_000
    + r["cache_read"] * 0.30 / 1_000_000
    + r["input_tokens"] * 3.00 / 1_000_000
    for r in results
)

print(f"\n不使用 cache 的成本: ${total_normal_cost:.6f}")
print(f"使用 cache 的成本:   ${total_cached_cost:.6f}")
print(f"节省: ${total_normal_cost - total_cached_cost:.6f} "
      f"({(1 - total_cached_cost / total_normal_cost) * 100:.1f}%)")
```

### 思考题

1. 如果你把 `system_prompt` 缩短到 500 tokens 以下（低于 1024 最小值），`cache_creation_input_tokens` 还会出现吗？
2. 如果你在两次请求之间 `time.sleep(360)`（等 6 分钟，超过 5 分钟 TTL），cache 还会命中吗？
3. 如果你去掉 `cache_control` breakpoint，行为会有什么变化？

---

## 练习 2：OpenAI Prompt Caching 自动触发实验

### 目标

验证 OpenAI 的全自动 Prompt Caching 行为，观察 `cached_tokens` 字段。

### 实验代码

```python
"""
实验 2：观察 OpenAI Prompt Caching 的自动触发行为。
"""
from openai import OpenAI
import time

client = OpenAI()

# 构造长 system prompt（> 1024 tokens）
system_prompt = """You are a senior software architect with extensive 
experience in designing large-scale distributed systems. Your expertise 
covers microservices architecture, event-driven design, CQRS, event 
sourcing, domain-driven design, and cloud-native patterns.

Key areas of expertise:

## Distributed Systems Fundamentals
- Consensus algorithms: Raft, Paxos, ZAB
- Distributed transactions: 2PC, Saga pattern
- Consistent hashing and data partitioning
- Vector clocks and conflict resolution
- Gossip protocols and membership management

## Cloud-Native Architecture
- Container orchestration with Kubernetes
- Service mesh (Istio, Linkerd, Envoy)
- Serverless computing patterns
- Multi-region deployment strategies
- Chaos engineering and resilience testing

## Data Engineering
- Stream processing with Apache Kafka and Flink
- Data lake architectures (Delta Lake, Apache Iceberg)
- OLAP engines (ClickHouse, Apache Druid, StarRocks)
- Graph databases (Neo4j, TigerGraph)
- Time-series databases (InfluxDB, TimescaleDB)

## Machine Learning Infrastructure
- Model serving frameworks (vLLM, TensorRT-LLM, Triton)
- Feature stores (Feast, Tecton)
- ML pipelines (Kubeflow, MLflow, Airflow)
- GPU cluster management and scheduling
- A/B testing and experimentation platforms

When providing architectural guidance:
1. Always consider scalability, reliability, and maintainability
2. Discuss trade-offs explicitly
3. Provide concrete examples from real-world systems
4. Include monitoring and observability considerations
5. Address security and compliance requirements
""" * 2  # 重复确保超过 1024 tokens

questions = [
    "How would you design a rate limiter?",
    "Explain the Saga pattern for distributed transactions.",
    "What are the trade-offs of event sourcing?",
    "How does consistent hashing work?",
    "Describe a circuit breaker pattern.",
]

print("=" * 60)
print("实验 2：OpenAI Prompt Caching 自动触发")
print("=" * 60)

for i, question in enumerate(questions):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=100,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )
    
    usage = response.usage
    cached = 0
    if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
        cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
    
    total_prompt = usage.prompt_tokens
    
    print(f"\n--- 请求 {i+1}: {question[:50]} ---")
    print(f"  prompt_tokens:  {total_prompt}")
    print(f"  cached_tokens:  {cached}")
    print(f"  output_tokens:  {usage.completion_tokens}")
    print(f"  cache hit rate: {cached/total_prompt*100:.1f}%" if total_prompt > 0 else "  N/A")
    
    time.sleep(1)

print("\n" + "=" * 60)
print("观察要点：")
print("1. 第一次请求的 cached_tokens 应该为 0")
print("2. 后续请求的 cached_tokens 应该接近 system prompt 的 token 数")
print("3. cached_tokens 是 128 的整数倍")
print("=" * 60)
```

### 思考题

1. OpenAI 的 `cached_tokens` 值为什么总是 128 的整数倍？这暗示了什么内部实现细节？
2. 如果两个请求之间间隔 15 分钟，cache 还会命中吗？测试一下。
3. 如果你改变 system prompt 中的一个字符，cache 命中情况如何变化？

---

## 练习 3：vLLM Prefix Caching 性能测量

### 目标

在本地部署的 vLLM 服务上测量 Prefix Caching 的 TTFT 加速效果。

### 准备

```bash
# 安装 vLLM（需要 GPU）
pip install vllm

# 启动 vLLM 服务（启用 prefix caching）
vllm serve meta-llama/Llama-3.2-1B-Instruct \
    --enable-prefix-caching \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9
```

如果没有 GPU，可以使用 vLLM 提供的 mock 模式或跳到练习 4。

### 实验代码

```python
"""
实验 3：测量 vLLM Prefix Caching 对 TTFT 的影响。
"""
import time
import requests
import json
import statistics

VLLM_URL = "http://localhost:8000/v1/chat/completions"

# 长 system prompt
system_prompt = "You are a helpful assistant. " * 200  # ~800 tokens
# 如果需要更明显的效果，可以增加到 * 500

def send_request(system: str, user: str) -> dict:
    """发送请求并测量 TTFT。"""
    start = time.perf_counter()
    
    response = requests.post(
        VLLM_URL,
        json={
            "model": "meta-llama/Llama-3.2-1B-Instruct",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": 50,
            "stream": True,
        },
        stream=True,
    )
    
    # 测量到第一个 token 的时间
    ttft = None
    full_response = ""
    
    for line in response.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: ") and line != "data: [DONE]":
                if ttft is None:
                    ttft = time.perf_counter() - start
                try:
                    data = json.loads(line[6:])
                    delta = data["choices"][0]["delta"]
                    if "content" in delta:
                        full_response += delta["content"]
                except json.JSONDecodeError:
                    pass
    
    total_time = time.perf_counter() - start
    
    return {
        "ttft": ttft,
        "total_time": total_time,
    }


def run_experiment(
    name: str,
    system: str,
    questions: list[str],
    num_warmup: int = 2,
    num_measure: int = 5,
):
    """运行实验并统计 TTFT。"""
    print(f"\n{'=' * 50}")
    print(f"实验: {name}")
    print(f"{'=' * 50}")
    
    # Warmup：建立 cache
    print(f"\nWarmup ({num_warmup} 次)...")
    for i in range(num_warmup):
        result = send_request(system, questions[i % len(questions)])
        print(f"  Warmup {i+1}: TTFT = {result['ttft']*1000:.1f} ms")
        time.sleep(0.5)
    
    # Measure：测量 cache hit 场景
    print(f"\n测量 ({num_measure} 次，相同前缀)...")
    ttfts = []
    for i in range(num_measure):
        q = questions[(i + num_warmup) % len(questions)]
        result = send_request(system, q)
        ttfts.append(result["ttft"])
        print(f"  测量 {i+1}: TTFT = {result['ttft']*1000:.1f} ms")
        time.sleep(0.5)
    
    avg_ttft = statistics.mean(ttfts)
    p50_ttft = statistics.median(ttfts)
    
    print(f"\n结果:")
    print(f"  平均 TTFT: {avg_ttft*1000:.1f} ms")
    print(f"  P50 TTFT:  {p50_ttft*1000:.1f} ms")
    
    return ttfts


# 运行对比实验
questions = [
    "What is machine learning?",
    "Explain neural networks briefly.",
    "What is gradient descent?",
    "Define overfitting.",
    "What is regularization?",
    "Explain backpropagation.",
    "What is a loss function?",
    "Define batch normalization.",
    "What is transfer learning?",
    "Explain attention mechanism.",
]

# 实验 A：相同 system prompt（应该有 cache hit）
ttfts_cached = run_experiment(
    "相同前缀（Cache Hit）",
    system_prompt,
    questions,
)

# 实验 B：每次不同的 system prompt（无 cache hit）
ttfts_uncached = []
print(f"\n{'=' * 50}")
print("实验: 不同前缀（Cache Miss）")
print(f"{'=' * 50}")
for i in range(5):
    unique_system = f"Session {i}: " + system_prompt
    result = send_request(unique_system, questions[i])
    ttfts_uncached.append(result["ttft"])
    print(f"  测量 {i+1}: TTFT = {result['ttft']*1000:.1f} ms")
    time.sleep(0.5)

# 对比
print(f"\n{'=' * 50}")
print("对比结果")
print(f"{'=' * 50}")
avg_cached = statistics.mean(ttfts_cached)
avg_uncached = statistics.mean(ttfts_uncached)
print(f"Cache Hit 平均 TTFT:  {avg_cached*1000:.1f} ms")
print(f"Cache Miss 平均 TTFT: {avg_uncached*1000:.1f} ms")
print(f"加速比: {avg_uncached/avg_cached:.2f}x")
```

### 思考题

1. TTFT 加速比与 system prompt 长度之间是什么关系？尝试 500 / 1000 / 2000 token 的 system prompt。
2. 重启 vLLM 服务后再次测量——cache 是否丢失？这说明了什么？
3. 观察 vLLM 的日志输出，能否找到 cache hit/miss 的相关日志？

---

## 练习 4：Prompt 结构优化实验

### 目标

通过改变 prompt 中动态和静态内容的位置，测量对 cache hit rate 的影响。

### 实验代码

```python
"""
实验 4：Prompt 结构对 Cache Hit Rate 的影响。

本实验不需要实际调用 API——通过模拟来理解前缀匹配的原理。
"""
import hashlib


def simulate_prefix_matching(
    prompts: list[str], 
    block_size: int = 64,  # 字符为单位简化模拟
) -> dict:
    """模拟前缀匹配过程。
    
    Args:
        prompts: 多个请求的完整 prompt 文本
        block_size: 每个 block 的字符数（简化版）
    
    Returns:
        统计信息
    """
    cache = {}  # hash -> block_content
    stats = {
        "total_blocks": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "hit_rate_per_request": [],
    }
    
    for prompt_idx, prompt in enumerate(prompts):
        # 分 block
        blocks = []
        for i in range(0, len(prompt), block_size):
            blocks.append(prompt[i : i + block_size])
        
        # 链式 hash + 前缀匹配
        parent_hash = "INIT"
        hits = 0
        misses = 0
        
        for block in blocks:
            block_hash = hashlib.md5(
                f"{parent_hash}:{block}".encode()
            ).hexdigest()
            
            if block_hash in cache:
                hits += 1
            else:
                cache[block_hash] = block
                misses += 1
                # 注意：一旦 miss，后续 block 也一定 miss（前缀匹配）
                # 但我们这里不 break，为了展示哪些 block 理论上可以匹配
                # 实际系统会在第一个 miss 处停止
            
            parent_hash = block_hash
        
        stats["total_blocks"] += len(blocks)
        stats["cache_hits"] += hits
        stats["cache_misses"] += misses
        
        hit_rate = hits / len(blocks) if blocks else 0
        stats["hit_rate_per_request"].append(hit_rate)
        
        print(f"  请求 {prompt_idx+1}: {len(blocks)} blocks, "
              f"{hits} hits, {misses} misses, "
              f"hit rate = {hit_rate:.1%}")
    
    overall = (
        stats["cache_hits"] / stats["total_blocks"]
        if stats["total_blocks"] > 0 else 0
    )
    stats["overall_hit_rate"] = overall
    
    return stats


# ============================================
# 场景 A：动态内容在前面（BAD）
# ============================================
print("场景 A：动态内容在前面（BAD）")
print("-" * 40)

static_content = "You are a helpful assistant with expertise in Python, " * 20
prompts_bad = [
    f"[Timestamp: 2025-01-01T{hour:02d}:00:00] {static_content} Question: Hello"
    for hour in range(10)
]

stats_a = simulate_prefix_matching(prompts_bad)
print(f"总体 Hit Rate: {stats_a['overall_hit_rate']:.1%}\n")

# ============================================
# 场景 B：静态内容在前面（GOOD）
# ============================================
print("场景 B：静态内容在前面（GOOD）")
print("-" * 40)

prompts_good = [
    f"{static_content} [Timestamp: 2025-01-01T{hour:02d}:00:00] Question: Hello"
    for hour in range(10)
]

stats_b = simulate_prefix_matching(prompts_good)
print(f"总体 Hit Rate: {stats_b['overall_hit_rate']:.1%}\n")

# ============================================
# 场景 C：多轮对话的增量缓存
# ============================================
print("场景 C：多轮对话的增量缓存")
print("-" * 40)

base = static_content
conversation_prompts = []
for turn in range(5):
    base += f" User: Question {turn}. Assistant: Answer {turn}."
    conversation_prompts.append(base + f" User: Question {turn+1}?")

stats_c = simulate_prefix_matching(conversation_prompts)
print(f"总体 Hit Rate: {stats_c['overall_hit_rate']:.1%}\n")

# ============================================
# 总结
# ============================================
print("=" * 50)
print("总结")
print("=" * 50)
print(f"场景 A（动态在前）: Hit Rate = {stats_a['overall_hit_rate']:.1%}")
print(f"场景 B（静态在前）: Hit Rate = {stats_b['overall_hit_rate']:.1%}")
print(f"场景 C（增量对话）: Hit Rate = {stats_c['overall_hit_rate']:.1%}")
print()
print("结论：")
print("1. 将动态内容放在前缀中会严重降低 cache hit rate")
print("2. 静态内容放前面可以显著提高 cache hit rate")
print("3. 多轮对话天然具有增量前缀匹配的特性")
```

### 思考题

1. 在场景 A 中，为什么整体 hit rate 接近 0%？
2. 场景 C 中，为什么后面的请求 hit rate 比前面的高？
3. 如果有 100 个不同的 system prompt 变体（如 A/B test），对 cache hit rate 有什么影响？如何优化？

---

## 练习 5：Cache 驱逐策略模拟

### 目标

实现并对比 LRU 和 LFU 两种驱逐策略在 LLM Serving 场景下的表现。

### 实验代码

```python
"""
实验 5：Cache 驱逐策略对比。

模拟不同驱逐策略在 LLM Serving 工作负载下的 cache hit rate。
"""
from collections import OrderedDict, defaultdict
import random
import heapq


class LRUBlockCache:
    """LRU 驱逐策略的 block cache。"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def access(self, block_hash: int) -> bool:
        """访问一个 block。返回是否命中 cache。"""
        if block_hash in self.cache:
            self.cache.move_to_end(block_hash)
            return True
        
        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)  # 驱逐最久未使用的
        
        self.cache[block_hash] = True
        return False
    
    def __len__(self):
        return len(self.cache)


class LFUBlockCache:
    """LFU 驱逐策略的 block cache。"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.freq = defaultdict(int)
        self.time = 0
    
    def access(self, block_hash: int) -> bool:
        """访问一个 block。返回是否命中 cache。"""
        self.time += 1
        
        if block_hash in self.cache:
            self.freq[block_hash] += 1
            self.cache[block_hash] = self.time
            return True
        
        if len(self.cache) >= self.capacity:
            # 驱逐频率最低的（频率相同时驱逐最久未使用的）
            min_key = min(
                self.cache,
                key=lambda k: (self.freq[k], self.cache[k])
            )
            del self.cache[min_key]
            del self.freq[min_key]
        
        self.cache[block_hash] = self.time
        self.freq[block_hash] = 1
        return False
    
    def __len__(self):
        return len(self.cache)


def generate_workload(
    num_requests: int,
    num_system_prompts: int = 5,
    blocks_per_prompt: int = 10,
    popularity_skew: float = 2.0,
) -> list[list[int]]:
    """生成模拟工作负载。
    
    Args:
        num_requests: 请求总数
        num_system_prompts: 不同 system prompt 的数量
        blocks_per_prompt: 每个 system prompt 的 block 数
        popularity_skew: 流行度偏斜程度（Zipf 分布参数）
    
    Returns:
        每个请求的 block hash 序列列表
    """
    # 为每个 system prompt 生成固定的 block hash 序列
    prompt_blocks = {}
    for i in range(num_system_prompts):
        prompt_blocks[i] = [
            hash((i, j)) for j in range(blocks_per_prompt)
        ]
    
    # 按 Zipf 分布生成请求
    workload = []
    for _ in range(num_requests):
        # Zipf 分布选择 system prompt
        prompt_id = min(
            int(random.paretovariate(popularity_skew)),
            num_system_prompts - 1,
        )
        
        # 每个请求 = system prompt blocks + 随机 user blocks
        user_blocks = [random.randint(0, 10**9) for _ in range(3)]
        request_blocks = prompt_blocks[prompt_id] + user_blocks
        workload.append(request_blocks)
    
    return workload


def simulate_cache(
    cache_impl,
    workload: list[list[int]],
) -> dict:
    """模拟 cache 行为。
    
    注意：使用前缀匹配语义——在第一个 miss 处停止。
    """
    total_blocks = 0
    total_hits = 0
    total_misses = 0
    
    for request_blocks in workload:
        for block_hash in request_blocks:
            total_blocks += 1
            hit = cache_impl.access(block_hash)
            
            if hit:
                total_hits += 1
            else:
                total_misses += 1
                # 前缀匹配：第一个 miss 之后的 block 也要存入 cache
                # 但不计为 hit
    
    return {
        "total_blocks": total_blocks,
        "hits": total_hits,
        "misses": total_misses,
        "hit_rate": total_hits / total_blocks if total_blocks > 0 else 0,
    }


# 运行实验
print("=" * 60)
print("实验 5：Cache 驱逐策略对比")
print("=" * 60)

random.seed(42)

# 不同 cache 容量下的对比
cache_sizes = [20, 50, 100, 200]
num_requests = 1000

print(f"\n工作负载：{num_requests} 个请求")
print(f"  5 种不同 system prompt，每种 10 个 block")
print(f"  每个请求附加 3 个随机 user block")
print(f"  流行度：Zipf 分布（少数 prompt 非常热门）")

for cache_size in cache_sizes:
    workload = generate_workload(num_requests)
    
    lru_cache = LRUBlockCache(cache_size)
    lfu_cache = LFUBlockCache(cache_size)
    
    # 需要用同一份 workload
    lru_result = simulate_cache(lru_cache, workload)
    lfu_result = simulate_cache(lfu_cache, workload)
    
    print(f"\n--- Cache 容量: {cache_size} blocks ---")
    print(f"  LRU Hit Rate: {lru_result['hit_rate']:.1%}")
    print(f"  LFU Hit Rate: {lfu_result['hit_rate']:.1%}")
    
    if lfu_result["hit_rate"] > lru_result["hit_rate"]:
        diff = lfu_result["hit_rate"] - lru_result["hit_rate"]
        print(f"  → LFU 优势: +{diff:.1%}")
    else:
        diff = lru_result["hit_rate"] - lfu_result["hit_rate"]
        print(f"  → LRU 优势: +{diff:.1%}")


# 极端场景：一个非常热门的 prompt + 大量冷 prompt
print(f"\n\n{'=' * 60}")
print("极端场景：1 个热门 prompt + 99 个冷 prompt")
print("=" * 60)

workload_extreme = generate_workload(
    num_requests=2000,
    num_system_prompts=100,
    blocks_per_prompt=10,
    popularity_skew=1.5,  # 更极端的偏斜
)

for cache_size in [50, 100, 200]:
    lru_cache = LRUBlockCache(cache_size)
    lfu_cache = LFUBlockCache(cache_size)
    
    lru_result = simulate_cache(lru_cache, workload_extreme)
    lfu_result = simulate_cache(lfu_cache, workload_extreme)
    
    print(f"\n--- Cache 容量: {cache_size} blocks ---")
    print(f"  LRU Hit Rate: {lru_result['hit_rate']:.1%}")
    print(f"  LFU Hit Rate: {lfu_result['hit_rate']:.1%}")


print(f"\n\n{'=' * 60}")
print("结论")
print("=" * 60)
print("""
1. 在 cache 容量充足时，LRU 和 LFU 的差异不大
2. 在 cache 容量紧张时，LFU 对热门 prompt 的保护更好
3. 但 LFU 可能导致冷启动问题（新 prompt 频率低，容易被驱逐）
4. 实际系统中，LRU 因其简单性和较好的整体表现被广泛采用
5. ARC 等自适应策略可以结合两者优势，但实现更复杂
""")
```

### 思考题

1. 在什么工作负载模式下 LFU 明显优于 LRU？在什么情况下 LRU 更好？
2. 如果 cache 容量只有 10 个 block（非常小），两种策略的表现如何？
3. 尝试实现一个简单的 ARC 策略，并与 LRU/LFU 对比。
4. 在生产环境中，除了 hit rate，还需要考虑哪些因素来选择驱逐策略？（提示：考虑实现复杂度、并发安全性、时间复杂度）

---

## 总结

完成以上 5 个练习后，你应该能够：

1. **理解 cache 行为**：通过 API 响应字段判断 cache 是否命中
2. **测量性能影响**：量化 Prefix Caching 对 TTFT 和成本的影响
3. **优化 prompt 结构**：通过合理的 prompt 组织最大化 cache hit rate
4. **评估驱逐策略**：理解不同驱逐策略在不同工作负载下的表现
5. **成本分析**：根据 cache hit rate 预估使用 Prompt Caching 的成本节省

---

**返回：** [章节概述](README.md)
