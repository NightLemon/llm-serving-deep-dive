# Cache-Aware 路由与负载均衡

> 在多实例 LLM 推理部署中，路由策略直接决定了 KV Cache 命中率和整体吞吐量。
> 本节深入探讨如何从传统的 round-robin 路由进化到 cache-aware 路由。

## 1. 传统路由的问题

### 1.1 Round-Robin 路由的致命缺陷

在经典的 Web 服务中，round-robin 负载均衡简单高效——每个请求是无状态的，发到哪个后端节点都一样。但 LLM 推理服务有一个关键区别：**KV Cache 是有状态的**。

考虑以下场景：你有 4 个 vLLM replica，用户持续发送带有相同 system prompt 的请求：

```
请求 1: [system_prompt] + "什么是机器学习？"    → Replica 0
请求 2: [system_prompt] + "解释神经网络"        → Replica 1
请求 3: [system_prompt] + "什么是反向传播？"    → Replica 2
请求 4: [system_prompt] + "介绍 Transformer"   → Replica 3
```

每个 replica 都需要独立计算 `system_prompt` 的 KV Cache。如果 system prompt 有 2000 tokens，那么这 2000 tokens 的 prefill 计算被重复了 4 次。

**量化影响：**

```
假设条件:
- system_prompt = 2000 tokens
- 每个 token prefill 耗时 ~0.05ms（H100 上 70B 模型）
- 请求速率 = 100 req/s

Round-Robin 路由:
- 每个 replica 收到 25 req/s
- Cache hit rate ≈ 25%（仅同一 replica 内命中）
- 浪费的 prefill 计算 = 2000 × 0.05ms × 75 req/s = 7.5s/s GPU 时间

Cache-Aware 路由:
- 相同前缀的请求都发到同一 replica
- Cache hit rate ≈ 95%+
- 浪费的 prefill 计算 ≈ 0
```

### 1.2 Cache 命中率对延迟和成本的影响

在 Ch02 前缀缓存中我们知道，prefix cache hit 可以跳过整个 prefill 阶段。这意味着：

| 指标 | Cache Miss | Cache Hit | 改善 |
|------|-----------|-----------|------|
| TTFT (2K prefix) | ~100ms | ~5ms | **20x** |
| GPU 计算量 | 2K tokens prefill | 仅 attention 查找 | **>95% 减少** |
| API 成本 (OpenAI) | $15/M input tokens | $3.75/M cached tokens | **4x 降低** |

因此，路由策略不仅影响性能，还直接影响成本。

## 2. Cache-Aware 路由原理

### 2.1 核心思想

Cache-aware 路由的核心很简单：**将具有相同前缀的请求路由到同一个 replica**，以最大化 KV Cache 的复用。

```
                    ┌─────────────┐
                    │   Router    │
                    │ hash(prefix)│
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
         ┌────────┐  ┌────────┐  ┌────────┐
         │Replica0│  │Replica1│  │Replica2│
         │ hash=0 │  │ hash=1 │  │ hash=2 │
         │        │  │        │  │        │
         │System A│  │System B│  │System C│
         │ cached │  │ cached │  │ cached │
         └────────┘  └────────┘  └────────┘
```

### 2.2 OpenAI 的实践

根据 OpenAI 的公开文档和 API 行为分析，他们的 prompt caching 实现有几个关键特征：

**前 256 tokens hash 路由：**
- 取 prompt 的前 256 个 tokens 计算 hash 值
- 相同 hash 值的请求路由到同一组 GPU
- 256 tokens 通常覆盖了 system prompt 的核心部分

**`prompt_cache_key` 显式路由（推测机制）：**
```python
# OpenAI API 的 cached_tokens 返回示例
response.usage = {
    "prompt_tokens": 2048,
    "prompt_tokens_details": {
        "cached_tokens": 1792  # 前 1792 tokens 命中缓存
    },
    "completion_tokens": 150
}
```

**缓存粒度为 128 tokens：**
- 缓存以 128 token 为单位对齐
- prompt 长度必须 >= 1024 tokens 才会触发缓存（2025 年初已降低到更小的粒度）
- 缓存存活时间：5-10 分钟不活跃后自动淘汰

### 2.3 Anthropic 的 Prompt Caching

Anthropic 采用了显式的 cache control 标记：

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "你是一个专业的代码审查助手...(很长的 system prompt)...",
            "cache_control": {"type": "ephemeral"}  # 显式标记缓存
        }
    ],
    messages=[{"role": "user", "content": "审查以下代码..."}]
)
```

这种显式标记的方式让路由更简单——可以直接基于被标记为 `cache_control` 的内容块做 hash 路由。

## 3. 路由策略详解

### 3.1 Content-Hash 路由

最直接的实现：对 prompt 前缀做 hash，映射到 replica。

```python
import hashlib
from typing import List

class ContentHashRouter:
    def __init__(self, num_replicas: int, prefix_token_count: int = 256):
        self.num_replicas = num_replicas
        self.prefix_token_count = prefix_token_count
    
    def route(self, prompt_tokens: List[int]) -> int:
        """根据 prompt 前缀 hash 选择 replica"""
        # 取前 N 个 tokens 作为路由 key
        prefix = tuple(prompt_tokens[:self.prefix_token_count])
        
        # 计算 hash
        hash_value = hashlib.sha256(
            str(prefix).encode()
        ).hexdigest()
        
        # 映射到 replica
        replica_id = int(hash_value, 16) % self.num_replicas
        return replica_id
```

**优点：** 实现简单，相同前缀一定路由到同一 replica。

**缺点：**
- replica 数量变化时，几乎所有路由都会改变 → 缓存全部失效
- 无法感知 replica 的负载状态
- 热门前缀可能导致单个 replica 过载

### 3.2 Consistent Hashing（一致性哈希）

一致性哈希解决了 replica 增减时缓存大面积失效的问题。

```python
import bisect
import hashlib
from typing import List, Dict

class ConsistentHashRouter:
    def __init__(self, replicas: List[str], virtual_nodes: int = 150):
        self.ring: List[int] = []
        self.ring_to_replica: Dict[int, str] = {}
        self.virtual_nodes = virtual_nodes
        
        for replica in replicas:
            self._add_replica(replica)
        self.ring.sort()
    
    def _hash(self, key: str) -> int:
        return int(hashlib.sha256(key.encode()).hexdigest(), 16)
    
    def _add_replica(self, replica: str):
        """为每个 replica 创建多个虚拟节点"""
        for i in range(self.virtual_nodes):
            hash_val = self._hash(f"{replica}:{i}")
            self.ring.append(hash_val)
            self.ring_to_replica[hash_val] = replica
    
    def remove_replica(self, replica: str):
        """移除 replica 时，只有该 replica 的请求需要重新路由"""
        for i in range(self.virtual_nodes):
            hash_val = self._hash(f"{replica}:{i}")
            self.ring.remove(hash_val)
            del self.ring_to_replica[hash_val]
    
    def route(self, prompt_prefix: str) -> str:
        """查找顺时针方向最近的 replica"""
        if not self.ring:
            raise ValueError("No replicas available")
        
        hash_val = self._hash(prompt_prefix)
        idx = bisect.bisect_right(self.ring, hash_val)
        if idx == len(self.ring):
            idx = 0
        return self.ring_to_replica[self.ring[idx]]
```

**关键特性：**
- 增加一个 replica 时，只有 `1/N` 的请求需要重新路由
- 删除一个 replica 时，只有该 replica 的请求被重新分配
- 虚拟节点确保负载均匀分布

### 3.3 热度感知路由

在实际场景中，不同前缀的请求频率差异巨大。例如：

- System Prompt A（客服场景）：占 60% 请求
- System Prompt B（代码助手）：占 30% 请求
- 其他长尾前缀：占 10% 请求

如果仅用 hash 路由，Prompt A 对应的 replica 会过载。热度感知路由通过动态调整解决这个问题：

```python
import time
from collections import defaultdict
from typing import List, Dict, Optional

class HeatAwareRouter:
    def __init__(self, replicas: List[str]):
        self.replicas = replicas
        self.prefix_heat: Dict[str, float] = defaultdict(float)  # 前缀热度
        self.prefix_replicas: Dict[str, List[str]] = {}  # 前缀 → replica 列表
        self.replica_load: Dict[str, int] = {r: 0 for r in replicas}
        self.decay_factor = 0.95  # 热度衰减
        self.heat_threshold = 100  # 触发扩展的阈值
    
    def record_request(self, prefix_hash: str):
        """记录请求，更新热度"""
        self.prefix_heat[prefix_hash] = (
            self.prefix_heat[prefix_hash] * self.decay_factor + 1.0
        )
        
        # 如果热度超过阈值，为该前缀分配更多 replica
        if self.prefix_heat[prefix_hash] > self.heat_threshold:
            self._expand_prefix_replicas(prefix_hash)
    
    def _expand_prefix_replicas(self, prefix_hash: str):
        """为热门前缀分配额外的 replica"""
        current = self.prefix_replicas.get(prefix_hash, [])
        if len(current) >= len(self.replicas):
            return
        
        # 找到负载最低的 replica 加入
        available = [r for r in self.replicas if r not in current]
        if available:
            least_loaded = min(available, key=lambda r: self.replica_load[r])
            current.append(least_loaded)
            self.prefix_replicas[prefix_hash] = current
    
    def route(self, prefix_hash: str) -> str:
        """在该前缀的 replica 列表中选择负载最低的"""
        self.record_request(prefix_hash)
        
        candidates = self.prefix_replicas.get(prefix_hash)
        if not candidates:
            # 首次见到的前缀，用 hash 分配默认 replica
            default = self.replicas[hash(prefix_hash) % len(self.replicas)]
            self.prefix_replicas[prefix_hash] = [default]
            candidates = [default]
        
        # 在候选 replica 中选负载最低的
        chosen = min(candidates, key=lambda r: self.replica_load[r])
        self.replica_load[chosen] += 1
        return chosen
```

### 3.4 负载感知 + Cache 感知的混合路由

实际生产中，纯粹的 cache-aware 路由可能导致负载不均。最佳实践是将 cache 亲和性和负载均衡结合：

```python
class HybridRouter:
    """
    路由决策公式:
    score(replica) = α × cache_affinity + β × (1 - load_ratio) + γ × queue_slack
    
    其中:
    - cache_affinity: 该 replica 是否已缓存该前缀 (0 或 1)
    - load_ratio: 当前负载 / 最大容量
    - queue_slack: (最大队列 - 当前队列) / 最大队列
    """
    
    def __init__(self, replicas, alpha=0.6, beta=0.25, gamma=0.15):
        self.replicas = replicas
        self.alpha = alpha   # cache 亲和性权重
        self.beta = beta     # 负载均衡权重
        self.gamma = gamma   # 队列余量权重
        self.cache_registry = {}  # replica → set of cached prefix hashes
    
    def score(self, replica, prefix_hash, load_info):
        cache_hit = 1.0 if prefix_hash in self.cache_registry.get(replica, set()) else 0.0
        load_ratio = load_info[replica]['running'] / load_info[replica]['max_capacity']
        queue_slack = 1.0 - (load_info[replica]['waiting'] / load_info[replica]['max_queue'])
        
        return (self.alpha * cache_hit + 
                self.beta * (1.0 - load_ratio) + 
                self.gamma * queue_slack)
    
    def route(self, prefix_hash, load_info):
        scores = {r: self.score(r, prefix_hash, load_info) for r in self.replicas}
        return max(scores, key=scores.get)
```

## 4. 工程实现

### 4.1 Nginx 自定义 Hash 路由

在 Nginx 中可以基于请求体的特定字段做 hash 路由：

```nginx
upstream vllm_backends {
    # 使用 consistent hash
    hash $request_body_prefix consistent;
    
    server vllm-replica-0:8000;
    server vllm-replica-1:8000;
    server vllm-replica-2:8000;
    server vllm-replica-3:8000;
}

# 需要配合 Lua 模块提取 prompt 前缀
# 使用 OpenResty (nginx + lua)
server {
    listen 80;
    
    location /v1/chat/completions {
        # 提取前缀用于 hash 路由
        set_by_lua_block $request_body_prefix {
            local cjson = require "cjson"
            ngx.req.read_body()
            local body = ngx.req.get_body_data()
            if not body then return "default" end
            
            local ok, data = pcall(cjson.decode, body)
            if not ok then return "default" end
            
            -- 提取 system message 作为路由 key
            local messages = data["messages"]
            if messages and messages[1] and messages[1]["role"] == "system" then
                local content = messages[1]["content"]
                -- 取前 200 字符做 hash
                return string.sub(content, 1, 200)
            end
            return "default"
        }
        
        proxy_pass http://vllm_backends;
        proxy_set_header Content-Type "application/json";
    }
}
```

### 4.2 Envoy 配置示例

Envoy 支持更灵活的 hash policy：

```yaml
# envoy.yaml
static_resources:
  listeners:
    - name: llm_listener
      address:
        socket_address:
          address: 0.0.0.0
          port_value: 8080
      filter_chains:
        - filters:
            - name: envoy.filters.network.http_connection_manager
              typed_config:
                "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
                route_config:
                  virtual_hosts:
                    - name: llm_service
                      domains: ["*"]
                      routes:
                        - match:
                            prefix: "/v1/"
                          route:
                            cluster: vllm_cluster
                            hash_policy:
                              - header:
                                  header_name: "x-prompt-cache-key"
                              - connection_properties:
                                  source_ip: true  # fallback
                
  clusters:
    - name: vllm_cluster
      type: STRICT_DNS
      lb_policy: RING_HASH
      ring_hash_lb_config:
        minimum_ring_size: 1024
      load_assignment:
        cluster_name: vllm_cluster
        endpoints:
          - lb_endpoints:
              - endpoint:
                  address:
                    socket_address:
                      address: vllm-replica-0
                      port_value: 8000
              - endpoint:
                  address:
                    socket_address:
                      address: vllm-replica-1
                      port_value: 8000
```

客户端在发送请求时注入 header：

```python
import hashlib
import httpx

def send_with_cache_key(messages, base_url="http://envoy:8080"):
    # 计算 prompt cache key
    system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
    cache_key = hashlib.sha256(system_msg.encode()).hexdigest()[:16]
    
    response = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={"model": "meta-llama/Llama-3.1-70B", "messages": messages},
        headers={"x-prompt-cache-key": cache_key}
    )
    return response.json()
```

### 4.3 vLLM 多实例路由方案

vLLM 本身不内置多实例路由功能，但社区有几种成熟的方案：

**方案 1：使用 SGLang Router**

SGLang 项目提供了一个专门的 cache-aware router：

```bash
# 启动多个 vLLM 实例
python -m vllm.entrypoints.openai.api_server --port 8000 &
python -m vllm.entrypoints.openai.api_server --port 8001 &
python -m vllm.entrypoints.openai.api_server --port 8002 &

# 使用 SGLang 的 router（支持 cache-aware 路由）
python -m sglang.srt.router \
    --worker-urls http://localhost:8000 http://localhost:8001 http://localhost:8002 \
    --policy cache-aware \
    --port 9000
```

**方案 2：自定义 FastAPI Router**

```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx
import hashlib

app = FastAPI()

REPLICAS = [
    "http://vllm-0:8000",
    "http://vllm-1:8000",
    "http://vllm-2:8000",
]

# 维护每个 replica 的健康状态和负载
replica_status = {url: {"healthy": True, "active_requests": 0} for url in REPLICAS}

def get_prefix_hash(messages: list, num_chars: int = 500) -> str:
    """提取 messages 前缀并计算 hash"""
    prefix_parts = []
    char_count = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):  # multimodal content
            content = str(content)
        remaining = num_chars - char_count
        if remaining <= 0:
            break
        prefix_parts.append(content[:remaining])
        char_count += len(content[:remaining])
    
    prefix = "||".join(prefix_parts)
    return hashlib.sha256(prefix.encode()).hexdigest()

def select_replica(prefix_hash: str) -> str:
    """基于一致性哈希选择健康的 replica"""
    hash_val = int(prefix_hash, 16)
    healthy_replicas = [url for url in REPLICAS if replica_status[url]["healthy"]]
    
    if not healthy_replicas:
        raise Exception("No healthy replicas")
    
    # 一致性哈希选择
    idx = hash_val % len(healthy_replicas)
    chosen = healthy_replicas[idx]
    
    # 如果选中的 replica 负载过高，fallback 到最空闲的
    if replica_status[chosen]["active_requests"] > 50:
        chosen = min(healthy_replicas, key=lambda r: replica_status[r]["active_requests"])
    
    return chosen

@app.post("/v1/chat/completions")
async def proxy_chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    prefix_hash = get_prefix_hash(messages)
    target = select_replica(prefix_hash)
    
    replica_status[target]["active_requests"] += 1
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(f"{target}/v1/chat/completions", json=body)
            return resp.json()
    finally:
        replica_status[target]["active_requests"] -= 1
```

## 5. 性能评估与调优

### 5.1 关键指标

评估路由策略效果时，关注以下指标：

| 指标 | 含义 | 目标 |
|------|------|------|
| Prefix Cache Hit Rate | 前缀缓存命中率 | > 80% |
| Load Imbalance Ratio | 最忙 replica 负载 / 平均负载 | < 1.3 |
| TTFT P99 | 99 分位首 token 延迟 | 取决于 SLA |
| Routing Overhead | 路由决策延迟 | < 1ms |
| Cache Churn Rate | 缓存淘汰频率 | 低 |

### 5.2 调优建议

**选择合适的前缀长度：**
- 太短（< 64 tokens）：不同 prompt 可能 hash 碰撞
- 太长（> 1024 tokens）：hash 计算开销增大，且降低命中灵活性
- 推荐：128-256 tokens，覆盖 system prompt 核心部分

**虚拟节点数量：**
- 一致性哈希中，虚拟节点越多，负载越均匀
- 推荐：每个物理 replica 100-200 个虚拟节点

**负载过高时的 fallback：**
- 当首选 replica 负载超过阈值时，应 fallback 到次优 replica
- cache miss 的代价远小于排队等待的代价

## 6. 总结

| 路由策略 | 适用场景 | Cache 命中率 | 负载均衡性 | 实现复杂度 |
|----------|---------|-------------|-----------|-----------|
| Round-Robin | 无状态服务 | 低 | 高 | 低 |
| Content-Hash | 前缀种类少 | 高 | 中 | 低 |
| Consistent Hash | 需要弹性伸缩 | 高 | 中 | 中 |
| 热度感知 | 流量分布不均 | 高 | 高 | 高 |
| 混合路由 | 生产环境推荐 | 最高 | 最高 | 高 |

**核心原则：** 在生产环境中，路由策略应该同时考虑 cache 亲和性和负载均衡。纯粹追求 cache 命中率可能导致热点 replica 过载，纯粹追求负载均衡会浪费大量 prefill 计算。混合路由通过加权评分函数在两者之间取得平衡。

---

> **延伸阅读：**
> - SGLang RadixAttention 论文中关于 cache-aware scheduling 的讨论
> - [vLLM 多实例部署文档](https://docs.vllm.ai/en/latest/)
> - Envoy Ring Hash Load Balancer 官方文档
