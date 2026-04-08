# OpenAI Extended Prompt Caching 分析

> OpenAI 于 2024 年底推出 Extended Prompt Caching，将 KV Cache 的保留时间从分钟级延长到 24 小时。本节分析其工作原理、技术推测和实际使用方式。

## 1. Prompt Caching 基础回顾

在分析 Extended Caching 之前，先回顾标准 Prompt Caching 的工作原理。

### 1.1 标准 Prompt Caching

当你向 OpenAI API 发送请求时，如果 prompt 的前缀与之前的请求相同（至少 1024 tokens），OpenAI 会复用之前计算好的 KV Cache：

```
请求 1: [System Prompt (5000 tokens)] [User: 什么是机器学习?]
         ↑ 计算 KV Cache 并缓存

请求 2: [System Prompt (5000 tokens)] [User: 什么是深度学习?]
         ↑ 命中缓存，跳过 prefill   ↑ 只需要计算这部分
```

**标准缓存的特征：**

| 属性 | 说明 |
|------|------|
| 触发条件 | prompt 前缀匹配 >= 1024 tokens |
| 匹配粒度 | 128 token 为一个 block |
| 保留时间 | 5-10 分钟（随负载动态变化） |
| 存储位置 | GPU HBM（推测） |
| 费用 | 缓存命中的 token 按 50% 折扣计费 |
| 自动生效 | 是，无需额外配置 |

### 1.2 标准缓存的局限

5-10 分钟的保留时间在很多场景下不够用：

```
场景 1：客服系统
  - 用户可能 10 分钟后才发下一条消息
  - 每次重新计算 system prompt + 历史对话的 KV Cache

场景 2：代码助手
  - 开发者在多个文件间切换，间隔可能超过 10 分钟
  - 大量代码上下文需要重复 prefill

场景 3：文档 QA
  - 同一份文档被不同用户反复查询
  - 文档 embedding 到 prompt 中的成本很高
```

## 2. Extended Prompt Caching

### 2.1 核心机制

Extended Caching 将 KV Cache 的保留时间延长到最长 **24 小时**：

```json
{
    "model": "gpt-4o",
    "messages": [...],
    "prompt_cache_retention": "24h"    // 关键配置
}
```

**与标准缓存的对比：**

| 维度 | 标准缓存 | Extended Caching |
|------|---------|-----------------|
| 保留时间 | 5-10 分钟 | 最长 24 小时 |
| 存储位置 | GPU HBM（推测） | GPU-local NVMe SSD（推测） |
| 存储内容 | KV tensors | KV tensors（不存原始文本） |
| 费用 | 缓存命中 50% 折扣 | 缓存命中 50% 折扣 + 存储费用 |
| ZDR 兼容 | 是 | 是 |
| 需要配置 | 否（自动） | 是（显式声明） |

### 2.2 ZDR (Zero Data Retention) 合规

一个重要的设计决策——Extended Caching **只存储 KV tensors，不存储原始 prompt 文本**：

```
原始 Prompt: "请分析以下患者病历：张三，男，45岁..."
                    ↓ Prefill 计算
KV Tensors:  [[[0.23, -0.15, ...], [0.87, 0.34, ...], ...]]  ← 只存这个
                    ↓
存储到 NVMe: 纯数值矩阵，无法反向推断原始文本
```

**为什么 KV tensors 是安全的？**

1. KV tensors 是 attention 机制的中间表示，经过多层非线性变换
2. 从 KV tensors 反向恢复原始文本在计算上不可行（不是简单的可逆映射）
3. 这使得 Extended Caching 可以兼容 ZDR 策略——即使在最严格的数据合规要求下也可以使用

### 2.3 缓存标识与定价

Extended Caching 的使用可以通过 API 响应中的 `usage` 字段确认：

```json
{
    "usage": {
        "prompt_tokens": 8192,
        "completion_tokens": 256,
        "prompt_tokens_details": {
            "cached_tokens": 7168,     // 命中缓存的 tokens 数量
            "audio_tokens": 0
        }
    }
}
```

**定价模型（以 GPT-4o 为例，2025 年）：**

| 类型 | 价格 |
|------|------|
| Input tokens（无缓存） | $2.50 / 1M tokens |
| Cached input tokens | $1.25 / 1M tokens (50% 折扣) |
| Extended Caching 存储 | 额外计费（按缓存占用时间） |
| Output tokens | $10.00 / 1M tokens |

## 3. 技术推测

以下内容基于 OpenAI 公开的信息、API 行为观察和工程推理。OpenAI 并未公开 Extended Caching 的完整实现细节。

### 3.1 存储层级推测

```
请求到达
    │
    ▼
┌──────────────────────────────┐
│  Step 1: 检查 GPU HBM Cache │  ← 标准缓存（5-10 分钟）
│  匹配？→ 直接使用           │
└──────────┬───────────────────┘
           │ 未命中
           ▼
┌──────────────────────────────┐
│  Step 2: 检查 GPU-local     │  ← Extended Caching
│  NVMe SSD                   │
│  匹配？→ 加载到 GPU HBM     │
│  NVMe → GPU 带宽: ~7 GB/s   │
│  加载延迟: ~100-500 ms       │
└──────────┬───────────────────┘
           │ 未命中
           ▼
┌──────────────────────────────┐
│  Step 3: 完整 Prefill       │  ← 从头计算
│  延迟: ~1-10 秒（取决于长度）│
└──────────────────────────────┘
```

为什么推测使用 GPU-local NVMe 而不是 CPU DRAM？
- 24 小时保留需要大量存储，NVMe 容量更大（TB 级）
- NVMe 持久化存储，即使进程重启数据也不会丢失
- GPU 服务器通常配备高速 NVMe（PCIe 5.0 x4, ~7 GB/s）
- CPU DRAM 虽然带宽更高，但容量有限且更贵

### 3.2 Hash-Based Routing

为了实现高缓存命中率，OpenAI 很可能使用了 **hash-based routing**——将具有相同 prompt 前缀的请求路由到同一台物理机器：

```
                          Hash(prompt_prefix)
请求 ──────────────────────────┬──────────────────────────
                               │
                               ▼
                    ┌──────────────────┐
                    │   Load Balancer  │
                    │   (Consistent    │
                    │    Hashing)      │
                    └───┬───┬───┬─────┘
                        │   │   │
                   ┌────┘   │   └────┐
                   ▼        ▼        ▼
              ┌────────┐┌────────┐┌────────┐
              │ GPU    ││ GPU    ││ GPU    │
              │ Node A ││ Node B ││ Node C │
              │        ││        ││        │
              │ NVMe:  ││ NVMe:  ││ NVMe:  │
              │ Cache  ││ Cache  ││ Cache  │
              │ for    ││ for    ││ for    │
              │ hash=A ││ hash=B ││ hash=C │
              └────────┘└────────┘└────────┘
```

**Hash 计算：**

```python
# 推测的路由逻辑
def route_request(prompt_tokens, prompt_cache_key=None):
    # 取 prompt 的前 N 个 tokens 作为 prefix
    prefix = prompt_tokens[:PREFIX_LENGTH]
    
    # 计算 hash
    if prompt_cache_key:
        # 用户指定了 cache key，优先使用
        hash_input = f"{prompt_cache_key}:{hash(prefix)}"
    else:
        hash_input = hash(prefix)
    
    # Consistent hashing 选择目标节点
    target_node = consistent_hash_ring.get_node(hash_input)
    return target_node
```

### 3.3 prompt_cache_key

`prompt_cache_key` 是 OpenAI 提供的一个高级配置，允许用户显式控制缓存路由：

```json
{
    "model": "gpt-4o",
    "messages": [...],
    "prompt_cache_key": "customer_support_v2",
    "prompt_cache_retention": "24h"
}
```

**使用场景：**

```python
# 场景：同一个 system prompt 用于多个对话
# 使用 prompt_cache_key 确保这些请求路由到同一台机器

system_prompt = "You are a helpful customer support agent..."

# 对话 1
response1 = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "我的订单在哪里？"}
    ],
    prompt_cache_key="cs_agent_v2",
    prompt_cache_retention="24h",
)

# 对话 2（不同用户，但 system prompt 相同）
response2 = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "如何退货？"}
    ],
    prompt_cache_key="cs_agent_v2",  # 相同的 cache key
    prompt_cache_retention="24h",
)
# → 第二个请求大概率命中第一个请求的 KV Cache
```

### 3.4 15 req/min 速率限制

OpenAI 对每个 `prefix + prompt_cache_key` 组合施加了约 **15 req/min** 的速率限制。

**为什么需要这个限制？**

```
假设没有限制：
  - 1000 个请求使用相同的 cache key
  - Consistent hashing 将它们全部路由到同一台机器
  - 单机过载，延迟飙升

有限制后：
  - 每个 cache key 最多 15 req/min
  - 超出部分可能被路由到其他节点（cache miss 但负载均衡）
  - 或者返回 429 错误
```

这个限制揭示了一个重要的架构权衡：**缓存效率 vs 负载均衡**。为了获得高缓存命中率，需要将相似请求集中到同一台机器；但过度集中会导致热点问题。

### 3.5 缓存失效与更新

推测的缓存失效策略：

```
缓存失效条件：
1. 时间过期（24 小时 TTL）
2. 存储空间压力（NVMe 使用率过高时 LRU 驱逐）
3. 模型更新（模型版本变化导致所有缓存失效）
4. 显式失效（用户可能通过 API 触发，目前未公开）

缓存键的构成（推测）：
cache_key = hash(
    model_version,           # 模型版本
    prompt_prefix_tokens,    # prompt 前缀的 token IDs
    prompt_cache_key,        # 用户指定的 cache key
    quantization_config,     # 量化配置（如有）
)
```

## 4. 与 Anthropic / Google 的对比

### 4.1 Anthropic Prompt Caching

Anthropic 的 Prompt Caching 采用了不同的设计哲学：

```python
# Anthropic 的 Prompt Caching 需要显式标记 cache breakpoints
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "Very long system prompt...",
            "cache_control": {"type": "ephemeral"}  # 显式标记
        }
    ],
    messages=[...]
)
```

**对比表：**

| 维度 | OpenAI | Anthropic | Google (Gemini) |
|------|--------|-----------|-----------------|
| 标记方式 | 自动前缀匹配 | 显式 `cache_control` | Context Caching API |
| 最小前缀 | 1024 tokens | 1024 tokens (Sonnet) / 2048 (Haiku) | 不适用（独立 API） |
| 标准保留时间 | 5-10 分钟 | 5 分钟 | 用户指定 TTL |
| Extended 保留 | 24 小时 | 无 | 用户指定（最长数小时） |
| 计费模式 | 缓存命中 50% 折扣 | 缓存写入 25% 加价 + 命中 90% 折扣 | 按缓存存储时间计费 |
| Cache Key | prompt_cache_key | 无 | 无 |
| ZDR 兼容 | 是 | 否（有 cache 就有数据保留） | 否 |

### 4.2 Google Gemini Context Caching

Google 的 Context Caching 是一个独立的 API，更像是一个显式的 KV Cache 存储服务：

```python
# Google Context Caching（概念性）
import google.generativeai as genai

# Step 1: 创建 cached content（显式存储）
cache = genai.caching.CachedContent.create(
    model="gemini-1.5-pro",
    display_name="support_docs",
    system_instruction="You are a support agent...",
    contents=[large_document],
    ttl=datetime.timedelta(hours=2),  # 显式指定 TTL
)

# Step 2: 使用缓存进行推理
model = genai.GenerativeModel.from_cached_content(cache)
response = model.generate_content("用户问题...")

# Step 3: 不再需要时删除缓存
cache.delete()
```

**Google 方案的特点：**
- 显式生命周期管理（创建、使用、删除）
- 按存储时间计费（不仅仅是使用次数）
- 更适合长期、固定的上下文（如文档库）
- 不适合动态对话场景

### 4.3 架构设计哲学对比

```
OpenAI：隐式 + 自动化
  "把 prompt 发过来，我们自动帮你缓存"
  → 用户体验最简单
  → 缓存行为不完全透明

Anthropic：显式 + 控制
  "你告诉我哪些内容需要缓存"
  → 用户有精确控制
  → 需要修改代码来利用缓存

Google：独立服务 + 完全显式
  "先创建一个缓存对象，然后引用它"
  → 类似传统缓存服务
  → 生命周期完全由用户管理
```

## 5. 配置方式与最佳实践

### 5.1 基本配置

```python
from openai import OpenAI

client = OpenAI()

# 基本 Extended Caching 配置
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": long_system_prompt},  # >= 1024 tokens
        {"role": "user", "content": user_query}
    ],
    prompt_cache_retention="24h",  # 启用 Extended Caching
)

# 检查缓存命中情况
cached = response.usage.prompt_tokens_details.cached_tokens
total = response.usage.prompt_tokens
print(f"缓存命中率: {cached/total*100:.1f}%")
```

### 5.2 使用 prompt_cache_key

```python
# 场景：多租户系统，每个租户有独立的 system prompt
def create_tenant_response(tenant_id: str, user_message: str):
    tenant_config = load_tenant_config(tenant_id)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": tenant_config.system_prompt},
            {"role": "user", "content": user_message}
        ],
        prompt_cache_key=f"tenant_{tenant_id}",  # 按租户分组路由
        prompt_cache_retention="24h",
    )
    return response
```

### 5.3 最佳实践

**1. prompt 结构优化：**

```
好的做法：将不变的部分放在 prompt 前面
┌────────────────────────────┐
│ System Prompt (不变，5000 tokens)  │ ← 这部分被缓存
├────────────────────────────┤
│ Few-shot Examples (不变)           │ ← 这部分也被缓存
├────────────────────────────┤
│ 用户上下文 (每次变化)              │ ← 这部分不被缓存
├────────────────────────────┤
│ 用户问题 (每次变化)               │
└────────────────────────────┘

不好的做法：变化的部分在前面
┌────────────────────────────┐
│ 当前时间戳 (每次变化)              │ ← 破坏前缀匹配！
├────────────────────────────┤
│ System Prompt (不变)               │
│ ...                                │
└────────────────────────────┘
```

**2. prompt_cache_key 分组策略：**

```python
# 好：按功能/场景分组
prompt_cache_key = "code_review"      # 代码审查场景
prompt_cache_key = "customer_support"  # 客服场景
prompt_cache_key = f"doc_qa_{doc_id}"  # 按文档分组

# 不好：过于细粒度
prompt_cache_key = f"user_{user_id}_{session_id}"  # 每个 session 独立
# → 缓存几乎不会被复用
```

**3. 监控缓存效果：**

```python
import statistics

cache_hit_rates = []

for query in queries:
    response = client.chat.completions.create(...)
    cached = response.usage.prompt_tokens_details.cached_tokens
    total = response.usage.prompt_tokens
    cache_hit_rates.append(cached / total if total > 0 else 0)

print(f"平均缓存命中率: {statistics.mean(cache_hit_rates)*100:.1f}%")
print(f"中位数缓存命中率: {statistics.median(cache_hit_rates)*100:.1f}%")
print(f"预估节省: ${sum(cache_hit_rates) * cost_per_token * total_tokens:.2f}")
```

## 6. 适用场景分析

### 6.1 高价值场景

| 场景 | 为什么适合 Extended Caching | 预估节省 |
|------|--------------------------|---------|
| 长 system prompt | prompt 不变，多次对话复用 | 40-50% |
| 文档 QA | 同一文档被反复查询 | 50-80% |
| 代码助手 | 代码库上下文在 session 内不变 | 30-50% |
| 多轮对话 | 对话历史逐渐积累，前缀持续匹配 | 20-40% |
| Few-shot Learning | 大量 examples 不变 | 50-70% |

### 6.2 不适合的场景

| 场景 | 为什么不适合 |
|------|-------------|
| 每次 prompt 完全不同 | 无前缀匹配可能 |
| 短 prompt (<1024 tokens) | 低于缓存最小阈值 |
| 高度动态的 system prompt | 每次变化导致缓存失效 |
| 对延迟极度敏感 | NVMe → GPU 加载有一定延迟 |

## 7. 成本优化模型

### 7.1 盈亏平衡分析

```python
def calculate_extended_cache_savings(
    prompt_length: int,          # tokens
    queries_per_day: int,
    cache_hit_rate: float,       # 0-1
    input_price: float,          # $/M tokens
    cached_price: float,         # $/M tokens
    storage_cost_per_day: float, # $/day for extended caching
):
    """计算 Extended Caching 的每日净节省"""
    
    # 无缓存的每日成本
    daily_cost_no_cache = (
        prompt_length * queries_per_day * input_price / 1_000_000
    )
    
    # 有缓存的每日成本
    cached_tokens = prompt_length * cache_hit_rate
    uncached_tokens = prompt_length * (1 - cache_hit_rate)
    daily_cost_with_cache = (
        (cached_tokens * cached_price + uncached_tokens * input_price)
        * queries_per_day / 1_000_000
        + storage_cost_per_day
    )
    
    savings = daily_cost_no_cache - daily_cost_with_cache
    return savings

# 示例计算
savings = calculate_extended_cache_savings(
    prompt_length=10000,       # 10K tokens prompt
    queries_per_day=1000,
    cache_hit_rate=0.7,        # 70% 命中率
    input_price=2.50,          # $2.50/M tokens
    cached_price=1.25,         # $1.25/M tokens
    storage_cost_per_day=0.50, # $0.50/day 存储费
)
print(f"每日净节省: ${savings:.2f}")
# → 每日净节省: $8.25
```

## 8. 小结

| 要点 | 说明 |
|------|------|
| Extended Caching | 将 KV Cache 保留时间从分钟级延长到 24 小时 |
| 存储推测 | 使用 GPU-local NVMe SSD 存储 KV tensors |
| Hash Routing | 通过 prompt 前缀 hash 将请求路由到同一台机器 |
| ZDR 合规 | 只存 KV tensors（不可逆的中间表示），不存原始文本 |
| prompt_cache_key | 显式控制缓存路由，提高命中率 |
| 速率限制 | 15 req/min per prefix+key，避免单机过载 |
| 最佳实践 | 将不变内容放在 prompt 前部，合理分组 cache key |

---

**下一节：** [LMCache 集成](04-lmcache.md) —— 了解如何在自建推理服务中实现类似的 KV Cache 持久化和共享。
