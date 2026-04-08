# API 提供商 Prompt Caching 实践

> 本节详细对比 OpenAI、Anthropic、Google Gemini 三大 API 提供商的 Prompt Caching 方案，深入分析其触发条件、定价策略、设计约束，并提供最大化 cache hit 的工程实践指南。

## 1. 三大提供商方案总览

### 1.1 对比矩阵

| 特性 | OpenAI | Anthropic | Google Gemini |
|------|--------|-----------|---------------|
| **方案名称** | Prompt Caching（自动） | Prompt Caching | Context Caching |
| **触发方式** | 全自动 | 自动（部分场景需手动 breakpoint） | 隐式（自动）+ 显式（API 手动创建） |
| **最小 token 数** | 1024 | 1024 (Sonnet/Opus) / 2048 (Haiku) | 1024-4096（因模型而异） |
| **增量粒度** | 128 tokens | 1 token（但以 block 对齐） | 不公开 |
| **Cache 寿命** | 5-10 min | 5 min（自动刷新） | 隐式：自动管理 / 显式：最长 48h |
| **Cache write 费用** | 无额外费用 | 1.25x 基础 input 价格 | 隐式：无 / 显式：同 input |
| **Cache read 折扣** | 50% off | 90% off | 隐式：有折扣 / 显式：75% off |
| **跨请求共享** | 同一 organization | 同一 workspace | 同一项目 |
| **支持内容类型** | messages, system, tools | system, messages, tools, images | 所有内容 |
| **API 字段** | `cached_tokens` | `cache_read_input_tokens`, `cache_creation_input_tokens` | `cached_content_token_count` |

### 1.2 设计哲学差异

**OpenAI：** "Zero-config"——用户无需做任何配置，系统自动检测可缓存的前缀。没有额外的 write 费用，降低了使用门槛。折扣力度适中（50% off）。

**Anthropic：** "显式可控"——提供 `cache_control` breakpoint 让用户标记缓存边界（虽然很多场景也能自动触发）。Write 费用 1.25x 换来 read 时 90% 的折扣——对高频复用场景非常有利。

**Google Gemini：** "双轨制"——隐式缓存全自动且免费，显式缓存（Context Caching API）允许用户手动创建长生命周期的缓存，适合需要持久化缓存的场景（如大型文档分析）。

## 2. OpenAI Prompt Caching 深度分析

### 2.1 工作原理

OpenAI 的 Prompt Caching 完全自动，不需要任何 API 参数或标记：

```python
from openai import OpenAI
client = OpenAI()

# 第一次请求：建立 cache
response1 = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": very_long_system_prompt},  # 2000 tokens
        {"role": "user", "content": "What is machine learning?"}
    ]
)

# 第二次请求：相同的 system prompt → cache hit
response2 = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": very_long_system_prompt},  # cache hit!
        {"role": "user", "content": "Explain neural networks."}
    ]
)
```

### 2.2 响应中的 Cache 信息

```json
{
  "usage": {
    "prompt_tokens": 2150,
    "completion_tokens": 250,
    "total_tokens": 2400,
    "prompt_tokens_details": {
      "cached_tokens": 2048
    }
  }
}
```

`cached_tokens` 表示命中 cache 的 token 数。注意这个值是 **128 的整数倍**——OpenAI 以 128 token 为增量粒度。

### 2.3 缓存规则

1. **前 1024 token 是缓存的最小单位**。如果 prompt 不足 1024 token，不会触发 cache
2. **超过 1024 后，每 128 token 为一个增量**。例如 1200 token 的 prompt，前 1152 = 1024 + 128 可以被缓存
3. **Cache 按前缀匹配**——两个请求必须从位置 0 开始完全相同
4. **Cache 在同一 organization 内共享**——不同项目、不同 API key（同一 org）可以共享
5. **寿命约 5-10 分钟**——无活跃引用后逐渐失效
6. **支持 tools 和 function definitions**——作为前缀的一部分参与 caching

### 2.4 定价

OpenAI 的 cache 策略是最简单的：

| 模型 | 正常 input（$/M tokens） | Cached input（$/M tokens） | 折扣 |
|------|-------------------------|--------------------------|------|
| GPT-4o | $2.50 | $1.25 | 50% |
| GPT-4o-mini | $0.15 | $0.075 | 50% |
| o1 | $15.00 | $7.50 | 50% |
| o3-mini | $1.10 | $0.55 | 50% |

**无 cache write 溢价**，任何复用都是纯赚。

## 3. Anthropic Prompt Caching 深度分析

### 3.1 Cache Control Breakpoint

Anthropic 允许用户在 message 中插入 `cache_control` 标记来指定缓存边界：

```python
import anthropic
client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": very_long_system_prompt,
            "cache_control": {"type": "ephemeral"}  # 标记缓存边界
        }
    ],
    messages=[
        {"role": "user", "content": "What is machine learning?"}
    ]
)
```

`"cache_control": {"type": "ephemeral"}` 告诉 API：**到这里为止的所有内容应该被缓存**。

### 3.2 多重 Breakpoint

可以设置多个 breakpoint，形成层级缓存：

```python
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": system_instructions,       # 第一层缓存
            "cache_control": {"type": "ephemeral"}
        }
    ],
    tools=[
        {
            "name": "search",
            "description": "Search the web...",
            "input_schema": {...},
            "cache_control": {"type": "ephemeral"}  # 第二层缓存
        }
    ],
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": long_document,         # 第三层缓存
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        },
        {"role": "assistant", "content": "I've read the document."},
        {"role": "user", "content": "Summarize it."}  # 不缓存
    ]
)
```

### 3.3 响应中的 Cache 信息

```json
{
  "usage": {
    "input_tokens": 50,
    "output_tokens": 200,
    "cache_creation_input_tokens": 5000,
    "cache_read_input_tokens": 3000
  }
}
```

四个字段的含义：

| 字段 | 含义 |
|------|------|
| `input_tokens` | 未被缓存的 input token 数（cache 边界之后的内容） |
| `output_tokens` | 输出 token 数 |
| `cache_creation_input_tokens` | **本次请求新写入缓存** 的 token 数（1.25x 计费） |
| `cache_read_input_tokens` | **从缓存中读取** 的 token 数（0.1x 计费） |

判断 cache 状态：
- `cache_creation_input_tokens > 0`：cache miss（首次写入）
- `cache_read_input_tokens > 0`：cache hit（复用）
- 两者都 > 0：部分命中（前面的 breakpoint 命中，后面的 miss）

### 3.4 TTL 与刷新机制

Anthropic 的 cache TTL 为 5 分钟，但有 **自动刷新** 机制：

- 每次 cache hit 都会重置 TTL 计时器
- 只要在 5 分钟内有任何请求命中同一缓存，缓存就不会过期
- 实际上对于活跃的应用，缓存可以保持数小时甚至数天

### 3.5 最小 Token 数要求

| 模型 | 最小缓存 token 数 |
|------|-------------------|
| Claude Opus 4 | 1024 |
| Claude Sonnet 4 | 1024 |
| Claude Haiku 3.5 | 2048 |

如果 breakpoint 前的内容不足最小 token 数，缓存不会生效（但不会报错）。

### 3.6 定价（以 Claude Sonnet 4 为例）

| 类型 | 价格（$/M tokens） | 倍率 |
|------|--------------------|----|
| Base input | $3.00 | 1.0x |
| Cache write | $3.75 | 1.25x |
| Cache read | $0.30 | 0.1x |
| Output | $15.00 | - |

## 4. Google Gemini Context Caching 深度分析

### 4.1 双轨制：隐式 vs 显式

**隐式缓存（Implicit Caching）：**
- 自动触发，无需任何代码修改
- Google 自动检测重复的前缀并缓存
- 无额外 write 费用
- 折扣不透明（Google 称"当适用时提供折扣"）
- 支持 Gemini 1.5 Pro/Flash 及以上

**显式缓存（Explicit Caching / Context Caching API）：**
- 用户通过 API 手动创建缓存
- 指定 TTL（默认 1 小时，最长 48 小时）
- 存储费用 + 使用折扣
- 适合长文档分析、代码库理解等场景

### 4.2 显式缓存 API

```python
import google.generativeai as genai

# Step 1: 创建缓存
cache = genai.caching.CachedContent.create(
    model="models/gemini-1.5-pro-latest",
    display_name="my_long_document",
    system_instruction="You are an expert analyst.",
    contents=[
        # 要缓存的长内容
        {"role": "user", "parts": [very_long_document]}
    ],
    ttl=datetime.timedelta(hours=2),  # 缓存 2 小时
)

# Step 2: 使用缓存进行对话
model = genai.GenerativeModel.from_cached_content(cached_content=cache)
response = model.generate_content("Summarize the key findings.")

# Step 3: 查看缓存状态
print(cache.usage_metadata)
# total_token_count: 50000
# cached_token_count: 48000

# Step 4: 手动删除缓存
cache.delete()
```

### 4.3 显式缓存的定价

| 模型 | Input（$/M tokens） | Cached input（$/M tokens） | 存储（$/M tokens/hour） |
|------|---------------------|--------------------------|----------------------|
| Gemini 1.5 Pro | $1.25-$5.00 | $0.3125-$1.25 | $1.00 |
| Gemini 1.5 Flash | $0.075-$0.30 | $0.01875-$0.075 | $0.025 |
| Gemini 2.0 Flash | $0.10 | $0.025 | $0.025 |

折扣为 75% off（即 0.25x），介于 OpenAI 的 50% off 和 Anthropic 的 90% off 之间。

注意 Google 有 **存储费用**——cache 占用存储空间需要按小时付费。这意味着对于低频使用的场景，长 TTL 可能反而不划算。

### 4.4 Google 独特的存储成本模型

假设缓存 50000 token 的文档，使用 Gemini 1.5 Pro：

```
TTL = 2 hours
存储成本 = 50000 / 1M × $1.00/hour × 2 hours = $0.10

每次使用节省 = 50000 / 1M × ($5.00 - $1.25) = $0.1875

盈亏平衡：$0.10 / $0.1875 ≈ 0.53 次
→ 在 2 小时内使用 1 次就能回本
```

但如果 TTL 设为 48 小时而只使用 2 次：

```
存储成本 = 50000 / 1M × $1.00/hour × 48 hours = $2.40
使用节省 = 2 × $0.1875 = $0.375
→ 亏损 $2.025
```

**教训：** TTL 不应该设置过长，应该根据实际使用频率来选择。

## 5. Prompt 结构设计最佳实践

### 5.1 核心原则：静态在前，动态在后

三大提供商都使用前缀匹配机制，因此 prompt 的组织顺序至关重要：

```
┌────────────────────────────────────┐
│ System Prompt（最静态）             │ ← 放最前面
│ - 角色定义                          │
│ - 行为准则                          │
│ - 输出格式要求                      │
├────────────────────────────────────┤
│ Tool Definitions（较静态）          │ ← 第二层
│ - function schemas                  │
│ - tool descriptions                 │
├────────────────────────────────────┤
│ Reference Documents（会话级静态）    │ ← 第三层
│ - 上传的文件内容                    │
│ - 知识库片段                        │
├────────────────────────────────────┤
│ 对话历史（增量变化）                │ ← 第四层
│ - 每次新增最后一轮                  │
├────────────────────────────────────┤
│ 当前用户消息（每次不同）            │ ← 放最后面
└────────────────────────────────────┘
```

### 5.2 避免"前缀污染"

以下做法会破坏前缀匹配，应该避免：

```python
# BAD: 时间戳在 system prompt 中
system_prompt = f"You are a helpful assistant. Current time: {datetime.now()}"
# → 每秒变化，cache 永远不会命中

# GOOD: 时间戳放在用户消息中
system_prompt = "You are a helpful assistant."
user_message = f"Current time is {datetime.now()}. User question: ..."

# BAD: 随机 session ID 在前缀中
system_prompt = f"Session: {uuid4()}\nYou are a helpful assistant."
# → 每次不同，cache 永远不会命中

# GOOD: session ID 放在后面或 metadata 中
system_prompt = "You are a helpful assistant."
user_message = f"[Session: {session_id}] {user_question}"
```

### 5.3 Anthropic 的 Breakpoint 策略

使用 Anthropic 时，`cache_control` breakpoint 的放置位置需要考虑：

```python
# 策略 1: 只在 system prompt 末尾放一个 breakpoint
# 适合：system prompt 稳定，tools 偶尔变化
system=[
    {"type": "text", "text": system_prompt, 
     "cache_control": {"type": "ephemeral"}}
]

# 策略 2: system prompt 和 tools 各放一个 breakpoint
# 适合：tools 也很稳定，或者 tools 定义很长
system=[
    {"type": "text", "text": system_prompt,
     "cache_control": {"type": "ephemeral"}}
],
tools=[
    {..., "cache_control": {"type": "ephemeral"}}
]

# 策略 3: 在对话历史中也放 breakpoint
# 适合：多轮对话，每轮只新增少量 token
messages=[
    {"role": "user", "content": [
        {"type": "text", "text": long_document,
         "cache_control": {"type": "ephemeral"}}
    ]},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": [
        {"type": "text", "text": follow_up_context,
         "cache_control": {"type": "ephemeral"}}
    ]},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "New question here"}  # 无 breakpoint
]
```

### 5.4 多轮对话的 Cache 复用策略

在多轮对话中，每一轮新消息都会使 prompt 变长，但前面的对话历史是不变的：

```
Turn 1: [System] [User_1]
Turn 2: [System] [User_1] [Asst_1] [User_2]
Turn 3: [System] [User_1] [Asst_1] [User_2] [Asst_2] [User_3]
```

**不使用 cache：** 每轮都要重新处理完整 prompt
**使用 cache：** Turn 3 可以复用 Turn 2 的前缀计算（如果 cache 未过期）

```
Turn 3 的 cache 行为:
  [System] [User_1] [Asst_1] [User_2] [Asst_2]  ← cache hit
  [User_3]                                        ← 新计算
```

Anthropic 建议在倒数第二轮消息末尾放置 breakpoint，以确保前面所有对话历史都被缓存：

```python
messages = build_conversation_history()

# 在倒数第二条消息上放 breakpoint
if len(messages) >= 2:
    second_to_last = messages[-2]
    if isinstance(second_to_last["content"], str):
        messages[-2] = {
            "role": second_to_last["role"],
            "content": [
                {
                    "type": "text",
                    "text": second_to_last["content"],
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        }
```

## 6. Cache Hit Rate 监控与分析

### 6.1 计算 Cache Hit Rate

```python
def calculate_cache_hit_rate(usage: dict) -> dict:
    """从 API 响应中计算 cache hit rate。"""
    
    # Anthropic 格式
    if "cache_read_input_tokens" in usage:
        cached = usage.get("cache_read_input_tokens", 0)
        created = usage.get("cache_creation_input_tokens", 0)
        uncached = usage.get("input_tokens", 0)
        total_input = cached + created + uncached
        
        return {
            "hit_rate": cached / total_input if total_input > 0 else 0,
            "cached_tokens": cached,
            "created_tokens": created,
            "uncached_tokens": uncached,
            "total_input_tokens": total_input,
        }
    
    # OpenAI 格式
    if "prompt_tokens_details" in usage:
        cached = usage["prompt_tokens_details"].get("cached_tokens", 0)
        total = usage["prompt_tokens"]
        
        return {
            "hit_rate": cached / total if total > 0 else 0,
            "cached_tokens": cached,
            "total_input_tokens": total,
        }
    
    return {"hit_rate": 0}
```

### 6.2 监控指标建议

在生产环境中应该监控以下指标：

```python
# 推荐的 Prometheus metrics
cache_hit_rate = Gauge(
    "llm_cache_hit_rate", 
    "Prompt cache hit rate",
    ["model", "endpoint"]
)

cache_savings_dollars = Counter(
    "llm_cache_savings_dollars_total",
    "Total dollars saved by prompt caching",
    ["model"]
)

cache_creation_tokens = Counter(
    "llm_cache_creation_tokens_total",
    "Total tokens written to cache",
    ["model"]
)

cache_read_tokens = Counter(
    "llm_cache_read_tokens_total",
    "Total tokens read from cache",
    ["model"]
)
```

### 6.3 分析 Cache Miss 的常见原因

当 cache hit rate 低于预期时，检查以下几点：

1. **前缀变化**：system prompt 中是否包含动态内容？
2. **Token 数不足**：前缀是否达到最小 token 数要求？
3. **TTL 过期**：请求间隔是否超过 5 分钟？
4. **路由不一致**：是否被路由到不同的服务器（不同 cache 分区）？
5. **Message 格式变化**：message 的结构或顺序是否一致？
6. **工具定义变化**：tool 列表或顺序是否一致？

## 7. 实际案例分析

### 7.1 案例：客服机器人

```
背景：
- System prompt: 3000 tokens（角色定义 + 知识库摘要）
- Tool definitions: 1500 tokens（10 个工具）
- 平均对话长度: 8 轮
- 日请求量: 100,000

不使用 cache:
  每次请求平均处理: 3000 + 1500 + 4轮历史平均 ≈ 6500 tokens
  日总 input: 100,000 × 6500 = 650M tokens
  日成本 (Claude Sonnet 4): 650 × $3 = $1,950

使用 cache:
  cache hit（90% 的请求）: 4500 tokens cached, 2000 tokens uncached
  cache miss（10%）: 4500 tokens write, 2000 tokens uncached
  
  日成本:
    cache write: 10,000 × 4500 / 1M × $3.75 = $168.75
    cache read: 90,000 × 4500 / 1M × $0.30 = $121.50
    uncached: 100,000 × 2000 / 1M × $3.00 = $600.00
    总计: $890.25
  
  节省: $1,950 - $890.25 = $1,059.75 / 天（54% 节省）
```

### 7.2 案例：代码审查助手

```
背景：
- System prompt: 1500 tokens
- 每次附带完整文件内容: 平均 5000 tokens
- 文件内容每次不同
- 日请求量: 10,000

分析：
  可缓存前缀: 1500 tokens（仅 system prompt）
  每次变化的部分: 5000 tokens（文件内容）
  
  cache 效果有限：
    命中 1500 tokens × $0.30/M × 9,000 = $4.05 节省
    写入 1500 tokens × $3.75/M × 1,000 = $5.625 成本
    
  净节省微乎其微——因为可缓存的前缀太短，占比太小

改进方案：
  将常用的代码规范文档（5000 tokens）加入 system prompt
  可缓存前缀: 6500 tokens
  → cache 效果显著提升
```

### 7.3 案例：RAG 系统中的 Cache 策略

RAG（Retrieval-Augmented Generation）系统中，每次检索的文档片段通常不同，这给 caching 带来挑战。

```
传统 RAG prompt:
  [System Prompt] [Retrieved Doc 1] [Retrieved Doc 2] [User Query]
  → retrieved docs 每次不同，只有 system prompt 可以缓存

优化方案 1: 热门文档预缓存
  - 统计高频检索的文档 Top-K
  - 将 Top-K 文档内容固定放入 prompt 前缀
  [System] [Top-K Docs (fixed)] [Additional Docs (dynamic)] [Query]
  → Top-K 文档可以被缓存

优化方案 2: 两阶段架构
  - 第一阶段：用小模型 + 检索文档生成摘要（不使用 cache）
  - 第二阶段：用大模型 + 固定 system prompt + 摘要回答（使用 cache）
```

## 8. 跨提供商迁移注意事项

### 8.1 从 OpenAI 迁移到 Anthropic

```python
# OpenAI（自动，无需修改）
response = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[...]
)

# Anthropic（需要添加 cache_control）
response = anthropic_client.messages.create(
    model="claude-sonnet-4-20250514",
    system=[{
        "type": "text",
        "text": system_prompt,
        "cache_control": {"type": "ephemeral"}  # 需要手动添加
    }],
    messages=[...]
)
```

### 8.2 从 Anthropic 迁移到 Google

```python
# Anthropic（ephemeral cache）
# → Google 隐式缓存：不需要任何标记

# → Google 显式缓存：需要预先创建 cache 对象
cache = genai.caching.CachedContent.create(
    model="models/gemini-2.0-flash",
    contents=[{"role": "user", "parts": [system_content]}],
    ttl=datetime.timedelta(hours=1),
)
```

### 8.3 费用对比计算器

假设 10000 个请求，前缀 5000 tokens，cache hit rate 90%：

```
OpenAI (GPT-4o):
  Normal:  5000 × 10000 / 1M × $2.50 = $125.00
  Cached:  5000 × 9000 / 1M × $1.25 + 5000 × 1000 / 1M × $2.50
         = $56.25 + $12.50 = $68.75
  节省: $56.25 (45%)

Anthropic (Claude Sonnet 4):
  Normal:  5000 × 10000 / 1M × $3.00 = $150.00
  Cached:  5000 × 9000 / 1M × $0.30 + 5000 × 1000 / 1M × $3.75
         = $13.50 + $18.75 = $32.25
  节省: $117.75 (78.5%)

Google (Gemini 2.0 Flash, 隐式):
  Normal:  5000 × 10000 / 1M × $0.10 = $5.00
  Cached:  5000 × 9000 / 1M × $0.025 + 5000 × 1000 / 1M × $0.10
         = $1.125 + $0.50 = $1.625
  节省: $3.375 (67.5%)
```

**结论：** Anthropic 在高 cache hit rate 场景下节省比例最高（90% off 的读取折扣），但 Google Gemini 的绝对成本最低。OpenAI 的优势在于零配置。

---

**下一节：** [动手练习](exercises.md) — 通过实际操作测量 cache hit rate，体验 Prompt Caching 的效果。
