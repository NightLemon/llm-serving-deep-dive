# 生产环境模式

> 从"能生成结构化输出"到"在生产环境中稳定、高效地生成结构化输出"，还需要解决一系列工程问题：模式选择、流式输出、性能优化、监控告警。

## 1. JSON Mode vs Structured Output

生产环境中，"结构化输出"有两种常见的保证级别，理解它们的区别至关重要。

### 1.1 JSON Mode（松保证）

JSON mode 只保证输出是**语法上合法的 JSON**，但不保证符合任何特定 schema。

```python
# OpenAI API 示例
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    response_format={"type": "json_object"}  # JSON mode
)

# 输出一定是合法 JSON，但字段名、类型、结构不可控
# 可能输出 {"result": "Alice is 25"} 而不是 {"name": "Alice", "age": 25}
```

**实现方式**：通常是通过一个通用的 JSON grammar 做 constrained decoding——确保括号匹配、引号配对、逗号正确，但不约束具体的 key 和 value 类型。

**适用场景**：快速原型、对输出结构有一定容忍度的场景。

### 1.2 Structured Output（强保证）

Structured output 保证输出**严格匹配提供的 JSON Schema**——字段名、字段类型、必填/可选、枚举值都被精确约束。

```python
# OpenAI API 示例
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "user_info",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer", "minimum": 0}
                },
                "required": ["name", "age"],
                "additionalProperties": False
            }
        }
    }
)
```

**实现方式**：将具体的 JSON Schema 编译为 DFA/PDA，在每步 decode 时做精确的 logit masking。

**适用场景**：需要直接将输出传入下游系统（数据库、API、工作流）的生产场景。

### 1.3 选择建议

| 维度 | JSON Mode | Structured Output |
|------|-----------|-------------------|
| Schema 合规率 | ~70-90%（取决于 prompt 质量） | 100% |
| 延迟开销 | 低（通用 JSON grammar） | 低-中（取决于 schema 复杂度） |
| 开发成本 | 低（无需定义 schema） | 中（需要定义精确 schema） |
| 适用阶段 | 原型/探索 | 生产/集成 |

## 2. 常见生产模式

### 2.1 Function Calling 作为结构化输出

Function calling（工具调用）本质上就是结构化输出的一种形式——模型输出一个函数名和参数列表，参数必须符合函数签名。

```json
{
  "function": "search_database",
  "arguments": {
    "query": "recent orders",
    "limit": 10,
    "sort_by": "date"
  }
}
```

在 serving 层面，function calling 的实现与 structured output 完全一致——将函数签名编译为 schema，使用 constrained decoding 保证输出格式。一些框架（如 vLLM）在内部统一处理 function calling 和 structured output。

### 2.2 Streaming Structured Output

流式输出（streaming）是生产环境中降低首 token 延迟（TTFT）的标准做法。但结构化输出增加了复杂性——客户端收到的是**不完整的 JSON fragment**。

```
Stream chunk 1: {"name": "Ali
Stream chunk 2: ce", "age":
Stream chunk 3:  25, "skills
Stream chunk 4: ": ["Python"
Stream chunk 5: , "Go"]}
```

**挑战**：

- 中间状态的 JSON 是不完整的，无法直接 parse
- 客户端需要增量解析（incremental parsing）能力
- 网络中断时需要处理不完整输出

**解决方案**：

1. **Partial JSON parsing**：使用容忍不完整 JSON 的解析器（如 `partial-json-parser`）
2. **Event-based streaming**：每当一个完整的 field 被生成，emit 一个事件
3. **Server-Sent Events (SSE)**：标准的流式传输协议，每个 SSE 事件包含一个 JSON fragment

### 2.3 Error Recovery：模型"想"违反 Schema 时

当 constrained decoding 强制模型输出不符合其概率分布的 token 时，可能出现质量退化——模型被迫走上一条"不自然"的生成路径。

**常见表现**：

- 字段值语义不合理（如 `"age": 999`）
- 文本字段中出现重复或乱码
- 数组长度异常（极短或极长）

**根本原因**：模型内部的分布与 schema 约束冲突严重。例如，模型"想"输出一段解释文本，但 schema 要求此处是一个 integer。

**缓解策略**：

1. **Schema 设计**：让 schema 尽可能贴合模型的自然输出倾向——例如增加一个 `reasoning` 字段让模型先"思考"
2. **Prompt 工程**：在 prompt 中明确说明每个字段的含义和期望值
3. **模型选择**：使用经过结构化输出训练的模型（如 GPT-4o、Llama-3-Instruct），它们对 JSON 格式更"熟悉"

## 3. 性能优化

### 3.1 Schema 复杂度与延迟

Schema 越复杂，DFA 状态数越多，Index 表越大，编译和运行时开销越高。

```
简单 schema（5 字段，扁平）  → ~50 DFA 状态  → 编译 <50ms
中等 schema（20 字段，嵌套） → ~500 DFA 状态 → 编译 ~200ms
复杂 schema（递归 + 数组）   → ~5000 DFA 状态 → 编译 ~2s
```

**优化原则**：**保持 schema 尽可能简单。** 如果业务逻辑需要复杂的嵌套结构，考虑拆分为多次调用，每次使用简单 schema。

### 3.2 DFA/FSM 缓存

同一个 schema 的 DFA 和 Index 表只需编译一次。生产环境中应实现 schema 级缓存：

```python
# 伪代码：schema 级 DFA 缓存
class DFACache:
    def __init__(self, max_size=1000):
        self.cache = LRUCache(max_size)

    def get_or_compile(self, schema: dict) -> DFAIndex:
        key = hash_schema(schema)
        if key not in self.cache:
            dfa = compile_schema_to_dfa(schema)
            index = precompute_token_index(dfa, tokenizer)
            self.cache[key] = index
        return self.cache[key]
```

**预热策略**：在服务启动时，预编译已知的常用 schema（从历史请求日志中统计 top-N schema）。

### 3.3 同 Schema 请求批量优化

当 batch 中多个请求使用同一个 schema 时，可以共享 Index 表查询结果：

- 虽然每个请求的 DFA 状态不同，但 Index 表是同一个
- 可以减少内存占用（一份 Index 表被多个请求引用）
- 在极端情况下（所有请求使用同一 schema），logit masking 的内存开销降至最低

## 4. 监控指标

### 4.1 Schema 合规率

$$
\text{Schema Compliance Rate} = \frac{\text{符合 schema 的输出数}}{\text{总输出数}} \times 100\%
$$

使用 constrained decoding 时，理论上这个值应该是 **100%**。如果低于 100%，说明实现存在 bug 或使用了非严格模式（JSON mode）。

### 4.2 Forced-Token 比率

$$
\text{Forced-Token Ratio} = \frac{\text{只有 1 个合法 token 的 decode steps}}{\text{总 decode steps}}
$$

这个指标反映模型被"强制"输出特定 token 的频率：

- **过高**（>50%）：schema 过于严格，模型几乎没有自由度，输出质量可能受损
- **适中**（20-40%）：正常范围，大部分 forced token 是 JSON 结构字符（`{`, `}`, `:`, `,`, `"`）
- **过低**（<10%）：schema 非常宽松，约束效果有限

### 4.3 延迟影响

监控 constrained decoding 带来的额外延迟：

```
constrained_overhead_ms = constrained_latency - unconstrained_latency
constrained_overhead_pct = constrained_overhead_ms / unconstrained_latency * 100
```

建议设置告警阈值：

| 指标 | 正常范围 | 告警阈值 |
|------|---------|---------|
| 每 token 额外延迟 | <0.5ms | >2ms |
| 吞吐量下降百分比 | <10% | >20% |
| Schema 编译耗时 | <200ms | >1s |

### 4.4 与推理监控的集成

> 参考 [Ch10: 生产环境 — 监控](../10-production-patterns/02-monitoring.md) 了解推理服务监控的完整框架。

将 constrained decoding 指标集成到现有的监控体系中：

- **Prometheus metrics**：`constrained_decoding_compile_seconds`、`constrained_decoding_forced_token_ratio`
- **Dashboard**：在推理延迟面板中增加 constrained vs unconstrained 对比
- **日志**：记录每个请求的 schema hash、DFA 状态数、forced-token 比率

## 5. 与 llm-engineering-fundamentals 的关系

> 参考 [llm-engineering-fundamentals Ch06](https://github.com/NightLemon/llm-engineering-fundamentals) 中关于应用层结构化输出的讨论。

本章聚焦 **serving 层面** 的结构化输出实现（constrained decoding 的原理与 serving pipeline 交互），而 llm-engineering-fundamentals 的相关章节聚焦 **应用层面**（如何设计 schema、如何在应用代码中使用结构化输出 API、错误处理策略）。两者是互补的：

| 维度 | 本仓库（Serving 深度剖析） | llm-engineering-fundamentals |
|------|------------------------|------------------------------|
| 视角 | 推理引擎内部实现 | 应用开发者视角 |
| 关注点 | DFA/PDA、logit masking、吞吐量 | API 使用、schema 设计、错误处理 |
| 目标读者 | 推理系统工程师 | 应用开发工程师 |
