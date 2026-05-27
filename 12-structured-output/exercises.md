# 结构化输出与 Constrained Decoding — 练习

## 练习 1：FSM 状态转换模拟

给定如下 JSON Schema：

```json
{
  "type": "object",
  "properties": {
    "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
    "score": {"type": "number"}
  },
  "required": ["sentiment", "score"]
}
```

1. 手动画出该 schema 对应的 DFA 状态转换图（可用 Mermaid）
2. 在解码到 `{"sentiment": "` 这个位置时，列出所有合法的 next token（假设词表包含 `p`, `po`, `positive`, `n`, `ne`, `negative`, `neutral`, `"`, `123` 等）
3. 如果模型想输出 `"happy"`，constrained decoding 会如何处理？

## 练习 2：Speculative Decoding 兼容性分析

阅读 [02-serving-interaction.md](02-serving-interaction.md) 第 3 节后：

1. 假设 draft model 的 draft length $\gamma = 5$，在 JSON key 区域（如 `"sentiment":` ）的期望接受长度是多少？为什么？
2. 设计一个实验方案：对比"有 constrained decoding"和"无 constrained decoding"场景下 speculative decoding 的接受率变化
3. 在什么类型的 schema 下，speculative decoding + constrained decoding 的组合效果最差？最好？

## 练习 3：生产环境部署

假设你负责一个在线信息提取服务，使用 vLLM 部署 Llama-3-8B-Instruct，日均处理 100 万次请求，涉及 50 种不同的 JSON Schema。

1. 设计 schema 缓存策略：缓存多少个 DFA？如何决定驱逐策略？
2. 计算 DFA Index 表的总内存占用（假设平均每个 schema 有 200 个 DFA 状态，词表大小 128K）
3. 设计监控 dashboard，列出你会跟踪的 5 个关键指标及其告警阈值

## 练习 4：Jump-Forward 优化收益估算

给定如下 schema 的一次典型输出：

```json
{"name": "Alice", "age": 25, "city": "Beijing", "active": true}
```

1. 数出总共有多少个 token 是完全被 schema 决定的（forced token）
2. 估算 jump-forward 优化能减少多少 decode steps
3. 如果单次 decode step 耗时 10ms，jump-forward 能节省多少延迟？
