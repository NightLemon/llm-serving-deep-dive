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

---

## 练习 5：OpenAI Structured Output API vs Outlines 对照实验（🟢 纯本地 + API）

**目标**：把 [02-serving-interaction.md](02-serving-interaction.md) 讲的"服务端 constrained decoding"放到真实环境对比。

### 准备

```bash
pip install openai outlines
export OPENAI_API_KEY="..."
```

任选一个 7B 以下的本地可加载模型，或者直接用 OpenAI `gpt-4o-mini` 作为对比基线。

### 任务

对**同一个非平凡 schema**（建议：嵌套 3 层、至少 1 个 enum、1 个 array），用以下三种方式跑 100 个测试样本：

1. **无约束**：普通 chat completion + 提示词要求 JSON 输出
2. **OpenAI Structured Output**：用 `response_format={"type": "json_schema", "json_schema": ...}`
3. **Outlines + 本地小模型**：用 `outlines.generate.json(model, schema)` 跑本地推理

### 测量

| 维度 | 无约束 | OpenAI SO | Outlines |
|------|--------|-----------|----------|
| 100% 合法 JSON 的比例 | ? | ? | ? |
| 100% 符合 schema 的比例 | ? | ? | ? |
| 平均输出 token 数 | ? | ? | ? |
| 平均 e2e 延迟 | ? | ? | ? |
| 单次成本（美元） | ? | ? | ? |

### 验收

回答这 3 个问题：
1. 三种方式里，**合法率**和**延迟**是否存在反比关系？反直觉吗？
2. OpenAI Structured Output 内部用的是什么实现？（去查文档/blog——它和 Outlines 的 logits mask 是同一类技术吗？）
3. 如果你在生产里要在"调 GPT-4o + Structured Output" 和 "自部署 Llama + Outlines" 间选，决策点是什么？

---

## 练习 6：vLLM Guided Decoding 实测（🟡 L4 单卡）

**目标**：在真实推理引擎里观察 constrained decoding 对吞吐的影响，验证 [03-production-patterns.md](03-production-patterns.md) 里的取舍说法。

### 环境

- 1 张 L4 / L40S / A100，24GB+
- vLLM ≥ 0.6.x
- 模型：Qwen2.5-7B-Instruct

### 实验

启动 vLLM：

```bash
# Baseline: 无 guided decoding
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --port 8000

# 对照: 启用 xgrammar backend（也可改为 outlines / guidance / lm-format-enforcer）
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --max-model-len 4096 \
  --structured-outputs-config.backend xgrammar \
  --port 8001
```

> 旧版 vLLM 文档中常见 `--guided-decoding-backend`；新版服务端参数已迁移到 `--structured-outputs-config.backend`。如果你固定使用旧版本 vLLM，请以对应版本的 CLI 帮助为准。

写一个 benchmark 脚本，向两个端口同时发 200 个并发请求（同一个 schema），测：

1. **吞吐量**：tokens/s
2. **TTFT** p50/p99
3. **首次合法 JSON 的成功率**（baseline 用 prompt 引导，对照走 `response_format`）
4. **FSM 编译延迟**：第一次发某 schema 的请求 vs 后续命中 cache 的请求，TTFT 差多少？

### 验收

- 至少试 2 个不同 backend（outlines / xgrammar / lm-format-enforcer），找出哪个延迟最低
- 画出"baseline 吞吐 vs guided 吞吐"的对比柱状图
- 用一段话回答："如果我的服务 90% 请求需要 JSON 输出，应该全局开 guided decoding 吗？" 给出你的判断和理由

---

## 练习 7：Schema Cache 容量与灰度方案（📖 设计题）

**目标**：把 [04-capacity-and-runbook.md](04-capacity-and-runbook.md) 里的上线 checklist 变成一份可执行设计。

### 场景

你负责一个信息抽取服务：

- 日均 300 万请求，峰值 200 QPS
- 共有 800 个注册 schema，其中 Top 50 覆盖 85% 流量
- 平均 schema 状态数 180，P95 状态数 900
- tokenizer 词表 128K
- 目标：structured output 启用后，TTFT P99 增量 < 150 ms

### 任务

1. 估算 Top 50 schema 和全部 800 schema 的 mask bitset 内存上界。
2. 设计 schema cache：缓存上限、驱逐策略、启动预热策略。
3. 设计灰度流程：从 shadow mode 到 100% 流量需要看哪些指标？
4. 写出 fallback 策略：schema 编译失败、backend 超时、cache 内存打满时分别怎么处理？

### 验收

交付一页 runbook，至少包含：容量估算表、指标阈值、fallback 决策树。
