# 版本基线与 Freshness Gate

> 本页记录本仓库的“学习基线”和“学习前复核流程”。LLM serving 生态变化很快，尤其是 vLLM/SGLang/LMCache、API provider 的模型名、CLI 参数和计费规则。学习时先跑完本页的 freshness gate，再进入具体章节。

## 1. 当前基线

更新时间：2026-05-29

| 项目 | 当前检查结果 | 用途 | 官方入口 |
|------|-------------|------|----------|
| vLLM | latest release: `v0.21.0`，发布于 2026-05-15 | 本仓库主要 serving engine 基线 | https://github.com/vllm-project/vllm / https://docs.vllm.ai/en/latest/ |
| SGLang | latest release: `v0.5.12.post1`，发布于 2026-05-26 | RadixAttention、structured output、serving 对照 | https://github.com/sgl-project/sglang / https://docs.sglang.ai/ |
| LMCache | core stable release: `v0.4.5`，发布于 2026-05-15；repo 同时有 nightly 与 operator release | KV cache offloading / sharing | https://github.com/LMCache/LMCache |
| OpenAI API | 不在本文档中固定“最新模型名” | structured output、prompt caching、vision API 实验 | https://platform.openai.com/docs |
| Anthropic API | 不在本文档中固定“最新模型名” | prompt caching、vision API 实验 | https://docs.anthropic.com/ |

为什么 API provider 不固定“最新模型名”：模型名、定价和支持能力变化频繁。教材里的 `gpt-*`、`claude-*` 示例只代表代码形状，真正实验前必须看官方 docs 和 pricing 页面。

## 2. Freshness Gate

开始学习或跑实验前，先做下面 5 件事。

### 2.1 检查框架版本

```bash
gh release view -R vllm-project/vllm --json tagName,publishedAt,url
gh release view -R sgl-project/sglang --json tagName,publishedAt,url
gh release list -R LMCache/LMCache --limit 10
```

如果你本地安装的版本落后最新 release 很多，先不要直接照着章节命令跑，尤其是这些高频变化点：

- vLLM `vllm serve` CLI 参数
- structured output / guided decoding backend 配置
- context parallel / disaggregated serving / KV transfer 配置
- Prometheus metric 名称

### 2.2 检查 vLLM CLI 参数

```bash
vllm serve --help | rg "structured|guided|context|kv-transfer|speculative|chunked|prefix"
```

如果帮助输出和章节中的参数不一致，以 `vllm serve --help` 和 vLLM latest docs 为准，并在本仓库开 issue 或直接更新文档。

### 2.3 检查 API provider 能力

跑 OpenAI / Anthropic / Google 等 API 实验前，至少确认：

- 当前推荐模型名
- structured output / JSON schema 支持方式
- prompt caching 或 cached tokens 的计费规则
- vision input 的 token/价格规则
- Batch API 或异步任务折扣是否变化

不要把章节里的模型名当成“最新推荐”。模型名只是示例，能力和价格以官方文档为准。

### 2.4 固定实验环境

每次 benchmark 都记录：

```text
date:
GPU:
driver:
CUDA:
python:
vllm:
sglang:
lmcache:
model:
model revision / commit:
command:
```

没有这些信息，吞吐量、TTFT、TBT、显存占用都很难复现。

### 2.5 区分“概念稳定”和“实现易变”

| 内容 | 稳定性 | 学习策略 |
|------|--------|----------|
| KV cache 公式、prefill/decode 区分、PagedAttention 思想 | 高 | 可以直接学 |
| vLLM/SGLang 文件路径、类名、CLI 参数 | 中低 | 学概念，跑实验前复核 |
| API 模型名、定价、prompt caching 规则 | 低 | 每次实验前查官方 docs |
| benchmark 数字 | 低 | 只看量级和趋势，自己复测 |

## 3. 章节学习建议

- Ch01 / Ch03 / Ch04 的数学和内存模型相对稳定，但源码路径可能变。
- Ch05 / Ch06 的 disaggregated serving、KV transfer、LMCache 集成变化较快，必须看版本基线。
- Ch07 speculative decoding 的算法稳定，但 vLLM 支持的 speculative backend 和 metrics 可能变。
- Ch08 scheduler 概念稳定，但 `scheduler.py` 的路径和实现细节会随 vLLM 重构变化。
- Ch12 structured output 的概念稳定，但 backend 参数、OpenAI-compatible API 字段、provider 支持能力变化很快。
- Ch13 multimodal serving 的容量模型稳定，但具体 VLM 支持列表、image processor、vision token 规则变化很快。

## 4. 文档维护规则

新增或修改章节时，遵守这几条：

1. 涉及框架命令时，优先写“验证方法 + 当前示例”，不要只写死一个参数。
2. 涉及 API provider 时，注明“模型名和价格以官方 docs 为准”。
3. 涉及 benchmark 时，必须写硬件、版本、模型和命令。
4. 涉及源码走读时，标出框架版本或 commit；如果只读 latest，明确写“latest docs / main branch”。
5. 每次大改后跑 `mkdocs build --strict`，避免 stale nav 和坏链接。

这页不是装饰品。它是学习这个 repo 的第一道门槛：先确认你正在学的是当前生态里的实现，再进入具体章节。
