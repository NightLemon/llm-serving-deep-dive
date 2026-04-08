# Ch02: 前缀缓存与 Prompt Caching

> 前置知识：Ch01 KV Cache 深度剖析

## 🎯 学习目标

- 理解 Prefix Caching 的核心原理：为什么相同的 prompt 前缀可以共享 KV Cache？
- 掌握 vLLM APC（Automatic Prefix Caching）的实现机制：hash 匹配、block 粒度
- 理解 SGLang RadixAttention 的 Radix Tree 方案及其与 APC 的差异
- 了解 API 提供商（OpenAI / Anthropic / Google）的 Prompt Caching 如何工作
- 能够设计 prompt 结构以最大化 cache hit rate

## 📑 内容大纲

### 1. 前缀缓存原理（01-principles.md）

**核心概念：**
- 为什么前缀可以共享？—— Causal Attention 的因果性保证
- Cache 粒度：token 级 vs block 级 vs 请求级
- Hash 匹配机制：如何快速判断两个前缀是否相同？
- Cache 命中的条件：完全前缀匹配（exact prefix match）
- TTL（Time-To-Live）与驱逐策略：LRU、LFU、ARC

**关键洞察：**
- Cached token 的成本结构（以 Anthropic 为例）：
  - Cache write = 1.25x 基础价格
  - Cache read = 0.1x 基础价格（打一折）
  - 意味着同一前缀被读 2 次以上就赚回来了
- OpenAI 的方案：完全自动、无额外费用、基于路由的 cache 热度维护
- Google Gemini：隐式缓存（自动）+ 显式缓存（手动 TTL 控制）

### 2. vLLM APC 源码分析（02-vllm-apc.md）

**源码走读：**
- `vllm/v1/core/kv_cache_utils.py` — hash 计算与 block 匹配逻辑
- `vllm/v1/core/kv_cache_manager.py` — cache hit 判定与 block 复用
- `vllm/v1/core/block_pool.py` — 物理 block 的引用计数与回收
- hash 是如何计算的？—— content hash vs position hash
- Lookback 窗口：为什么 vLLM 限制 20 block 的回溯？
- `--enable-prefix-caching` 参数的代码路径

**关键数据结构：**
- `PrefixHash`：block 内容的唯一标识
- `BlockPool`：物理 block 管理器，支持引用计数和 copy-on-write

### 3. SGLang RadixAttention（03-radix-attention.md）

**论文解读：**
- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)
- Radix Tree 数据结构在 KV Cache 管理中的应用
- 与 vLLM APC 的对比：树形结构 vs 哈希表
- 优势：支持任意共享前缀的子树复用，不仅仅是完整前缀

**源码走读：**
- SGLang RadixCache 的核心实现
- insert / match / evict 操作

### 4. API 提供商的 Prompt Caching 实践（04-api-caching.md）

**各家方案对比：**

| 特性 | OpenAI | Anthropic | Google Gemini |
|------|--------|-----------|---------------|
| 触发方式 | 自动 | 自动/手动 breakpoint | 隐式/显式 |
| 最小 token 数 | 1024 | 1024-4096（按模型） | 1024-4096 |
| Cache 寿命 | 5-10min（内存）/ 24h（扩展） | 5min / 1h | 按 TTL |
| 额外写入费用 | 无 | 1.25x | 无 |
| 读取折扣 | 50% off | 90% off | 有但不保证 |
| 跨请求共享 | 同 org | 同 workspace | 同项目 |

**工程实践：**
- 如何设计 prompt 结构以最大化 cache hit？
  - 静态内容（system prompt、tools）放最前面
  - 动态内容（用户消息）放最后
  - 避免在前缀中包含时间戳等变化内容
- Cache hit rate 的监控与分析
- `cache_read_input_tokens` / `cache_creation_input_tokens` 字段解读

## 📄 参考论文

| 论文 | 年份 | 核心贡献 |
|------|------|----------|
| [SGLang: Efficient Execution of Structured LM Programs](https://arxiv.org/abs/2312.07104) | 2023 | RadixAttention |
| [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) | 2023 | Prefix sharing via paging |
| [ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache](https://arxiv.org/abs/2402.15220) | 2024 | Chunk-level prefix caching |

## 📁 文件清单

- [x] `01-principles.md` — 前缀缓存原理
- [x] `02-vllm-apc.md` — vLLM APC 源码分析
- [x] `03-radix-attention.md` — SGLang RadixAttention
- [x] `04-api-caching.md` — API 提供商 Prompt Caching 实践
- [x] `exercises.md` — 动手练习（测量 cache hit rate）
