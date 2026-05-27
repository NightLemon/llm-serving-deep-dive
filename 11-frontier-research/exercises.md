# 前沿研究 — 练习

## 练习 1：论文精读与总结

从 [05-paper-list.md](05-paper-list.md) 中选择 **2 篇标注为 ⭐ 必读** 的论文，完成以下任务：

1. 用一段话（100-200 字）总结论文的核心贡献
2. 画出论文的系统架构图（可用 Mermaid 或手绘）
3. 列出该技术的 3 个适用场景和 3 个不适用场景
4. 该论文的方法与本仓库对应章节的内容有何关联？

## 练习 2：技术趋势分析

阅读 [06-trends.md](06-trends.md) 后回答：

1. 从成本、延迟、吞吐量三个维度，分析 2023-2026 年 LLM 推理技术的演进方向
2. 选择一个你认为最有前景的短期趋势（2026-2027），说明理由
3. 如果你要设计一个面向 2027 年的推理服务架构，会采用哪些本章提到的技术？

## 练习 3：Hybrid KV Cache 设计

阅读 [01-hybrid-kv-cache.md](01-hybrid-kv-cache.md) 后：

1. 解释为什么 Transformer + Mamba 混合架构需要 Hybrid KV Cache Manager
2. 在 vLLM 的 `KVCacheCoordinator` 中，不同 layer group 的 cache 策略是如何配置的？
3. 设计一个场景：3 种不同类型的 attention layer 混合使用时，如何分配 KV Cache 预算？

## 练习 4：编译优化实验

阅读 [03-compilation.md](03-compilation.md) 后：

1. 对比 `torch.compile` 开启前后的 vLLM 推理延迟（如有 GPU 环境）
2. 解释 CUDA Graph 如何减少 kernel launch overhead
3. TensorRT-LLM 的静态图优化与 vLLM 的动态 JIT 编译，各适合什么场景？

---

## 练习 5：vLLM Hybrid KV Cache 源码走读

**目标**：阅读 vLLM 中 hybrid KV cache 的真实实现，把 [01-hybrid-kv-cache.md](01-hybrid-kv-cache.md) 里的概念落到代码上。

### 推荐源码位置

以 vLLM main 分支（commit 至少在 2025 年 Q1 之后）为准，重点读：

- `vllm/v1/core/kv_cache_manager.py` — 顶层管理器，知道每个 layer 走哪个子 manager
- `vllm/v1/core/kv_cache_coordinator.py` — 协调多种 layer group（full attention / sliding window / Mamba state 等）
- `vllm/v1/core/specialized_managers.py` — 每种类型的具体实现
- `vllm/config.py` 中的 `KVCacheConfig` / `KVCacheSpec` —— 看不同 layer group 如何在配置层声明

> 若 vLLM 重命名或重构了文件，按 grep 关键字 `KVCacheCoordinator` 或 `HybridKVCacheManager` 在最新代码里定位。

### 任务

1. 画一张时序图：一个 request 从进入 scheduler 到 KV cache 真正被分配，途经哪几个对象、调用哪些方法
2. 找到 "每个 layer group 独立维护 block table" 的具体代码位置，用 5 行以内引文 + 一句话解释回答
3. 解释：当一个 hybrid 模型有 2 个 full attention layer + 28 个 sliding window layer 时，`gpu_memory_utilization` 是如何被切分到不同 group 的？看看代码是按比例分还是按需扩张
4. 设想一个 bug：如果某 group 的 `block_size` 写错了，会在哪一步崩溃？给出你认为最可能的报错栈位置

### 验收

能在白板上 5 分钟讲清楚 `KVCacheCoordinator` 与 `KVCacheManager` 的职责边界，且能指着代码说"这一行就是 hybrid 设计的核心抽象"。

---

## 练习 6：论文精读报告

**目标**：把"读论文"变成产出，避免读完就忘。

### 任务

从 [05-paper-list.md](05-paper-list.md) 中选择 **1 篇 ⭐ 必读论文**（推荐：DistServe、SGLang、Mooncake、Hydragen、Sarathi-Serve 任选其一），写一份 **2 页（约 1500 字）的技术报告**。

报告必须包含以下小节：

1. **TL;DR（150 字以内）**：用一段话回答"它解决了什么问题、用了什么手段、收益是多少"
2. **动机（200 字）**：作者为什么觉得现有方案不够？给出至少一个具体的数字 / 场景
3. **核心方法（400 字 + 1 张图）**：用自己的话复述算法或架构，画一张图（Mermaid 即可）
4. **关键实验（200 字）**：选 1-2 个最重要的实验，列出 baseline / 提升幅度 / 评估指标
5. **局限性（200 字）**：作者自己承认了什么？你额外看到的局限？
6. **与本仓库的关联（150 字）**：这篇论文对应本仓库哪一章？如果用在生产中，会和我们学过的哪些技术冲突或互补？

### 写作要求

- 不允许直接复制论文摘要的句子
- 数字必须有出处（论文的表/图编号）
- 图必须是你重画的，不是截图

### 验收

写完后回头读一遍，问自己：**如果一个面试官问"你最近读过哪篇 LLM serving 论文"，我能否用这份报告里的内容讲 5 分钟而不卡壳？** 如果不能，回去补。
