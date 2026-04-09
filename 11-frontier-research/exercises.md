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
