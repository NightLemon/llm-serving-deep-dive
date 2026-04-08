# 术语表

| 术语 | 英文全称 | 简要说明 |
|------|---------|---------|
| KV Cache | Key-Value Cache | 存储已计算的 attention K、V 向量，避免重复计算 |
| Prefill | Prefill Phase | 推理的第一阶段，处理整个 prompt 生成 KV Cache |
| Decode | Decode Phase | 推理的第二阶段，逐 token 自回归生成 |
| APC | Automatic Prefix Caching | vLLM 的自动前缀缓存功能 |
| MHA | Multi-Head Attention | 标准多头注意力 |
| MQA | Multi-Query Attention | 多查询注意力，所有 head 共享一组 KV |
| GQA | Grouped Query Attention | 分组查询注意力，多个 head 共享一组 KV |
| MLA | Multi-head Latent Attention | DeepSeek-V2 提出的 latent attention，KV 压缩到低维空间 |
| TTFT | Time To First Token | 首 token 延迟 |
| TBT | Time Between Tokens | token 间延迟 |
| ITL | Inter-Token Latency | 同 TBT |
| TP | Tensor Parallel | 张量并行 |
| PP | Pipeline Parallel | 流水线并行 |
| DP | Data Parallel | 数据并行 |
| EP | Expert Parallel | 专家并行（MoE 模型） |
| CP | Context Parallel | 上下文并行（序列维度切分） |
| MoE | Mixture of Experts | 混合专家模型 |
| SLA | Service Level Agreement | 服务等级协议 |
| TCO | Total Cost of Ownership | 总拥有成本 |
| HBM | High Bandwidth Memory | GPU 高带宽显存 |
| TTL | Time To Live | 缓存过期时间 |
| LRU | Least Recently Used | 最近最少使用（驱逐策略） |
| ARC | Adaptive Replacement Cache | 自适应替换缓存（驱逐策略） |
| DBO | Dual Batch Overlap | 双批次重叠执行 |
