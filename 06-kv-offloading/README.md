# Ch06: KV Cache 卸载

> 前置知识：Ch01 KV Cache 深度剖析、Ch04 PagedAttention

## 🎯 学习目标

- 理解 GPU → CPU → SSD 分层存储策略在 KV Cache 管理中的应用
- 了解 OpenAI Extended Prompt Caching 的工作原理（GPU-local storage 持久化 KV tensors）
- 走读 vLLM KV Offloading 相关源码
- 掌握 LMCache 的集成方案
- 理解 offloading 的延迟-成本权衡

## 📑 内容大纲

### 1. 分层存储原理（01-tiered-storage.md）

**内存层级与特征：**
| 层级 | 容量 | 带宽 | 延迟 | 成本 |
|------|------|------|------|------|
| GPU HBM | 40-80 GB | ~3 TB/s | ~ns | $$$ |
| CPU DRAM | 256-2048 GB | ~100 GB/s | ~100ns | $$ |
| NVMe SSD | 1-8 TB | ~7 GB/s | ~10μs | $ |

**关键洞察：**
- GPU HBM 是最稀缺的资源，KV Cache 是 HBM 的最大消费者
- CPU DRAM 容量是 GPU HBM 的 10-50 倍，成本更低
- 将 "冷" KV Cache offload 到 CPU，为 "热" 请求腾出 GPU 空间
- 异步传输：利用 CUDA stream overlap 隐藏传输延迟

**与 OS 虚拟内存的类比：**
- GPU HBM ≈ 物理内存
- CPU DRAM ≈ Swap 分区
- SSD ≈ 磁盘
- eviction policy ≈ 页面替换算法（LRU、ARC）

### 2. vLLM KV Offloading 源码分析（02-vllm-offloading.md）

**核心源码：**
- `vllm/v1/kv_offload/` — offloading 框架
  - `abstract.py`：抽象接口定义
  - `cpu/manager.py`：CPU offload 管理器
  - `cpu/policies/` — 驱逐策略（LRU、ARC）
  - `worker/cpu_gpu.py` — CPU↔GPU 数据传输 worker

- `vllm/v1/simple_kv_offload/` — 简化版 offloading
  - `manager.py`：统一管理接口
  - `cuda_mem_ops.py`：CUDA 内存操作封装

- `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py` — 连接器

**驱逐策略：**
- **LRU (Least Recently Used)**：最简单有效的策略
- **ARC (Adaptive Replacement Cache)**：自适应地在 recency 和 frequency 间平衡
- 何时触发驱逐？—— GPU block pool 使用率超过阈值时

### 3. OpenAI Extended Prompt Caching 分析（03-extended-caching.md）

**OpenAI 的做法（截至 2026 年 4 月）：**
- 标准缓存：KV tensors 保持在 GPU 显存中，5-10 分钟过期
- **Extended Caching**：将 KV tensors offload 到 GPU-local storage（本地 NVMe）
  - 最长保留 24 小时
  - 只存 KV tensors（注意力层的中间表示），不存原始文本
  - 支持 ZDR（Zero Data Retention）合规
- 配置：`"prompt_cache_retention": "24h"`

**技术推测（基于已公开信息）：**
- Hash-based routing：通过 prompt 前缀的 hash 将请求路由到同一台机器
- `prompt_cache_key`：用户可指定额外的路由 key，提高 cache 命中率
- 每个 prefix + key 组合限制约 15 req/min，避免单机过载

### 4. LMCache 集成（04-lmcache.md）

**LMCache 是什么：**
- 独立的 KV Cache 管理库，可与 vLLM 集成
- 支持多种后端：CPU DRAM、Redis、本地磁盘
- 跨请求、跨实例的 KV Cache 共享

**vLLM 集成方式：**
- `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py`
- 配置 `--kv-transfer-config` 使用 LMCache 后端

**适用场景：**
- 多个 vLLM 实例共享 KV Cache（如 A/B 测试场景）
- 需要超大容量 KV Cache 存储

### 5. FlexKV 与未来方向（05-flexkv.md）

**FlexKV：**
- vLLM 中的灵活 KV Cache 管理实验特性
- 支持更细粒度的 offloading 控制
- `prefix_caching_flexkv` 示例

## 📄 参考论文

| 论文 | 年份 | 核心贡献 |
|------|------|----------|
| [FlexGen: High-Throughput Generative Inference with a Single GPU](https://arxiv.org/abs/2303.06865) | 2023 | GPU-CPU-SSD offloading 策略 |
| [InfiniGen: Efficient Generative Inference with Dynamic KV Cache Management](https://arxiv.org/abs/2406.19707) | 2024 | 动态 KV Cache 管理 |
| [CacheGen: KV Cache Compression and Streaming for Fast LLM Serving](https://arxiv.org/abs/2310.07240) | 2023 | KV Cache 压缩传输 |

## 📁 文件清单

- [ ] `01-tiered-storage.md` — 分层存储原理
- [ ] `02-vllm-offloading.md` — vLLM Offloading 源码分析
- [ ] `03-extended-caching.md` — OpenAI Extended Prompt Caching
- [ ] `04-lmcache.md` — LMCache 集成
- [ ] `05-flexkv.md` — FlexKV 与未来方向
- [ ] `exercises.md` — 动手练习
