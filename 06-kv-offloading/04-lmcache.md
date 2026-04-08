# LMCache 集成

> LMCache 是一个独立的 KV Cache 管理库，支持多种存储后端（CPU DRAM、Redis、本地磁盘），可以与 vLLM 深度集成，实现跨请求、跨实例的 KV Cache 共享。

## 1. LMCache 概述

### 1.1 为什么需要 LMCache？

vLLM 内置的 KV Cache 管理有几个局限：

```
vLLM 内置 KV Cache 管理的局限：

1. 单实例范围：KV Cache 只在单个 vLLM 进程内有效
   - 多实例部署时，每个实例独立维护 KV Cache
   - 相同的 prompt 在不同实例上需要重复 prefill

2. 存储在 GPU HBM / CPU DRAM 中：
   - 进程重启后 KV Cache 全部丢失
   - 无法持久化

3. 存储后端固定：
   - 只支持 GPU HBM 和 CPU DRAM
   - 无法使用 Redis、磁盘等外部存储
```

LMCache 正是为了解决这些问题而设计的：

```
┌─────────────────────────────────────────────────┐
│                  LMCache 架构                    │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ vLLM     │  │ vLLM     │  │ vLLM     │      │
│  │ Instance │  │ Instance │  │ Instance │      │
│  │    A     │  │    B     │  │    C     │      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘      │
│       │              │              │            │
│       └──────┬───────┴──────┬───────┘            │
│              ▼              ▼                    │
│       ┌────────────────────────────┐             │
│       │     LMCache Connector      │             │
│       │  (vLLM KV Transfer API)    │             │
│       └────────┬───────────────────┘             │
│                │                                 │
│       ┌────────┴───────────────────┐             │
│       │     LMCache Core Engine    │             │
│       │  - Token-level hashing     │             │
│       │  - Chunk management        │             │
│       │  - Serialization           │             │
│       └──┬─────────┬──────────┬────┘             │
│          ▼         ▼          ▼                  │
│    ┌─────────┐┌─────────┐┌─────────┐            │
│    │CPU DRAM ││  Redis  ││  Disk   │            │
│    │ Backend ││ Backend ││ Backend │            │
│    └─────────┘└─────────┘└─────────┘            │
└─────────────────────────────────────────────────┘
```

### 1.2 核心功能

| 功能 | 说明 |
|------|------|
| 多后端存储 | 支持 CPU DRAM、Redis、本地磁盘（可组合使用） |
| 跨请求共享 | 同一实例内不同请求共享 prefix KV Cache |
| 跨实例共享 | 多个 vLLM 实例通过 Redis 共享 KV Cache |
| Token-level Hashing | 基于 token sequence 的 hash 匹配，精确到 chunk 级别 |
| 异步 I/O | 后台线程处理存储读写，不阻塞推理 |
| 压缩传输 | 可选的 KV Cache 压缩，减少存储和带宽开销 |

## 2. LMCache 架构详解

### 2.1 核心组件

```python
# LMCache 的核心组件（概念性）

class LMCacheEngine:
    """LMCache 的核心引擎，管理 KV Cache 的存储和检索。"""
    
    def __init__(self, config: LMCacheConfig):
        # 1. 初始化存储后端（可以是多级组合）
        self.backends = self._init_backends(config)
        
        # 2. Token Hasher — 将 token sequence 映射到 cache key
        self.hasher = TokenHasher(
            chunk_size=config.chunk_size,  # 通常 256 tokens
            hash_algorithm="xxhash",       # 高性能 hash
        )
        
        # 3. Chunk Manager — 管理 KV Cache 的分块存储
        self.chunk_manager = ChunkManager(
            num_layers=config.num_layers,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
        )
        
        # 4. 异步 I/O 线程池
        self.io_executor = ThreadPoolExecutor(
            max_workers=config.io_threads
        )
    
    def store(
        self,
        token_ids: List[int],
        kv_tensors: torch.Tensor,
        layer_range: Optional[Tuple[int, int]] = None,
    ):
        """存储 KV Cache 到后端。"""
        # 将 token sequence 分成 chunks
        chunks = self.hasher.chunk_and_hash(token_ids)
        
        for chunk_hash, chunk_kv in zip(chunks, self._split_kv(kv_tensors)):
            # 序列化 KV tensor
            serialized = self.chunk_manager.serialize(chunk_kv)
            
            # 异步写入所有后端（先写快的，再写慢的）
            for backend in self.backends:
                self.io_executor.submit(
                    backend.put, chunk_hash, serialized
                )
    
    def retrieve(
        self,
        token_ids: List[int],
    ) -> Optional[torch.Tensor]:
        """从后端检索 KV Cache。"""
        chunks = self.hasher.chunk_and_hash(token_ids)
        
        kv_parts = []
        for chunk_hash in chunks:
            # 按优先级查询后端（CPU DRAM → Redis → Disk）
            for backend in self.backends:
                data = backend.get(chunk_hash)
                if data is not None:
                    kv_parts.append(
                        self.chunk_manager.deserialize(data)
                    )
                    break
            else:
                # 某个 chunk 未命中，后续 chunks 也无效
                break
        
        if not kv_parts:
            return None
        
        return torch.cat(kv_parts, dim=-2)  # 沿 sequence 维度拼接
```

### 2.2 Token-Level Hashing

LMCache 的缓存匹配基于 token sequence 的 hash，而不是原始文本：

```python
class TokenHasher:
    """将 token sequence 分块并计算 hash。
    
    为什么用 token-level 而不是 text-level？
    1. 同一文本可能有不同的 tokenization 结果（不同模型/版本）
    2. Token IDs 是确定性的（不依赖文本编码方式）
    3. Hash 粒度更精确
    """
    
    def __init__(self, chunk_size: int = 256):
        self.chunk_size = chunk_size  # 每个 chunk 包含的 tokens 数
    
    def chunk_and_hash(self, token_ids: List[int]) -> List[str]:
        """将 token sequence 分块并计算每个 chunk 的 hash。
        
        关键：hash 是累积的（prefix-aware）。
        chunk_n 的 hash = hash(chunk_0 + chunk_1 + ... + chunk_n)
        这确保了前缀匹配的语义正确性。
        """
        hashes = []
        for i in range(0, len(token_ids), self.chunk_size):
            chunk_end = min(i + self.chunk_size, len(token_ids))
            # 累积 hash：包含从开头到当前 chunk 的所有 tokens
            prefix = token_ids[:chunk_end]
            chunk_hash = xxhash.xxh128(
                bytes(prefix)  # 序列化为 bytes
            ).hexdigest()
            hashes.append(chunk_hash)
        
        return hashes
```

```
示例：

Token IDs: [101, 2023, 3456, ..., 7890]  (总共 1024 tokens)
Chunk size: 256 tokens

Chunk 0: hash([101, 2023, ..., tokens[0:256]])     → "a1b2c3..."
Chunk 1: hash([101, 2023, ..., tokens[0:512]])     → "d4e5f6..."
                                                         ↑ 包含 chunk 0 的内容
Chunk 2: hash([101, 2023, ..., tokens[0:768]])     → "g7h8i9..."
Chunk 3: hash([101, 2023, ..., tokens[0:1024]])    → "j0k1l2..."
```

这种累积 hash 的设计确保了：
- 只有**完全相同前缀**的请求才会匹配
- 中间某个 token 不同，后续所有 chunk 的 hash 都不同
- 与 vLLM 的 prefix caching 语义一致

### 2.3 多级存储后端

LMCache 支持多种存储后端的组合使用：

```python
# 后端接口定义
class StorageBackend(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[bytes]:
        """检索 KV Cache 数据"""
        ...
    
    @abstractmethod
    def put(self, key: str, value: bytes) -> None:
        """存储 KV Cache 数据"""
        ...
    
    @abstractmethod
    def delete(self, key: str) -> None:
        """删除 KV Cache 数据"""
        ...
    
    @abstractmethod
    def contains(self, key: str) -> bool:
        """检查 key 是否存在"""
        ...


class CPUDRAMBackend(StorageBackend):
    """基于 CPU DRAM 的存储后端。
    
    特点：
    - 延迟最低（~100 ns 访问 + CPU→GPU PCIe 传输）
    - 容量有限（受限于系统 DRAM）
    - 非持久化（进程重启丢失）
    """
    def __init__(self, max_size_gb: float):
        self.max_size = int(max_size_gb * 1024**3)
        self.store = {}       # key -> bytes
        self.current_size = 0
        self.lru = OrderedDict()  # LRU 驱逐
    
    def get(self, key: str) -> Optional[bytes]:
        if key in self.store:
            self.lru.move_to_end(key)
            return self.store[key]
        return None
    
    def put(self, key: str, value: bytes):
        while self.current_size + len(value) > self.max_size:
            self._evict_lru()
        self.store[key] = value
        self.current_size += len(value)
        self.lru[key] = True


class RedisBackend(StorageBackend):
    """基于 Redis 的存储后端。
    
    特点：
    - 支持跨实例共享（所有 vLLM 实例共享一个 Redis）
    - 网络延迟较高（~1 ms LAN）
    - 容量可扩展（Redis Cluster）
    - 支持 TTL 自动过期
    """
    def __init__(self, redis_url: str, ttl: int = 3600):
        import redis
        self.client = redis.Redis.from_url(redis_url)
        self.ttl = ttl  # 默认 1 小时过期
    
    def get(self, key: str) -> Optional[bytes]:
        return self.client.get(f"lmcache:{key}")
    
    def put(self, key: str, value: bytes):
        self.client.setex(f"lmcache:{key}", self.ttl, value)


class DiskBackend(StorageBackend):
    """基于本地磁盘的存储后端。
    
    特点：
    - 容量最大（TB 级）
    - 持久化存储（进程重启不丢失）
    - 延迟最高（NVMe ~10 μs + 数据加载）
    - 适合大规模 KV Cache 存储
    """
    def __init__(self, cache_dir: str, max_size_gb: float):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = int(max_size_gb * 1024**3)
    
    def get(self, key: str) -> Optional[bytes]:
        path = self.cache_dir / f"{key}.bin"
        if path.exists():
            return path.read_bytes()
        return None
    
    def put(self, key: str, value: bytes):
        path = self.cache_dir / f"{key}.bin"
        path.write_bytes(value)
```

**多级存储组合：**

```python
# 配置示例：三级存储
config = LMCacheConfig(
    backends=[
        {"type": "cpu_dram", "max_size_gb": 50},    # L1: 快但小
        {"type": "redis", "url": "redis://cache:6379", "ttl": 7200},  # L2: 跨实例共享
        {"type": "disk", "path": "/mnt/nvme/lmcache", "max_size_gb": 500},  # L3: 大但慢
    ],
    chunk_size=256,
    io_threads=4,
)

# 查询顺序：CPU DRAM → Redis → Disk
# 写入顺序：同时写入所有后端（异步）
```

```
查询延迟对比：

L1 命中 (CPU DRAM): ~0.5 ms (PCIe 传输)
L2 命中 (Redis):    ~2-5 ms (网络 + PCIe)
L3 命中 (Disk):     ~5-20 ms (NVMe 读 + PCIe)
完全未命中:          ~100-5000 ms (full prefill)
```

## 3. vLLM 集成方式

### 3.1 集成架构

LMCache 通过 vLLM 的 KV Transfer Connector 接口集成：

```
vLLM Internal
┌─────────────────────────────────────────┐
│  Scheduler                               │
│    │                                     │
│    ▼                                     │
│  Block Manager                           │
│    │                                     │
│    ▼                                     │
│  KV Connector Interface                  │
│    │                                     │
│    ├── OffloadingConnector (built-in)     │
│    ├── LMCacheConnector ◄── 这个         │
│    └── ... (其他 connector)              │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│  LMCache Library    │
│  (独立进程或嵌入式)  │
└─────────────────────┘
```

### 3.2 LMCache Connector 源码分析

```python
# vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py (概念性)

from lmcache import LMCacheEngine, LMCacheConfig

class LMCacheConnector:
    """将 LMCache 集成到 vLLM 的 KV Transfer 框架。
    
    实现 vLLM 的 KVConnector 接口，将 KV Cache 的
    存储和检索委托给 LMCache Engine。
    """
    
    def __init__(self, vllm_config, kv_transfer_config):
        # 从 vLLM 配置中提取模型参数
        model_config = vllm_config.model_config
        
        # 初始化 LMCache
        lmcache_config = LMCacheConfig.from_dict(
            kv_transfer_config.get("lmcache", {})
        )
        self.engine = LMCacheEngine(lmcache_config)
        
        # 映射 vLLM 的 block 到 LMCache 的 chunk
        self.block_size = vllm_config.cache_config.block_size
    
    def send_kv_caches(
        self,
        request_id: str,
        token_ids: List[int],
        kv_tensors: torch.Tensor,
    ):
        """将 KV Cache 存储到 LMCache。
        
        在以下时机调用：
        1. Prefill 完成后，存储新计算的 KV Cache
        2. 请求被 preempt 时，保存其 KV Cache
        """
        self.engine.store(token_ids, kv_tensors)
    
    def recv_kv_caches(
        self,
        request_id: str,
        token_ids: List[int],
    ) -> Optional[Tuple[torch.Tensor, int]]:
        """从 LMCache 检索 KV Cache。
        
        在以下时机调用：
        1. 新请求到来时，检查是否有可用的 prefix KV Cache
        2. 被 preempt 的请求恢复时
        
        Returns:
            (kv_tensors, num_cached_tokens) 或 None
        """
        result = self.engine.retrieve(token_ids)
        if result is not None:
            kv_tensors, cached_length = result
            return kv_tensors, cached_length
        return None
    
    def close(self):
        """清理资源"""
        self.engine.close()
```

### 3.3 配置方法

**方式一：命令行参数**

```bash
# 使用 LMCache 作为 KV Cache 存储后端
vllm serve meta-llama/Llama-3.1-8B \
    --kv-transfer-config '{
        "kv_connector": "LMCacheConnector",
        "lmcache": {
            "backends": [
                {"type": "cpu_dram", "max_size_gb": 30},
                {"type": "redis", "url": "redis://localhost:6379"}
            ],
            "chunk_size": 256
        }
    }'
```

**方式二：配置文件**

```yaml
# lmcache_config.yaml
kv_connector: LMCacheConnector
lmcache:
  chunk_size: 256
  io_threads: 4
  compression: none          # 可选: none, zstd, lz4
  
  backends:
    - type: cpu_dram
      max_size_gb: 30
    
    - type: redis
      url: redis://cache-cluster:6379
      ttl: 7200               # 2 小时 TTL
      password: ${REDIS_PASSWORD}
    
    - type: disk
      path: /mnt/nvme/lmcache
      max_size_gb: 200
```

```bash
vllm serve meta-llama/Llama-3.1-8B \
    --kv-transfer-config lmcache_config.yaml
```

## 4. 跨实例 KV Cache 共享

### 4.1 场景描述

在生产环境中，通常部署多个 vLLM 实例来处理高并发：

```
                     Load Balancer
                     ┌──────────┐
                     │  Nginx / │
                     │  Envoy   │
                     └──┬───┬───┘
                        │   │
              ┌─────────┘   └─────────┐
              ▼                       ▼
        ┌──────────┐            ┌──────────┐
        │  vLLM    │            │  vLLM    │
        │  GPU 0-3 │            │  GPU 4-7 │
        │          │            │          │
        │ LMCache  │            │ LMCache  │
        │ Connector│            │ Connector│
        └────┬─────┘            └────┬─────┘
             │                       │
             └───────┬───────────────┘
                     ▼
              ┌──────────────┐
              │    Redis     │
              │  (共享后端)   │
              └──────────────┘
```

**没有 LMCache 时的问题：**

```
请求 A → vLLM Instance 1: 
  [System Prompt (8K tokens)] [User: Question 1]
  → Prefill 8K tokens (500ms)

请求 B → vLLM Instance 2:
  [System Prompt (8K tokens)] [User: Question 2]  ← 相同的 system prompt！
  → Prefill 8K tokens (500ms)  ← 重复计算！
```

**有 LMCache (Redis Backend) 时：**

```
请求 A → vLLM Instance 1:
  [System Prompt (8K tokens)] [User: Question 1]
  → Prefill 8K tokens (500ms)
  → 存储 KV Cache 到 Redis (异步, ~2ms)

请求 B → vLLM Instance 2:
  [System Prompt (8K tokens)] [User: Question 2]
  → 从 Redis 加载 KV Cache (~5ms)  ← 100x 加速！
  → 只需 prefill 剩余的 user message
```

### 4.2 跨实例共享的实现细节

```python
# 跨实例共享的关键：统一的 cache key 计算

# Instance 1 存储时：
token_ids = tokenizer.encode("System prompt text...")  # [101, 2023, ...]
chunk_hash = xxhash.xxh128(bytes(token_ids[:256])).hexdigest()
# → "a1b2c3d4e5f6..."

# Instance 2 查询时：
token_ids = tokenizer.encode("System prompt text...")  # 相同的 tokenizer → 相同的 token IDs
chunk_hash = xxhash.xxh128(bytes(token_ids[:256])).hexdigest()
# → "a1b2c3d4e5f6..."  ← 相同的 hash → Redis 命中！
```

**前提条件：**
- 所有实例使用相同的 tokenizer 和模型
- 所有实例连接同一个 Redis
- Token IDs 的确定性（相同文本 → 相同 token IDs）

### 4.3 一致性考虑

跨实例共享时需要注意一致性问题：

```
问题：模型更新后，旧的 KV Cache 是否还有效？

答案：不一定。模型权重变化会导致 KV Cache 数据失效。

解决方案：
1. 在 cache key 中包含模型版本/hash
   cache_key = hash(model_version + token_ids)

2. 模型更新时清除所有缓存
   redis.flushdb()

3. 使用 TTL 自然过期
   redis.setex(key, ttl=3600, value=kv_data)
```

## 5. 适用场景分析

### 5.1 高价值场景

| 场景 | 推荐后端 | 原因 |
|------|---------|------|
| 单实例长 context | CPU DRAM | 延迟最低，适合扩展 KV Cache 容量 |
| 多实例共享 system prompt | Redis | 跨实例共享，减少重复 prefill |
| A/B 测试 | Redis | 不同版本的 prompt 共享基础 KV Cache |
| 离线批处理 | Disk | 大容量存储，不需要低延迟 |
| 长期缓存（>24h） | Redis + Disk | Redis 提供快速访问，Disk 提供持久化 |

### 5.2 性能特征

```python
# 不同后端的性能特征

benchmarks = {
    "cpu_dram": {
        "store_latency_ms": 0.1,      # 很快（内存拷贝）
        "retrieve_latency_ms": 0.5,   # 快（PCIe 传输）
        "max_capacity_gb": 50,        # 受限于系统 DRAM
        "persistence": False,
    },
    "redis": {
        "store_latency_ms": 1.0,      # 网络 + 序列化
        "retrieve_latency_ms": 2.0,   # 网络 + 反序列化 + PCIe
        "max_capacity_gb": 100,       # Redis Cluster 可扩展
        "persistence": True,          # Redis 支持持久化
    },
    "disk": {
        "store_latency_ms": 5.0,      # NVMe 写入
        "retrieve_latency_ms": 10.0,  # NVMe 读 + PCIe
        "max_capacity_gb": 2000,      # TB 级
        "persistence": True,
    },
}
```

### 5.3 什么时候不应该使用 LMCache？

| 情况 | 原因 |
|------|------|
| 每次 prompt 完全不同 | 缓存命中率接近零，overhead 白费 |
| 短 prompt (<256 tokens) | 低于 chunk size，prefill 本身就很快 |
| 对延迟极度敏感（<5ms SLA） | Redis/Disk 检索延迟可能超标 |
| 单实例 + HBM 充足 | vLLM 内置的 prefix caching 已经够用 |
| 模型频繁更新 | 每次更新都会使缓存失效 |

## 6. 与 vLLM 内置 Offloading 的对比

| 维度 | vLLM 内置 Offloading | LMCache |
|------|---------------------|---------|
| 设计目标 | 扩展单实例 KV Cache 容量 | 跨实例 KV Cache 共享 + 扩展容量 |
| 存储后端 | CPU DRAM | CPU DRAM, Redis, Disk |
| 跨实例 | 不支持 | 支持（通过 Redis） |
| 持久化 | 不支持 | 支持（Redis/Disk） |
| 管理粒度 | Per-block | Per-chunk (configurable) |
| 驱逐策略 | LRU/ARC（内置） | LRU（每个后端独立） |
| 配置复杂度 | 低（几个参数） | 中（需要配置后端） |
| 额外依赖 | 无 | lmcache 库 + 后端服务 |
| 适合场景 | 单机、长 context | 多实例、共享 prefix |

**何时选择哪个？**

```
决策流程：

Q1: 是否需要跨实例共享 KV Cache？
  → 是 → LMCache (Redis 后端)
  → 否 → Q2

Q2: 是否需要持久化（进程重启后保留）？
  → 是 → LMCache (Disk 后端)
  → 否 → Q3

Q3: 是否需要超大容量 KV Cache 存储？
  → 是 → LMCache (CPU DRAM + Disk) 或 vLLM Offloading
  → 否 → vLLM 内置 prefix caching 已足够
```

## 7. 部署建议

### 7.1 Redis 后端部署

```yaml
# docker-compose.yml for Redis backend
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    command: >
      redis-server
      --maxmemory 64gb
      --maxmemory-policy allkeys-lru
      --save ""
      --appendonly no
    ports:
      - "6379:6379"
    deploy:
      resources:
        limits:
          memory: 68g   # 留一些 overhead

  vllm-instance-1:
    image: vllm/vllm-openai:latest
    command: >
      --model meta-llama/Llama-3.1-8B
      --kv-transfer-config /config/lmcache.yaml
    volumes:
      - ./config:/config
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  vllm-instance-2:
    image: vllm/vllm-openai:latest
    command: >
      --model meta-llama/Llama-3.1-8B
      --kv-transfer-config /config/lmcache.yaml
    volumes:
      - ./config:/config
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 7.2 监控指标

```python
# 关键监控指标
lmcache_metrics = {
    # 命中率
    "lmcache_hit_rate": "总体缓存命中率",
    "lmcache_hit_rate_by_backend": "各后端命中率",
    
    # 延迟
    "lmcache_store_latency_ms": "存储延迟",
    "lmcache_retrieve_latency_ms": "检索延迟",
    
    # 容量
    "lmcache_storage_used_gb": "已用存储空间",
    "lmcache_eviction_count": "驱逐次数",
    
    # 节省
    "lmcache_prefill_tokens_saved": "通过缓存节省的 prefill tokens",
    "lmcache_prefill_time_saved_ms": "通过缓存节省的 prefill 时间",
}
```

## 8. 小结

| 要点 | 说明 |
|------|------|
| LMCache 定位 | 独立的 KV Cache 管理库，弥补 vLLM 内置功能的不足 |
| 多后端支持 | CPU DRAM（快）、Redis（共享）、Disk（大） |
| 核心创新 | Token-level hashing + 跨实例共享 |
| 集成方式 | 通过 vLLM KV Connector 接口，配置即用 |
| 最佳场景 | 多实例部署 + 共享 system prompt |
| 注意事项 | 模型更新时缓存失效、额外的运维复杂度 |

---

**下一节：** [FlexKV 与未来方向](05-flexkv.md) —— 探索更灵活的 KV Cache 管理方式和未来技术趋势。
