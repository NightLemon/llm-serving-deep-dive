# vLLM Disaggregated Prefill 源码分析

> 本节深入 vLLM 的 disaggregated prefill 实现，从框架设计到具体 connector，完整走读请求的生命周期。

## 1. 整体架构

vLLM 的 disaggregated prefill 实现分布在以下几个关键目录中：

```
vllm/
├── distributed/
│   └── kv_transfer/
│       ├── kv_connector/
│       │   ├── v1/                        # V1 引擎 connector 实现
│       │   │   ├── base.py                # KVConnectorBase_V1 抽象基类
│       │   │   ├── nixl_connector.py      # NIXL 传输 connector
│       │   │   ├── lmcache_connector.py   # LMCache connector
│       │   │   └── p2p/
│       │   │       └── p2p_nccl_connector.py  # P2P NCCL connector
│       │   └── factory.py                 # Connector 工厂方法
│       └── kv_transfer_agent.py           # 传输代理，管理 connector 生命周期
├── entrypoints/
│   └── serve/
│       └── disagg/                        # 分离服务入口
│           ├── __init__.py
│           └── disagg_entrypoint.py       # 分离部署的启动逻辑
└── config.py                              # KVTransferConfig 配置类
```

## 2. 配置体系

### 2.1 KVTransferConfig

分离架构的配置通过 `--kv-transfer-config` 参数传入，解析为 `KVTransferConfig` 对象：

```python
# vllm/config.py 中的相关定义
@dataclass
class KVTransferConfig:
    """KV Cache 传输配置"""
    
    # connector 类型：决定使用哪种传输协议
    kv_connector: str  # "NixlConnector", "P2pNcclConnector", "MooncakeConnector", ...
    
    # 当前节点角色
    kv_role: str  # "kv_producer" (prefill) 或 "kv_consumer" (decode) 或 "kv_both"
    
    # 传输并行度
    kv_parallel_size: int = 1
    
    # 连接相关配置
    kv_port: int = 14579  # KV 传输端口
    kv_host: str = "localhost"
    
    # NIXL 特有配置
    nixl_buffer_size: int = 0
    nixl_buffer_device: str = "cuda"
    
    # 其他 connector 特有参数
    kv_connector_extra_config: Optional[dict] = None
```

### 2.2 启动方式

**Prefill 节点启动：**

```bash
vllm serve meta-llama/Llama-3-8B-Instruct \
    --port 8100 \
    --kv-transfer-config '{
        "kv_connector": "NixlConnector",
        "kv_role": "kv_producer"
    }'
```

**Decode 节点启动：**

```bash
vllm serve meta-llama/Llama-3-8B-Instruct \
    --port 8200 \
    --kv-transfer-config '{
        "kv_connector": "NixlConnector",
        "kv_role": "kv_consumer"
    }'
```

**Router（分离入口）启动：**

```bash
vllm serve \
    --disagg-config '{
        "prefill_urls": ["http://prefill-node:8100"],
        "decode_urls": ["http://decode-node:8200"]
    }'
```

### 2.3 使用 disagg 入口的高级配置

vLLM 提供了一个专门的 disaggregated serving 入口点，可以同时管理多个 prefill 和 decode 实例：

```bash
# 通过 YAML 配置文件启动
vllm serve --disagg-config disagg_config.yaml
```

```yaml
# disagg_config.yaml 示例
prefill:
  model: meta-llama/Llama-3-8B-Instruct
  tensor_parallel_size: 2
  instances: 2
  kv_connector: NixlConnector

decode:
  model: meta-llama/Llama-3-8B-Instruct
  tensor_parallel_size: 1
  instances: 4
  kv_connector: NixlConnector
  max_num_seqs: 256
```

## 3. KVConnectorBase_V1：抽象基类

所有 V1 引擎的 KV Transfer connector 都继承自 `KVConnectorBase_V1`，定义了统一的接口：

```python
# vllm/distributed/kv_transfer/kv_connector/v1/base.py
class KVConnectorBase_V1(ABC):
    """V1 KV Transfer Connector 抽象基类
    
    定义了 prefill 和 decode 两端需要实现的核心接口。
    这些方法会在 vLLM 引擎的不同阶段被调用。
    """
    
    def __init__(self, config: KVTransferConfig, role: str):
        self.config = config
        self.role = role  # "kv_producer" or "kv_consumer"
    
    # ==================== Scheduler 侧接口 ====================
    
    @abstractmethod
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> int:
        """查询远端是否有可复用的 KV Cache
        
        在 decode 端的 scheduler 中调用。如果远端（prefill 端）已经为
        这个 request 准备好了 KV Cache，返回可复用的 token 数量。
        
        Returns:
            可复用的 token 数量。如果返回 > 0，scheduler 会跳过对应
            token 的 prefill，直接从远端拉取 KV Cache。
        """
        ...
    
    @abstractmethod
    def update_state_after_alloc(
        self,
        request: "Request",
        block_ids: list[int],
        num_external_tokens: int,
    ):
        """分配 block 后的状态更新
        
        scheduler 为请求分配了 KV Cache block 之后调用。
        connector 需要记录 block 映射关系，以便后续传输。
        """
        ...
    
    # ==================== Worker 侧接口 ====================
    
    @abstractmethod
    def register_kv_caches(self, kv_caches: dict):
        """注册本地 KV Cache 内存区域
        
        在引擎启动时调用，将 KV Cache 的显存地址注册到传输层。
        对于 RDMA 方案，这一步会做 memory registration。
        """
        ...
    
    @abstractmethod
    def start_load_kv(self, forward_context: "ForwardContext"):
        """开始加载远端 KV Cache（decode 端）
        
        在 model forward 之前调用。对于 decode 端，这会触发从
        prefill 端拉取 KV Cache 到本地 block 中。
        """
        ...
    
    @abstractmethod
    def wait_for_layer_load(self, layer_idx: int):
        """等待某一层的 KV Cache 加载完成
        
        在计算该层 attention 之前调用，确保 KV Cache 已就绪。
        支持逐层流水线加载。
        """
        ...
    
    @abstractmethod
    def save_kv_layer(
        self,
        layer_idx: int,
        kv_cache: torch.Tensor,
        forward_context: "ForwardContext",
    ):
        """保存一层的 KV Cache（prefill 端）
        
        prefill 计算完一层后调用。对于流水线传输，
        这会立即开始传输这一层。
        """
        ...
    
    @abstractmethod
    def wait_for_save(self):
        """等待所有 KV Cache 保存/传输完成
        
        在 prefill 端的 model forward 完成后调用。
        """
        ...
```

## 4. NixlConnector 详细分析

### 4.1 初始化流程

```python
# vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py
class NixlConnector(KVConnectorBase_V1):
    
    def __init__(self, config: KVTransferConfig, ...):
        super().__init__(config, ...)
        
        # Step 1: 创建 NIXL Agent
        # 每个 vLLM worker 进程创建一个 NIXL agent
        self.nixl_agent = nixl.nixlAgent(
            name=f"vllm_{self.engine_id}",
        )
        
        # Step 2: 初始化连接管理
        self.remote_agents = {}  # 远端 agent 信息
        self.pending_transfers = {}  # 进行中的传输
        
        # Step 3: 准备 memory descriptor
        # KV Cache 的内存区域描述符（注册后用于 RDMA）
        self.kv_mem_descs = {}
```

### 4.2 KV Cache 注册

```python
    def register_kv_caches(self, kv_caches: dict):
        """将 KV Cache 显存注册到 NIXL，使其可被远端 RDMA 访问"""
        
        # kv_caches 的结构：
        # {
        #     layer_0: (k_cache_tensor, v_cache_tensor),
        #     layer_1: (k_cache_tensor, v_cache_tensor),
        #     ...
        # }
        
        for layer_idx, (k_cache, v_cache) in enumerate(kv_caches):
            # 获取 GPU 显存地址
            k_base_addr = k_cache.data_ptr()
            v_base_addr = v_cache.data_ptr()
            
            # 创建 NIXL memory descriptor
            # 这会触发 ibv_reg_mr (RDMA memory registration)
            k_desc = self.nixl_agent.create_mem_desc(
                addr=k_base_addr,
                size=k_cache.nbytes,
                mem_type="VRAM",  # GPU 显存
                gpu_id=self.local_gpu_id,
            )
            v_desc = self.nixl_agent.create_mem_desc(
                addr=v_base_addr,
                size=v_cache.nbytes,
                mem_type="VRAM",
                gpu_id=self.local_gpu_id,
            )
            
            self.kv_mem_descs[layer_idx] = (k_desc, v_desc)
        
        # 将本地 memory descriptor 发布到 metadata service
        # 使远端 agent 可以获取地址信息
        self._publish_memory_info()
```

### 4.3 Prefill 端：发送 KV Cache

```python
    def save_kv_layer(self, layer_idx, kv_cache, forward_context):
        """Prefill 端: 一层计算完后立即开始传输"""
        
        # 获取当前 batch 中需要传输的请求信息
        for req_id, req_info in forward_context.disagg_requests.items():
            # 获取本地 block IDs (源)
            local_block_ids = req_info.local_block_ids
            
            # 获取远端 block IDs (目的)
            remote_block_ids = req_info.remote_block_ids
            remote_agent_name = req_info.remote_agent
            
            # 构建传输描述符（block 级粒度）
            xfer_descs = []
            for local_bid, remote_bid in zip(local_block_ids, remote_block_ids):
                # 计算 block 在 KV Cache tensor 中的偏移
                local_offset = local_bid * self.block_size_bytes
                remote_offset = remote_bid * self.block_size_bytes
                
                # K cache block
                xfer_descs.append(nixl.XferDesc(
                    src_desc=self.kv_mem_descs[layer_idx][0],  # K
                    src_offset=local_offset,
                    dst_agent=remote_agent_name,
                    dst_desc_id=layer_idx * 2,  # K desc
                    dst_offset=remote_offset,
                    size=self.block_size_bytes,
                ))
                
                # V cache block
                xfer_descs.append(nixl.XferDesc(
                    src_desc=self.kv_mem_descs[layer_idx][1],  # V
                    src_offset=local_offset,
                    dst_agent=remote_agent_name,
                    dst_desc_id=layer_idx * 2 + 1,  # V desc
                    dst_offset=remote_offset,
                    size=self.block_size_bytes,
                ))
            
            # 提交异步传输（RDMA WRITE）
            handle = self.nixl_agent.submit_xfer(xfer_descs)
            self.pending_transfers.setdefault(req_id, []).append(handle)
    
    def wait_for_save(self):
        """等待所有传输完成"""
        for req_id, handles in self.pending_transfers.items():
            for handle in handles:
                self.nixl_agent.wait_xfer(handle)
        self.pending_transfers.clear()
```

### 4.4 Decode 端：接收 KV Cache

```python
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        """检查远端是否已准备好 KV Cache"""
        
        # 向 metadata service 查询该 request 的 KV Cache 状态
        remote_status = self._query_kv_status(request.request_id)
        
        if remote_status is None:
            return 0  # 远端还没有这个 request 的 KV Cache
        
        # 返回远端已准备好的 token 数
        return remote_status.num_tokens - num_computed_tokens
    
    def start_load_kv(self, forward_context):
        """开始从远端拉取 KV Cache"""
        
        for req_id, req_info in forward_context.disagg_requests.items():
            # 对于 NIXL (RDMA WRITE)，数据由发送端主动推送
            # decode 端不需要主动拉取，只需等待完成通知
            # 这里可以做一些预处理（如验证 block 分配）
            pass
    
    def wait_for_layer_load(self, layer_idx):
        """等待第 layer_idx 层的 KV Cache 传输完成
        
        这个方法在 attention 计算前调用，确保该层的 KV Cache 已就绪。
        支持逐层流水线：可以边接收边计算。
        """
        for req_id in self._get_pending_requests():
            # 检查该层传输是否完成
            status = self._check_layer_transfer(req_id, layer_idx)
            while not status.completed:
                # 忙等或 yield
                status = self._check_layer_transfer(req_id, layer_idx)
```

## 5. P2P NCCL Connector 分析

### 5.1 核心实现

P2P NCCL connector 使用 PyTorch 的分布式通信原语，实现相对简单：

```python
# vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py
class P2pNcclConnector(KVConnectorBase_V1):
    
    def __init__(self, config: KVTransferConfig, ...):
        super().__init__(config, ...)
        
        # 建立 NCCL 通信组
        # Prefill worker rank 和 decode worker rank 需要在同一个 group
        self.kv_group = self._init_nccl_group()
        self.peer_rank = self._get_peer_rank()
    
    def save_kv_layer(self, layer_idx, kv_cache, forward_context):
        """Prefill 端: 通过 NCCL send 发送 KV Cache"""
        k_cache, v_cache = kv_cache
        
        # 序列化需要传输的 block 数据
        for req_id, req_info in forward_context.disagg_requests.items():
            k_data = self._gather_blocks(k_cache, req_info.block_ids)
            v_data = self._gather_blocks(v_cache, req_info.block_ids)
            
            # NCCL P2P send
            dist.send(k_data, dst=self.peer_rank, group=self.kv_group)
            dist.send(v_data, dst=self.peer_rank, group=self.kv_group)
    
    def wait_for_layer_load(self, layer_idx):
        """Decode 端: 通过 NCCL recv 接收 KV Cache"""
        k_buf = self._get_layer_recv_buffer(layer_idx, "key")
        v_buf = self._get_layer_recv_buffer(layer_idx, "value")
        
        dist.recv(k_buf, src=self.peer_rank, group=self.kv_group)
        dist.recv(v_buf, src=self.peer_rank, group=self.kv_group)
        
        # 将接收到的数据散射到对应的 block 中
        self._scatter_to_blocks(k_buf, v_buf, layer_idx)
```

### 5.2 与 NIXL 的关键差异

| 维度 | NixlConnector | P2pNcclConnector |
|------|---------------|------------------|
| 传输模型 | RDMA 单边写 (zero-copy) | NCCL send/recv (双边通信) |
| Block 映射 | 直接写入目标 block 偏移 | 先接收到 buffer，再 scatter |
| 流水线 | 天然支持（异步 RDMA） | 需要 isend/irecv 实现 |
| 跨节点 | GPUDirect RDMA，低开销 | NCCL Socket/IB，有额外开销 |
| 依赖 | NIXL 库 + RDMA 驱动 | 仅需 PyTorch + NCCL |

## 6. 分离服务入口

### 6.1 disagg_entrypoint.py

vLLM 的 disagg 入口点负责协调 prefill 和 decode 实例：

```python
# vllm/entrypoints/serve/disagg/disagg_entrypoint.py (简化)
class DisaggregatedRouter:
    """Disaggregated Serving 路由器
    
    作为前端代理，接收客户端请求，协调 prefill 和 decode 流程。
    """
    
    def __init__(self, config: DisaggConfig):
        # Prefill 实例列表
        self.prefill_urls = config.prefill_urls
        # Decode 实例列表
        self.decode_urls = config.decode_urls
        # HTTP client
        self.client = httpx.AsyncClient()
    
    async def handle_request(self, request: ChatCompletionRequest):
        """处理一个完整的推理请求"""
        
        # Step 1: 选择 prefill 实例
        prefill_url = self._select_prefill(request)
        
        # Step 2: 发送 prefill 请求
        # 将请求转发到 prefill 实例
        # Prefill 实例会：
        #   a) 执行 prefill forward
        #   b) 通过 KV connector 将 KV Cache 传输到 decode 实例
        #   c) 返回 prefill 结果（包括 KV metadata）
        prefill_response = await self.client.post(
            f"{prefill_url}/v1/completions",
            json=request.dict(),
        )
        
        kv_metadata = prefill_response.json()["kv_metadata"]
        
        # Step 3: 选择 decode 实例
        decode_url = self._select_decode(request, kv_metadata)
        
        # Step 4: 发送 decode 请求
        # Decode 实例会：
        #   a) 通过 KV connector 接收/等待 KV Cache
        #   b) 执行 autoregressive decode
        #   c) 流式返回生成的 tokens
        async for chunk in self._stream_decode(decode_url, kv_metadata):
            yield chunk
    
    def _select_prefill(self, request) -> str:
        """选择最优的 prefill 实例"""
        # 基于负载的路由策略
        loads = {url: self._get_load(url) for url in self.prefill_urls}
        return min(loads, key=loads.get)
    
    def _select_decode(self, request, kv_metadata) -> str:
        """选择最优的 decode 实例"""
        # 优先选择 KV Cache 传输目标节点
        if kv_metadata.get("target_decode_url"):
            return kv_metadata["target_decode_url"]
        
        # 否则选择负载最低的 decode 实例
        loads = {url: self._get_load(url) for url in self.decode_urls}
        return min(loads, key=loads.get)
```

## 7. 完整请求生命周期

以一个使用 NIXL connector 的分离部署为例，一个请求从到达到完成的完整流程：

```
Client                Router              Prefill Worker        Decode Worker
  │                     │                      │                     │
  │──── POST /v1/chat ─►│                      │                     │
  │                     │                      │                     │
  │                     │── route to prefill ──►│                     │
  │                     │                      │                     │
  │                     │                      │── [1] 加入 scheduler │
  │                     │                      │      等待调度        │
  │                     │                      │                     │
  │                     │                      │── [2] prefill forward│
  │                     │                      │   for layer in model:│
  │                     │                      │     compute K, V     │
  │                     │                      │     save_kv_layer() ─┼──►NIXL RDMA WRITE
  │                     │                      │     attention(Q,K,V) │   (逐层流水线传输)
  │                     │                      │                     │
  │                     │                      │── [3] wait_for_save()│
  │                     │                      │   等待所有层传输完成  │
  │                     │                      │                     │
  │                     │◄── prefill done ──── │                     │
  │                     │   (返回 kv_metadata)  │                     │
  │                     │                      │                     │
  │                     │──── route to decode ──┼────────────────────►│
  │                     │   (携带 kv_metadata)  │                     │
  │                     │                      │                     │
  │                     │                      │     [4] scheduler 发现│
  │                     │                      │     get_num_new_matched_tokens > 0
  │                     │                      │     跳过 prefill     │
  │                     │                      │                     │
  │                     │                      │     [5] decode forward│
  │                     │                      │     wait_for_layer_load(0)
  │                     │                      │     attention(Q,K,V) │
  │                     │                      │     wait_for_layer_load(1)
  │                     │                      │     ...              │
  │                     │                      │                     │
  │◄──── stream token ──┼◄─────────────────────┼─────── token ───────│
  │◄──── stream token ──┼◄─────────────────────┼─────── token ───────│
  │◄──── stream token ──┼◄─────────────────────┼─────── token ───────│
  │                     │                      │                     │
  │◄──── [DONE] ────────┼◄─────────────────────┼─────── EOS ─────────│
```

### 关键步骤详解

**Step 1：Scheduler 调度**

Prefill worker 的 scheduler 将请求加入 waiting queue，与其他 prefill 请求一起组 batch。

**Step 2：Prefill Forward + 流水线传输**

模型逐层执行前向传播。每计算完一层的 K、V，立即通过 `save_kv_layer()` 发起异步 RDMA 传输。计算和传输在时间上重叠。

**Step 3：等待传输完成**

所有层的 KV Cache 计算完后，调用 `wait_for_save()` 确保最后几层的传输也完成。

**Step 4：Decode Scheduler 跳过 Prefill**

Decode 端的 scheduler 调用 `get_num_new_matched_tokens()`，发现远端已有完整的 KV Cache，于是跳过 prefill 阶段，直接进入 decode。

**Step 5：Decode Forward + 逐层等待**

Decode 端逐层执行前向传播。在计算每层 attention 前，调用 `wait_for_layer_load(layer_idx)` 确保该层的 KV Cache 已就绪。

## 8. Connector 工厂与注册

### 8.1 Factory Pattern

vLLM 使用工厂模式创建 connector：

```python
# vllm/distributed/kv_transfer/kv_connector/factory.py
class KVConnectorFactory:
    """KV Connector 工厂"""
    
    _registry = {}
    
    @classmethod
    def register(cls, name: str, connector_cls: type):
        cls._registry[name] = connector_cls
    
    @classmethod
    def create(cls, config: KVTransferConfig, ...) -> KVConnectorBase_V1:
        connector_name = config.kv_connector
        if connector_name not in cls._registry:
            raise ValueError(f"Unknown connector: {connector_name}")
        return cls._registry[connector_name](config, ...)

# 注册内置 connector
KVConnectorFactory.register("NixlConnector", NixlConnector)
KVConnectorFactory.register("P2pNcclConnector", P2pNcclConnector)
KVConnectorFactory.register("MooncakeConnector", MooncakeConnector)
KVConnectorFactory.register("LMCacheConnector", LMCacheConnector)
```

### 8.2 自定义 Connector

用户可以实现自己的 connector：

```python
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1

class MyCustomConnector(KVConnectorBase_V1):
    """自定义 KV Cache 传输 connector"""
    
    def register_kv_caches(self, kv_caches):
        # 注册 KV Cache 内存区域
        ...
    
    def save_kv_layer(self, layer_idx, kv_cache, forward_context):
        # 保存/传输一层的 KV Cache
        ...
    
    def wait_for_layer_load(self, layer_idx):
        # 等待一层 KV Cache 加载完成
        ...
    
    # ... 实现其他抽象方法

# 注册自定义 connector
KVConnectorFactory.register("MyCustomConnector", MyCustomConnector)
```

## 9. 与 V1 引擎的集成

### 9.1 在 Model Runner 中的调用点

KV connector 的方法在 vLLM V1 引擎的 model runner 中被调用：

```python
# vllm/v1/worker/gpu_model_runner.py (简化)
class GPUModelRunner:
    
    def execute_model(self, scheduler_output):
        # 1. 准备 forward context
        forward_context = self._prepare_forward_context(scheduler_output)
        
        # 2. 如果有 disagg 请求，开始加载远端 KV Cache
        if self.kv_connector and forward_context.has_disagg_requests:
            self.kv_connector.start_load_kv(forward_context)
        
        # 3. 执行模型 forward
        hidden_states = self.model.forward(
            input_ids=...,
            positions=...,
            kv_caches=self.kv_caches,
            forward_context=forward_context,
        )
        
        # 4. 如果是 prefill 端，等待 KV Cache 传输完成
        if self.kv_connector and self.is_producer:
            self.kv_connector.wait_for_save()
        
        return hidden_states
```

### 9.2 在 Attention 层中的集成

```python
# 在每层 attention 计算中的集成点
class Attention(nn.Module):
    def forward(self, query, key, value, kv_cache, forward_context):
        layer_idx = self.layer_idx
        
        # 如果是 decode 端，等待该层 KV Cache 就绪
        if forward_context.kv_connector and forward_context.is_consumer:
            forward_context.kv_connector.wait_for_layer_load(layer_idx)
        
        # 正常的 attention 计算
        output = flash_attention(query, key, value, kv_cache)
        
        # 如果是 prefill 端，保存该层 KV Cache 并开始传输
        if forward_context.kv_connector and forward_context.is_producer:
            forward_context.kv_connector.save_kv_layer(
                layer_idx, kv_cache, forward_context
            )
        
        return output
```

## 10. 调试与监控

### 10.1 日志级别

```bash
# 启用 KV transfer 详细日志
VLLM_LOGGING_LEVEL=DEBUG vllm serve ... \
    --kv-transfer-config '{"kv_connector": "NixlConnector", ...}'
```

关键日志标记：
- `kv_transfer` - 传输相关日志
- `nixl` - NIXL 底层日志
- `disagg` - 分离架构协调日志

### 10.2 关键指标

| 指标 | 含义 | 关注阈值 |
|------|------|---------|
| `kv_transfer_latency_ms` | KV Cache 传输延迟 | > 100ms 需要关注 |
| `kv_transfer_bandwidth_gbps` | 传输有效带宽 | < 50% 理论带宽需排查 |
| `kv_transfer_queue_size` | 等待传输的请求数 | 持续增长说明传输是瓶颈 |
| `prefill_to_decode_handoff_ms` | P→D 切换总延迟 | 包括传输+调度+首 token |

## 11. 小结

| 组件 | 源码位置 | 职责 |
|------|---------|------|
| `KVConnectorBase_V1` | `kv_connector/v1/base.py` | 定义统一接口 |
| `NixlConnector` | `kv_connector/v1/nixl_connector.py` | NIXL RDMA 传输 |
| `P2pNcclConnector` | `kv_connector/v1/p2p/p2p_nccl_connector.py` | NCCL P2P 传输 |
| `DisaggregatedRouter` | `entrypoints/serve/disagg/` | 路由和协调 |
| `KVTransferConfig` | `config.py` | 配置管理 |

**核心设计模式：**
1. **Strategy Pattern**：通过 connector 抽象支持多种传输协议
2. **Pipeline**：逐层计算+传输流水线，最大化重叠
3. **Factory**：connector 工厂方法，支持自定义扩展

> **下一节**：[05-when-to-use.md](05-when-to-use.md) — 何时该用分离架构、何时不该用。
