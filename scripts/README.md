# Helper Scripts

这些脚本用于把章节里的稳定公式变成可重复验算的小工具。它们只依赖 Python 标准库，适合作为本地学习和 CI 冒烟测试入口。

| 脚本 | 对应章节 | 用途 |
|------|----------|------|
| [`kv_cache_calculator.py`](kv_cache_calculator.py) | Ch01 / Ch13 | 计算 KV per token、单请求 KV、VLM 流量 mix 和粗略并发容量 |
| [`kv_transfer_calculator.py`](kv_transfer_calculator.py) | Ch05 | 估算分离架构中 KV shard/full KV 的传输大小和网络耗时 |
| [`speculative_decoding_simulator.py`](speculative_decoding_simulator.py) | Ch07 | 用 rejection sampling 模拟投机解码接受 token 数，并对比理论公式 |
| [`batching_throughput_estimator.py`](batching_throughput_estimator.py) | Ch08 | 用输出长度分布估算 static batching 浪费和 continuous batching 理想收益 |
| [`tp_comm_estimator.py`](tp_comm_estimator.py) | Ch09 | 粗略拆解 decode 阶段权重读取、KV 读取和 TP AllReduce 通信耗时 |
| [`smoke_tests.py`](smoke_tests.py) | CI | 快速运行所有脚本的代表性样例，防止文档示例悄悄失效 |

常用入口：

```bash
python scripts/smoke_tests.py
```

这些脚本不是性能 benchmark，不能替代真实 vLLM/SGLang 压测。它们的定位是帮你先把数量级和公式关系算对，再上 GPU 做实测。
