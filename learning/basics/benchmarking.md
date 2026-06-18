# Benchmarking Basics

## 为什么要压测

模型能跑不等于能交付。推理部署工程师需要回答：

- 首 token 多久返回？
- 每秒能生成多少 token？
- 并发上来后 P95/P99 是否恶化？
- 长上下文是否导致 OOM？
- 参数怎么调更稳？

## 核心指标

| 指标 | 含义 |
|---|---|
| TTFT | Time To First Token，首 token 延迟 |
| TPOT | Time Per Output Token，每个输出 token 的平均耗时 |
| TPS | Tokens Per Second，每秒 token 数 |
| P50 | 50% 请求低于该延迟 |
| P95 | 95% 请求低于该延迟 |
| P99 | 99% 请求低于该延迟 |
| Error Rate | 请求失败比例 |

## Prefill 和 Decode

LLM 推理分两个阶段：

```text
Prefill：处理输入 prompt，生成 KV Cache
Decode：逐 token 生成输出
```

常见现象：

- 输入越长，prefill 越重，TTFT 可能变高。
- 输出越长，decode 越重，TPOT/TPS 更关键。
- 并发越高，KV Cache 占用越大，OOM 风险越高。

## 为什么本项目自研轻量 benchmark harness

原因：

- 训练自己理解指标采集。
- 能控制 workload。
- 能记录失败请求。
- 后续方便扩展到业务场景压测。
