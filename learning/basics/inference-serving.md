# Inference Serving Basics

## 什么是推理服务

大模型推理服务的目标是把模型权重变成可被业务系统调用的 API。

最小链路：

```text
模型权重
→ 推理引擎加载模型
→ HTTP API 暴露服务
→ 客户端发送 prompt
→ 服务返回生成结果
```

## 为什么需要 vLLM

vLLM 是高性能 LLM 推理引擎，常用于部署 OpenAI-compatible API。

核心价值：

- 管理模型权重。
- 管理 KV Cache。
- 支持 batching。
- 支持 streaming 输出。
- 暴露 `/v1/chat/completions` 等接口。

## 为什么不直接把 vLLM 暴露给用户

业界交付通常会在 vLLM 前面放一个 API Gateway。

原因：

- 统一鉴权。
- 统一错误码。
- 控制超时。
- 做限流。
- 做日志脱敏。
- 暴露业务 metrics。
- 隐藏后端推理引擎细节。

本项目使用 FastAPI Gateway 模拟这个交付边界。
