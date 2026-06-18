# Build Log

## 2026-06-08：项目初始化

### 背景

项目目标是把《大模型推理部署工程师学习笔记》转化为一个可执行、可展示、可复盘的 GitHub 作品集项目。

项目定位：

```text
LLM Inference Delivery Lab
= 企业交付包 + 实验平台
```

主线不是写一个 toy demo，而是模拟业界 LLM 推理服务交付流程：

```text
模型权重
→ vLLM / SGLang 推理服务
→ Docker 镜像封装
→ FastAPI Gateway
→ Benchmark 压测
→ Metrics
→ 后续 K8s / 量化 / 可观测 / 异构迁移
```

### 已完成决策

- 采用单一大项目，而不是 10 个分散 demo。
- 0 成本优先，主实验环境使用本地 GDX Spark / NVIDIA GB10。
- 3 台普通 Ubuntu 服务器用于 K8s、监控、日志、网关、压测控制面。
- 免费 GPU 平台只做 notebook 和资源限制评估，不作为稳定 benchmark 主证据。
- 主线模型按业务场景选择，不按模型热度选择。
- MVP 聚焦 Phase 0–1：项目基线 + 单机推理交付闭环。

### 当前执行阶段

正在执行：

```text
Phase 0–1 MVP Implementation Plan
```

当前任务：

```text
Task 1: Create repository baseline files
```

### 已创建内容

- `.gitignore`
- `.env.example`
- `pyproject.toml`
- `Makefile`
- `reports/.gitkeep`
- `benchmark/results/.gitkeep`
- `learning/` 学习复盘目录
