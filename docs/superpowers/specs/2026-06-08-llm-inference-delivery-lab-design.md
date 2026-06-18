# LLM Inference Delivery Lab 设计文档

## 1. 项目定位

`llm-inference-delivery-lab` 是一个面向 LLM 推理部署工程师能力训练与 GitHub 展示的端到端项目。它不是单一 demo，也不是学习笔记合集，而是模拟企业级 LLM 推理服务交付流程：从模型权重加载、推理引擎服务化、Docker/Kubernetes 部署、API 网关、压测调优、量化对比、可观测性、稳定性演练，到免费 GPU 平台评估和国产/异构工具链迁移设计。

项目主叙事是：把 14B 级大模型从“权重文件”交付为“可调用、可监控、可压测、可回滚、可复盘”的推理服务。

## 2. 资源约束

### 2.1 自有资源

- 本地主力设备：GDX Spark / NVIDIA GB10。
- 驱动与 CUDA：Driver 580.82.09，CUDA 13.0。
- 已观察到本机可运行 `VLLM::EngineCore`，进程曾占用约 84.5 GiB。
- 3 台普通 Ubuntu 22.04.5 服务器：CPU 均 24 核，内存约 31 / 47 / 39 GiB，磁盘均约 98 GiB。
- 联想电脑：作为开发机、客户端、文档与轻量压测节点。

### 2.2 预算策略

项目默认采用 0 成本优先策略：

- GB10 承担核心推理实验。
- 3 台服务器承担 K8s、监控、日志、网关和压测控制面。
- 免费 GPU 平台只用于 notebook、小模型验证和平台限制评估。
- 云 GPU 租赁默认不用，仅作为未来 optional extension。

### 2.3 免费 GPU 平台边界

免费 GPU 平台不作为稳定 serving 或可靠 benchmark 主证据。它们只用于评估：

- 是否提供免费 GPU。
- GPU 类型和可用时长。
- 是否支持安装依赖。
- 是否支持持久化模型缓存。
- 是否适合 vLLM、量化、notebook 复现或轻量实验。
- 为什么不适合长稳服务、稳定压测或企业交付。

## 3. 总体架构

项目采用“企业交付包 + 实验平台”的统一形态。

### 3.1 企业交付链路

```text
模型权重
→ 本地加载验证
→ vLLM / SGLang 推理服务
→ Docker 镜像封装
→ FastAPI API Gateway
→ Benchmark 压测与调参
→ Prometheus / Grafana / Loki 可观测
→ K8s 部署与灰度发布
→ 稳定性演练
→ 交付文档 / 运维手册 / 迁移 checklist
```

### 3.2 模块分层

#### Model Layer

职责：

- 模型加载验证。
- tokenizer 和 `config.json` 解读。
- 显存估算。
- 量化对比。
- 70B 级模型资源边界分析。
- 国产/异构工具链支持矩阵。

#### Serving Layer

职责：

- vLLM / SGLang 服务化。
- OpenAI-compatible API。
- streaming 输出。
- Docker 镜像与启动脚本。
- 推理引擎参数实验。

#### Gateway Layer

职责：

- FastAPI 网关。
- `/v1/chat/completions` 代理。
- streaming 转发。
- 超时控制、限流、简单鉴权。
- 错误码归一。
- 日志脱敏。
- metrics 暴露。

#### Evaluation Layer

职责：

- benchmark harness。
- TTFT、TPOT、TPS、P50/P95/P99 统计。
- 并发曲线。
- 7B vs 14B 对比。
- 量化前后性能和质量对比。
- reasoning workload 专题压测。

#### Platform Layer

职责：

- Docker Compose 部署。
- Kubernetes 部署。
- Helm 或 Kustomize。
- GPU worker 接入。
- Service / Ingress。
- HPA 或 KEDA。
- 灰度发布与回滚。

#### Ops / Reliability Layer

职责：

- Prometheus / Grafana / Loki。
- GPU metrics。
- 请求日志和错误日志。
- OOM 演练。
- 长上下文压测。
- 异常请求压测。
- 长稳运行报告。
- incident playbook。

## 4. 子项目矩阵

10 个子项目统一组成一个大项目，不作为分散 demo。

| 编号 | 子项目 | 目标 |
|---|---|---|
| 01 | Model Loading Baseline | 模型加载、tokenizer、显存估算、基础推理 |
| 02 | vLLM & SGLang Serving | 推理引擎服务化、OpenAI API、streaming |
| 03 | Benchmark Harness | TTFT / TPOT / TPS / P95 / P99 压测 |
| 04 | Quantization Lab | FP16/BF16/AWQ/GPTQ/INT4 对比 |
| 05 | Gateway & API Hardening | FastAPI 网关、超时、限流、鉴权、错误码 |
| 06 | Observability Stack | Prometheus / Grafana / Loki / GPU metrics |
| 07 | K8s Inference Platform | GPU 节点、Helm、HPA/KEDA、灰度、回滚 |
| 08 | Free GPU Platform Evaluation | 免费 GPU 平台适用边界评估 |
| 09 | Heterogeneous Toolchain Study | Ascend / Cambricon / DCU 迁移方法论 |
| 10 | Reliability & Incident Playbook | OOM、长稳、异常请求、事故复盘 |

每个子项目必须产出四类证据：

```text
代码：能运行的脚本、服务、配置
数据：benchmark CSV/JSON、日志、指标截图
报告：实验方法、结果、结论、限制
复盘：踩坑、失败原因、调参依据、下一步
```

## 5. 技术栈选型

### 5.1 推理引擎

主线：

- vLLM。

原因：

- OpenAI-compatible API 成熟。
- PagedAttention 是核心学习点。
- benchmark 和部署资料多。
- 参数调优贴近推理部署岗位。

辅线：

- SGLang。

原因：

- 用作 serving 对比。
- 补充 structured generation 和高性能 serving 视角。

后期 extension：

- TensorRT-LLM。
- Triton Inference Server。
- llama.cpp。

### 5.2 API 网关

- FastAPI。
- Uvicorn。
- httpx / aiohttp。
- Prometheus client。

职责：

- 统一 API。
- 代理 vLLM / SGLang。
- 处理 streaming。
- 暴露 `/health` 和 `/metrics`。
- 提供超时、限流、鉴权、错误码和日志脱敏。

### 5.3 部署

第一阶段：

- Docker。
- Docker Compose。

第二阶段：

- Kubernetes。
- Helm 或 Kustomize。

不建议一开始直接上 K8s，因为会把模型、容器、网络、GPU 调度和监控问题混在一起。

### 5.4 可观测性

- Prometheus。
- Grafana。
- Loki。
- node-exporter。
- DCGM exporter 或自定义 `nvidia-smi` exporter。

如果 GB10 对 DCGM 支持不完整，则降级为自定义 GPU metrics collector。

### 5.5 Benchmark

主线采用自研轻量 benchmark harness：

- Python。
- asyncio。
- httpx。
- streaming TTFT 采集。
- TPOT / TPS / P50 / P95 / P99 统计。
- CSV / JSON 输出。

可参考但不强依赖：

- vLLM benchmark scripts。
- llmperf。
- hey / wrk / locust。

## 6. 业务场景与模型策略

项目采用业务场景驱动，而不是模型热度驱动。

### 6.1 主业务场景：企业知识库问答

Workload 特征：

- 固定 system prompt。
- 用户问题较短到中等。
- RAG context 较长。
- 输出中等。
- 需要 streaming。
- 中文能力重要。

模型：

- 主线模型：`Qwen2.5-14B-Instruct`。
- 对照模型：`Qwen2.5-7B-Instruct`。
- 边界分析：`Qwen2.5-72B-Instruct`，只做资源估算，不承诺实跑。

重点指标：

- TTFT。
- P95 / P99 latency。
- TPS。
- KV Cache usage。
- Prefix cache hit rate。
- 并发稳定性。
- 错误率。

实验重点：

- 固定 system prompt 下 prefix cache 是否有效。
- 长 context 对 TTFT 的影响。
- `max_model_len` 对并发上限的影响。
- 7B vs 14B 的质量、延迟、显存取舍。

### 6.2 业务场景：代码/技术助手

Workload 特征：

- prompt 可能较长。
- 输出可能较长。
- 代码块多。
- streaming 体验重要。
- P99 敏感。

模型：

- 可选主模型：`Qwen2.5-Coder-14B-Instruct`。
- 可选对照模型：`Qwen2.5-Coder-7B-Instruct`。
- 第一版可复用 `Qwen2.5-14B-Instruct` 和 `Qwen2.5-7B-Instruct`，避免模型族过多。

重点指标：

- TTFT。
- TPOT。
- streaming token 间隔。
- 长上下文成功率。
- 输出截断率。

### 6.3 专题场景：Reasoning 分析任务

Reasoning 不是主交付场景，而是压力 workload。

Workload 特征：

- 输出长。
- decode 阶段压力大。
- `max_tokens` 敏感。
- 请求耗时长。
- 并发下队列容易堆积。

模型：

- `DeepSeek-R1-Distill-Qwen-14B`。

重点指标：

- TPOT。
- TPS。
- P99 latency。
- 请求超时率。
- 队列等待时间。

实验重点：

- reasoning 长输出如何拖慢并发。
- `max_tokens` 限制对服务稳定性的影响。
- streaming 是否改善用户体验。
- 限流和超时策略如何保护服务。

### 6.4 模型策略总结

正式项目主叙事只保留少量模型：

```text
主线模型：Qwen2.5-14B-Instruct
对照模型：Qwen2.5-7B-Instruct
Reasoning 专题：DeepSeek-R1-Distill-Qwen-14B
边界分析：Qwen2.5-72B-Instruct，只做估算
```

小模型最多作为开发 smoke test，不作为 GitHub 主叙事。

## 7. MVP 闭环

第一阶段最小可行闭环：

```text
Qwen2.5-14B-Instruct
→ vLLM serve
→ FastAPI gateway
→ Docker Compose
→ benchmark harness
→ Prometheus metrics
→ README + report
```

MVP 交付物：

1. 一条命令启动服务。
2. `curl` 可调用 `/v1/chat/completions`。
3. Python client 支持 streaming。
4. benchmark 输出 TTFT / TPOT / TPS / P50 / P95 / P99。
5. `/metrics` 可看到请求指标。
6. 基础报告解释部署过程、指标结果、瓶颈和限制。

MVP 不包含：

- 完整 K8s 平台。
- 多模型大规模矩阵。
- 国产芯片实测 benchmark。
- 生产级安全体系。

这些属于后续阶段。

## 8. 阶段里程碑

### Phase 0：仓库初始化与项目基线

目标：建立统一项目骨架和实验规范。

交付物：

- `README.md`。
- `docs/architecture.md`。
- `docs/scenario-slo-matrix.md`。
- `docs/experiment-template.md`。
- `docs/report-template.md`。
- `.env.example`。
- `Makefile`。
- 目录结构。

### Phase 1：单机推理交付闭环

目标：在 GB10 上完成 14B 模型的本地推理服务闭环。

交付物：

- vLLM 启动脚本。
- Dockerfile。
- `docker-compose.yaml`。
- OpenAI-compatible API 调用示例。
- streaming client。
- FastAPI gateway MVP。
- benchmark harness MVP。
- 基础 benchmark report。

### Phase 2：量化与参数调优

目标：围绕 14B 模型制造并分析真实资源压力。

交付物：

- 7B vs 14B 对比报告。
- BF16/FP16 vs AWQ/GPTQ/INT4 对比报告。
- `max_model_len` / `max_num_seqs` / `gpu_memory_utilization` 调参实验。
- 长上下文 OOM case study。
- reasoning workload 专题压测。

### Phase 3：可观测性与稳定性

目标：从“能跑”升级为“可运维”。

交付物：

- Prometheus metrics。
- Grafana dashboard。
- Loki 日志采集。
- `/health` / `/metrics`。
- 错误码规范。
- 日志脱敏。
- 异常请求压测。
- 长稳运行报告。
- incident playbook。

### Phase 4：K8s 推理平台

目标：用 GB10 + 3 台服务器模拟小型生产推理平台。

交付物：

- K8s manifests 或 Helm Chart。
- GPU worker 节点部署说明。
- Service / Ingress。
- ConfigMap / Secret。
- HPA 或 KEDA 策略。
- 灰度发布流程。
- 回滚流程。
- 平台部署报告。

### Phase 5：免费 GPU 平台评估

目标：系统评估免费 GPU 平台是否适合 LLM 推理部署实验。

交付物：

- Colab / Kaggle / Lightning / Modal 调研表。
- notebook 最小复现实验。
- 免费平台限制矩阵。
- 适用 / 不适用结论。

### Phase 6：国产/异构工具链迁移设计

目标：在没有国产硬件资源时，先建立专业迁移方法论。

交付物：

- Ascend / Cambricon / DCU 工具链对比。
- 驱动 / 运行时 / 容器 / 模型格式 / 精度 / 算子支持矩阵。
- NVIDIA → 国产芯片迁移 checklist。
- 常见错误 taxonomy。
- 后续真实资源实验计划。

推荐工具链：

- **KernelCAT**（kernelcat.cn）：面向国产算力的 AI 工具，支持用自然语言驱动算子开发、模型迁移与性能优化，已验证在昇腾 CANN 上的算子适配加速能力。有真实国产硬件资源时，KernelCAT 是替代手动写 Ascend C 的首选工具。
- **Kerminal**：基于 KernelCAT 引擎，支持本地化运行、文件读写和命令执行，适合本地开发环境下的算子调试。

## 9. 推荐仓库结构

```text
llm-inference-delivery-lab/
├── README.md
├── docs/
│   ├── architecture.md
│   ├── delivery-guide.md
│   ├── scenario-slo-matrix.md
│   ├── benchmark-methodology.md
│   ├── quantization-report.md
│   ├── free-gpu-evaluation.md
│   ├── heterogeneous-migration.md
│   └── incident-playbook.md
├── models/
│   ├── configs/
│   ├── memory-estimator/
│   └── loading-baseline/
├── serving/
│   ├── vllm/
│   ├── sglang/
│   ├── clients/
│   └── docker/
├── gateway/
│   ├── app/
│   ├── tests/
│   └── Dockerfile
├── benchmark/
│   ├── harness/
│   ├── workloads/
│   ├── results/
│   └── notebooks/
├── quantization/
│   ├── scripts/
│   ├── eval-prompts/
│   └── results/
├── observability/
│   ├── prometheus/
│   ├── grafana/
│   └── loki/
├── deploy/
│   ├── docker-compose/
│   ├── k8s/
│   └── helm/
├── free-gpu-evaluation/
├── heterogeneous/
├── reliability/
└── reports/
```

## 10. GitHub 展示方式

README 首页建议包含：

1. 项目一句话定位。
2. 架构图。
3. 当前硬件环境。
4. 支持的业务场景。
5. 快速启动。
6. Benchmark 摘要表。
7. 量化对比摘要表。
8. K8s 部署截图。
9. Grafana Dashboard 截图。
10. 免费 GPU 平台评估结论。
11. 国产工具链迁移 checklist。
12. 目录结构。
13. Roadmap。
14. 简历 bullet。

推荐英文项目描述：

```text
This project simulates an enterprise LLM inference delivery workflow: from model loading and vLLM/SGLang serving to Docker/Kubernetes deployment, OpenAI-compatible API gateway, benchmarking, quantization comparison, observability, reliability testing, and heterogeneous accelerator migration planning.
```

推荐中文简历描述：

```text
构建端到端 LLM 推理服务交付实验室，基于本地 NVIDIA GB10 设备完成 14B 级模型的 vLLM/SGLang 服务化、Docker/K8s 部署、OpenAI-compatible API 网关、并发压测、量化对比、Prometheus/Grafana 可观测性、稳定性演练，并沉淀免费 GPU 平台评估与国产芯片迁移 checklist。
```

## 11. 风险与边界

### 11.1 技术风险

- GB10 显存信息在 `nvidia-smi` 中显示 `Not Supported`，需要结合 vLLM 日志、进程占用和实际压测结果判断。
- CUDA 13.0 可能带来 vLLM、PyTorch、FlashAttention 等 wheel 兼容性问题，需实测。
- 14B 模型在高并发或长上下文下可能 OOM，这是预期内的工程训练点。
- 3 台普通服务器磁盘均约 98 GiB，不适合重复存放大模型权重。

### 11.2 资源风险

- 免费 GPU 平台可用性不稳定，不能承诺长期复现。
- 没有国产硬件时，不做虚假的国产芯片 benchmark。
- 云租赁不是默认依赖，避免项目成本失控。

### 11.3 安全风险

- API key 不提交到仓库。
- Docker 镜像不内置模型权重和密钥。
- 日志需要脱敏 prompt、token 和用户标识。
- 不暴露公网无鉴权接口。
- `.env.example` 只保留占位符。

## 12. 非目标

第一版不做：

- 训练或微调模型。
- 自研推理引擎。
- 手写 CUDA kernel。
- 真实国产芯片 benchmark。
- 多租户计费系统。
- 完整 Agent 平台。
- RAG 应用主线。

这些可以作为后续 extension，但不进入 MVP。

## 13. 价值锚点

本项目把“大模型推理部署工程师学习笔记”转化为可执行、可展示、可复盘的工程证据链。它解决的核心问题是：如何在有限资源下，把 14B 级模型交付成稳定、可观测、可压测、可调优、可迁移的推理服务。

## 14. 差异化

相比普通 `vllm serve` demo，本项目的差异化在于：

- 以业务场景和 SLO 驱动模型选择，而不是追模型热度。
- 以 14B 模型作为主线，制造真实显存、KV Cache、并发和长上下文压力。
- 用 7B 作为成本/性能/质量 baseline，而不是堆很多模型。
- 将 Docker、Gateway、Benchmark、Observability、K8s、Reliability 纳入同一条交付链路。
- 明确免费 GPU 平台的适用边界，不伪装成生产资源。
- 在无国产硬件资源时先沉淀迁移方法论，后续有资源再补真实实测。
- 用报告、指标、失败案例和 incident playbook 展示工程判断力。
