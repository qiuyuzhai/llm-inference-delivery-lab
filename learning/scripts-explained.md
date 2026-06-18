# Scripts Explained

## `.env.example`

`.env.example` 是环境变量模板。真实 `.env` 不应提交到 Git。

关键变量：

| 变量 | 作用 |
|---|---|
| `VLLM_BASE_URL` | FastAPI Gateway 访问 vLLM 的地址 |
| `GATEWAY_HOST` | Gateway 监听地址 |
| `GATEWAY_PORT` | Gateway 监听端口 |
| `GATEWAY_API_KEY` | 简单 API key，占位值不能用于生产 |
| `REQUEST_TIMEOUT_SECONDS` | 非流式请求超时 |
| `STREAM_TIMEOUT_SECONDS` | 流式请求超时 |
| `DEFAULT_MODEL` | 默认模型名称 |

## `pyproject.toml`

`pyproject.toml` 是 Python 项目的标准配置文件。本项目用它声明：

- 项目名称和版本。
- Python 版本要求。
- 运行依赖。
- 开发依赖。
- pytest 配置。
- ruff lint 配置。

## `Makefile`

`Makefile` 提供常用命令入口。

| 命令 | 作用 |
|---|---|
| `make test` | 运行测试 |
| `make lint` | 运行 ruff 检查 |
| `make gateway` | 本地启动 FastAPI Gateway |
| `make benchmark` | 跑一次 smoke benchmark |
| `make compose-up` | 启动 Docker Compose 交付包 |
| `make compose-down` | 停止 Docker Compose 交付包 |

当前 Makefile 使用 `.venv/bin/python`，目的是绑定项目本地虚拟环境，避免依赖系统 Python。

## `scripts/local/start-vllm-gb10.sh`

启动本地 GB10 smoke 验证用 vLLM 容器。

默认参数：

| 变量 | 默认值 | 作用 |
|---|---|---|
| `VLLM_CONTAINER_NAME` | `llm-vllm-streaming-bench` | Docker 容器名 |
| `VLLM_IMAGE` | `vllm-m27:gb10` | 本地 vLLM 镜像 |
| `MODEL_PATH` | `/home/aaron/Desktop/minimax-m2.7-local/models/Qwen2.5-1.5B-Instruct` | 宿主机模型目录 |
| `MAX_MODEL_LEN` | `4096` | vLLM 最大上下文长度 |
| `GPU_MEMORY_UTILIZATION` | `0.75` | GPU 显存利用比例 |
| `MAX_NUM_SEQS` | `8` | 最大并发序列数 |

用法：

```bash
scripts/local/start-vllm-gb10.sh
```

该脚本会检查 Docker、本地镜像、模型目录和启动脚本是否存在。它是前台进程，便于直接观察 vLLM 加载日志；需要后台运行时可由调用方自行加 `&` 或用终端多窗口。

## `scripts/local/start-gateway-local.sh`

本机启动 FastAPI Gateway，并显式覆盖 `.env` 中可能用于 Docker 内网的 `VLLM_BASE_URL`。

默认指向：

```bash
VLLM_BASE_URL=http://127.0.0.1:8000
DEFAULT_MODEL=/models/model
GATEWAY_API_KEY=change-me
```

用法：

```bash
scripts/local/start-gateway-local.sh
```

## `scripts/local/run-streaming-smoke.sh`

运行 streaming-only benchmark smoke test。脚本会先检查 Gateway `/health`，再调用 benchmark CLI。

用法：

```bash
scripts/local/run-streaming-smoke.sh
```

可通过环境变量覆盖请求数、并发和输出路径：

```bash
REQUESTS=10 CONCURRENCY=2 OUTPUT=benchmark/results/local.json scripts/local/run-streaming-smoke.sh
```

## `scripts/local/stop-local-runtime.sh`

停止本地验证链路占用的 vLLM 容器和 Gateway 进程。

用法：

```bash
scripts/local/stop-local-runtime.sh
```

该脚本只停止默认容器 `llm-vllm-streaming-bench` 和监听 `8080` 的 Gateway 进程，避免误杀其他服务。
