# Issues and Fixes

## 1. 系统没有 `python` 命令

### 现象

运行：

```bash
python -m pytest -q
```

报错：

```text
/bin/bash: line 1: python: command not found
```

### 根因

Ubuntu 环境里可能只有 `python3`，没有 `python` 软链接。

### 解决方案

项目命令不要依赖系统 `python` 别名，改用：

```bash
python3
```

创建虚拟环境后，统一使用：

```bash
.venv/bin/python
```

### 工程原则

项目命令应尽量绑定到项目本地环境，减少对系统全局配置的依赖。

---

## 2. 系统 Python 没有安装 `pytest`

### 现象

运行：

```bash
python3 -m pytest -q
```

报错：

```text
/usr/bin/python3: No module named pytest
```

### 根因

项目刚初始化，系统 Python 没有安装测试依赖。

### 解决方案

创建项目本地虚拟环境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

后续使用：

```bash
.venv/bin/python -m pytest -q
```

### 工程原则

不要污染系统 Python。每个工程项目应有独立依赖环境。

---

## 3. 在错误目录运行 pytest 会扫到其他项目

### 现象

在 `/home/aaron` 运行项目 pytest 时，pytest 扫到了其他目录中的测试，例如 llama.cpp、ComfyUI 相关测试，并出现大量无关依赖错误。

### 根因

命令执行目录不是项目根目录。pytest 会根据当前目录和配置查找测试文件。

### 解决方案

所有项目命令都应在项目根目录执行：

```bash
cd /home/aaron/Desktop/llm-inference-delivery-lab
.venv/bin/python -m pytest -q
```

或者显式指定工作目录：

```bash
git -C /home/aaron/Desktop/llm-inference-delivery-lab status
```

### 工程原则

工程命令必须明确工作目录，避免跨项目污染。

---

## 4. 空 tests 目录时 pytest 返回 exit code 5

### 现象

在项目目录运行：

```bash
.venv/bin/python -m pytest -q
```

结果没有测试文件，pytest 返回 exit code 5。

### 根因

pytest 的 exit code 5 表示“没有收集到测试”，不是代码测试失败。

### 解决方案

在 Task 1 阶段接受该状态；从 Task 3 开始添加测试后，再要求测试通过。

### 工程原则

验证结果要区分：

```text
没有测试
测试失败
环境失败
代码失败
```

不能把所有非 0 exit code 都当成同一种问题。

---

## 5. editable install 生成 `*.egg-info/`

### 现象

运行：

```bash
.venv/bin/python -m pip install -e ".[dev]"
```

后，仓库出现：

```text
llm_inference_delivery_lab.egg-info/
```

### 根因

Python editable install 会生成包元数据目录。

### 解决方案

在 `.gitignore` 中加入：

```gitignore
*.egg-info/
```

### 工程原则

构建产物和环境产物不应进入 Git。

---

## 6. `git init` 默认创建 `master` 分支

### 现象

执行：

```bash
git init
```

默认分支为：

```text
master
```

### 根因

Git 默认分支名由系统配置决定。

### 解决方案

为功能开发创建独立分支：

```bash
git checkout -b feat/phase-0-1-mvp
```

### 工程原则

不要直接在 `main` / `master` 上实现功能。即使是个人项目，也应保持分支纪律。

---

## 7. Git commit 需要本地作者身份

### 现象

执行 `git commit` 时报错：

```text
Author identity unknown
fatal: unable to auto-detect email address
```

### 根因

本机或当前仓库没有配置 `user.name` 和 `user.email`。Git commit 只是在本地生成版本快照，但仍需要记录作者身份。

### 解决方案

如果只想影响当前项目，可以设置仓库级配置：

```bash
git config user.name "Your Name"
git config user.email "you@example.com"
```

当前项目已按学习优先策略暂时跳过 Git commit，继续推进代码和文档。

### 工程原则

不要未经确认修改全局 Git 配置。仓库级配置影响范围最小，更安全。

---

## 8. `ruff B008` 检查 FastAPI / Typer 默认参数

### 现象

运行：

```bash
.venv/bin/ruff check gateway benchmark tests
```

报错示例：

```text
B008 Do not perform function call `typer.Option` in argument defaults
B008 Do not perform function call `Depends` in argument defaults
```

### 根因

`ruff` 启用了 `flake8-bugbear` 的 `B008` 规则。普通 Python 默认参数中的函数调用只会在函数定义时执行一次，容易产生共享状态问题。FastAPI 的 `Depends()`、`Header()` 和 Typer 的 `Option()` 虽然是框架常见写法，但仍会触发该规则。

### 解决方案

把框架元数据迁移到 `typing.Annotated`，让默认值保持为普通常量：

```python
from typing import Annotated

base_url: Annotated[str, typer.Option()] = "http://localhost:8080"
settings: Annotated[Settings, Depends(get_settings)]
```

修复后重新运行：

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check gateway benchmark tests
```

结果：

```text
11 passed, 1 warning
All checks passed!
```

### 工程原则

lint 失败不要简单关闭规则。优先理解规则目的，再选择兼容框架且更清晰的写法。

---

## 9. Docker Compose 没有读取项目根目录 `.env`

### 现象

已经在项目根目录 `.env` 中配置：

```dotenv
VLLM_IMAGE=vllm-m27:gb10
MODEL_PATH=/home/aaron/Desktop/minimax-m2.7-local/models/Qwen2.5-1.5B-Instruct
```

但运行：

```bash
docker compose -f deploy/docker-compose/docker-compose.yml up --build
```

仍然尝试拉取：

```text
vllm/vllm-openai:latest
```

### 根因

Compose 文件位于 `deploy/docker-compose/docker-compose.yml`。当前 Docker Compose v5 在这个调用方式下没有按预期读取项目根目录 `.env`，导致变量回退到 YAML 默认值。

### 解决方案

显式指定 env file：

```bash
docker compose --env-file .env -f deploy/docker-compose/docker-compose.yml config
```

先用 `config` 验证渲染结果，再执行 `up`。

### 工程原则

Compose 变量问题先看渲染后的配置，不要只看源码 YAML。`docker compose config` 是定位环境变量是否生效的第一工具。

---

## 10. Docker Hub 超时导致镜像无法拉取

### 现象

拉取 vLLM 或 Python 基础镜像时报错：

```text
Client.Timeout exceeded while awaiting headers
failed to resolve source metadata for docker.io/library/python:3.11-slim
```

### 根因

当前网络访问 Docker Hub 不稳定，导致 `vllm/vllm-openai:latest` 和 `python:3.11-slim` 都无法按需拉取。

### 解决方案

先检查本地已有镜像：

```bash
docker images
```

本机已有：

```text
vllm-m27:gb10
```

因此运行时验证改为使用本地 vLLM 镜像，并把 Gateway 暂时以本机 `.venv` 方式启动，绕过 Gateway Dockerfile 对 `python:3.11-slim` 的拉取依赖。

### 工程原则

真实交付中网络、镜像源、制品缓存都是部署链路的一部分。不能把镜像拉取失败误判为业务代码失败。

---

## 11. 流式输出接到 `head` 后出现 `BrokenPipeError`

### 现象

为了只查看前几行 streaming 输出，运行类似命令：

```bash
python serving/clients/stream_chat.py ... | head -20
```

客户端报错：

```text
BrokenPipeError: [Errno 32] Broken pipe
```

### 根因

`|` 是 shell 管道，表示把左边命令的输出交给右边命令。`head -20` 只读取前 20 行，读完就退出并关闭管道。左边的 streaming client 还在继续打印模型输出，写入一个已经关闭的管道，就会收到 `BrokenPipeError`。

### 解决方案

这是验证命令造成的截断现象，不是 Gateway 或 vLLM 推理失败。完整验证 streaming 时不要接 `head`：

```bash
python serving/clients/stream_chat.py --base-url http://127.0.0.1:8080 --api-key change-me --prompt "请解释 KV Cache。"
```

如果只想抽样查看输出，可以接受该错误，或改造客户端捕获 `BrokenPipeError` 后安静退出。

### 工程原则

区分“业务链路错误”和“命令行观察方式导致的错误”。流式程序遇到下游消费者提前关闭时，需要按场景决定是报错、忽略还是优雅退出。

---

## 12. 流式 benchmark 不能把字节数当 token 数

### 现象

流式响应天然会持续返回 SSE chunk。每个 chunk 有字节长度，但不一定包含 token usage。

### 根因

字节数、字符数和 token 数不是同一个概念。中文、英文、标点和 tokenizer 规则都会影响 token 数。用 bytes/sec 冒充 tokens/sec 会让压测报告失真。

### 解决方案

benchmark 只在 OpenAI-compatible stream chunk 明确返回 `usage.completion_tokens` 时计算 TPS 和 TPOT。没有 usage 时，token 字段保持 `null`。

### 工程原则

性能指标宁可缺失，也不能伪造。可观测性系统的可信度比表格完整性更重要。

---

## 13. benchmark 前 Gateway 必须处于运行状态

### 现象

运行 streaming benchmark 前先检查：

```bash
curl -fsS "http://127.0.0.1:8080/health"
```

报错：

```text
curl: (7) Failed to connect to 127.0.0.1 port 8080 after 0 ms: Couldn't connect to server
```

### 根因

本地 Gateway 进程没有监听 `127.0.0.1:8080`。benchmark harness 是压测执行器，不负责启动 vLLM 或 Gateway；被测服务必须先由部署脚本、Docker Compose 或本机 `uvicorn` 启动。

### 解决方案

先启动 vLLM 和 Gateway，再运行 benchmark。GB10 本地验证路径可以复用 `docs/mvp-report.md` 中的启动命令，并确保 Gateway 使用：

```bash
VLLM_BASE_URL="http://127.0.0.1:8000" DEFAULT_MODEL="/models/model" GATEWAY_API_KEY="change-me" \
  .venv/bin/python -m uvicorn gateway.app.main:app --host 127.0.0.1 --port 8080
```

### 工程原则

压测前必须先做 health check。把“服务未启动”和“benchmark 代码失败”区分开，避免误判根因。
