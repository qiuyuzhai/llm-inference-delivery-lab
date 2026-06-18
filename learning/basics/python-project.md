# Python Project Basics

## 为什么需要虚拟环境

虚拟环境让项目依赖和系统 Python 隔离。

创建方式：

```bash
python3 -m venv .venv
```

安装依赖：

```bash
.venv/bin/python -m pip install -e ".[dev]"
```

## `pyproject.toml`

`pyproject.toml` 是现代 Python 项目的核心配置文件。它可以声明：

- 项目名称。
- 版本。
- Python 版本要求。
- 运行依赖。
- 开发依赖。
- 测试配置。
- lint 配置。

## editable install

命令：

```bash
.venv/bin/python -m pip install -e ".[dev]"
```

含义：

- `-e` 表示 editable，本地代码改动会立即生效。
- `.[dev]` 表示安装项目依赖和 dev 依赖。

## pytest

运行测试：

```bash
.venv/bin/python -m pytest -q
```

常见 exit code：

| Exit code | 含义 |
|---|---|
| 0 | 测试通过 |
| 1 | 测试失败 |
| 2 | 使用或配置错误 |
| 5 | 没有收集到测试 |

## ruff

ruff 是 Python lint 工具，用于检查：

- import 顺序。
- 未使用变量。
- 语法和风格问题。
- 常见 bug 模式。

运行：

```bash
.venv/bin/ruff check gateway benchmark tests
```
