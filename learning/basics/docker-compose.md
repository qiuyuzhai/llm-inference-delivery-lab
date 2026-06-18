# Docker Compose Basics

## Docker Compose 是什么

Docker Compose 用一个 YAML 文件描述多个容器如何一起启动。

本项目 MVP 中至少有两个服务：

```text
vllm：真正加载模型并提供 OpenAI-compatible API
gateway：FastAPI 网关，对外暴露统一入口
```

## 为什么先用 Docker Compose

相比 Kubernetes，Docker Compose 更适合 MVP：

- 启动简单。
- 调试成本低。
- 适合单机闭环。
- 能先验证镜像、环境变量、端口和服务通信。

等单机链路稳定后，再迁移到 Kubernetes。

## Compose 文件关注点

关键配置：

- `image`：使用哪个镜像。
- `build`：如何构建本地服务镜像。
- `environment`：环境变量。
- `volumes`：模型缓存和脚本挂载。
- `ports`：端口映射。
- `depends_on`：服务启动依赖。

## GPU 容器注意事项

vLLM 容器需要访问 NVIDIA GPU。常见要求：

- 主机安装 NVIDIA driver。
- Docker 能访问 GPU。
- 容器镜像和 CUDA/PyTorch/vLLM 版本兼容。

如果容器启动失败，要优先看：

```text
驱动版本
CUDA 版本
容器镜像版本
PyTorch wheel
vLLM 版本
GPU 是否被 Docker 识别
```
