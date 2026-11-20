"""Explicit server guard shared by training and local-model inference."""

import sys


def require_cuda():
    if sys.platform == "darwin":
        raise RuntimeError("需要在 GPU 服务器验证：此入口不在 macOS 上加载训练依赖")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("需要在 GPU 服务器验证：CUDA 不可用")
    return torch
