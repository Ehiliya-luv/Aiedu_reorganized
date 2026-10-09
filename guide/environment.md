# 环境

项目拆成两个实际使用的环境，避免训练框架和评估工具互相限制。

## CUDA DPO

建议使用 Python 3.11。先安装与本机 CUDA 驱动兼容的 CUDA 12.4 PyTorch
轮子，再安装仓库锁定的训练依赖：

```bash
python -m venv .venv-dpo
# Linux/macOS
source .venv-dpo/bin/activate
# Windows PowerShell
.\.venv-dpo\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r dpo/requirements_gpu.txt
```

训练启动前确认 `python -c "import torch; print(torch.cuda.is_available())"`
返回 `True`。CPU 可以用于格式检查和小规模代码测试，但不作为正式大模型
训练目标。本仓库默认支持 Linux CUDA 和 Windows CUDA；不提供 NPU 入口。

## 评估和绘图

```bash
python -m venv .venv-eval
source .venv-eval/bin/activate          # Windows 使用对应 Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements_eval.txt
```

评估依赖包括远程 Judge 所需的 OpenAI-compatible SDK、统计所需的 NumPy/
Pandas、YAML manifest、绘图所需的 Matplotlib，以及测试所需的 pytest。模型权重、推理服务
和真实输入不由依赖安装脚本下载。

复制 `.env.example` 为本地 `.env` 后填写远程 Judge 的地址、密钥和模型名。
`.env` 永远不应提交；在线运行也可以直接使用进程环境变量。
