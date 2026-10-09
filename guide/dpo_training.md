# DPO 训练

## 最小运行

在仓库根目录执行：

```bash
python dpo_main.py \
  --dataset <path/to/dpo_pairs.jsonl> \
  --model-name <HuggingFace-repo-id-or-local-model> \
  --output-dir <path/to/run-output>
```

Windows PowerShell 将反斜杠换成反引号，或把命令写成一行。模型可以是本地
目录或 Hugging Face repo id；运行者自行下载并负责许可。默认使用 LoRA，
默认设备检测为 CUDA，否则退回 CPU 以便报错清晰。

## 常用参数

`--num-epochs`、`--batch-size`、`--gradient-accumulation-steps`、
`--learning-rate`、`--beta`、`--max-length` 和 `--max-prompt-length` 控制
训练；`--train-samples` 仅用于小样本试跑。`--full-finetune` 切换到全参数
训练，`--use-qlora` 在 CUDA 上启用 4-bit QLoRA。`--save-steps`、
`--save-total-limit`、`--resume` 控制 checkpoint。

训练不自动切出验证集。数据筛选阶段应由项目方保证训练数据和后续评估数据
独立。训练曲线可用：

```bash
python dpo/plot_training_metrics.py --state <run-output>/trainer_state.json
```

`python dpo/main.py` 仍然可以直接运行；根目录入口只负责转发参数。

## 运行产物

输出目录包含模型或 adapter、tokenizer、`trainer_state.json` 和 checkpoint。
运行日志默认写到被忽略的 `scratch/training.log`；也可以用
`AIEDU_TRAINING_LOG` 指定其他本地日志路径。
