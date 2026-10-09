# Question 质量评估

公开入口仍使用原来的 `compute_quality_metrics.py`，内部子命令名仍为
`qdr`，以保持代码和结果缓存的连续性。论文图表使用三类正式术语：
**answer-key accuracy**、**option-set validity** 和 **stem clarity**；
入口不会把内部实现名当作论文指标名称。

## 先解析

```bash
python compute_quality_metrics.py parse \
  --group-dir baseline=<path/to/case-md> \
  --group-dir candidate=<path/to/another-case-md> \
  --output-dir <local-quality-run>
```

程序会自动识别三种现有目录布局，严格按 UTF-8 解析，并在
`<local-quality-run>/parsed/` 写出每组 JSONL。用 `--limit N` 可以先处理前 N
个 case 进行格式验证。

## 运行质量审查

真实运行需要远程 Judge：

```bash
python compute_quality_metrics.py qdr \
  --parsed-dir <local-quality-run>/parsed \
  --output-dir <local-quality-run> \
  --workers 8
```

该流程先用多个 solver 独立作答，只对达到分歧阈值的题目调用 analyzer，并把
缓存写入输出目录。`--solvers` 可显式传入逗号分隔的模型名；`--limit` 只用于
小样试跑。程序不会读取或修改训练集，也不自动检查训练/评估题目重叠。

## 离线 fixture

无需密钥即可验证解析、缓存布局和摘要格式：

```bash
python compute_quality_metrics.py parse \
  --group-dir demo=fixtures/question_inputs/demo \
  --output-dir scratch/demo-quality
python compute_quality_metrics.py qdr \
  --parsed-dir scratch/demo-quality/parsed \
  --output-dir scratch/demo-quality \
  --mock
```

`--mock` 只用于流程验收，不能替代真实 Judge 结果。
