# 项目结构

```text
Aiedu_reorganized/
├── dpo/
│   ├── main.py                 # DPO 唯一训练入口
│   ├── plot_training_metrics.py# 从 trainer_state.json 绘制训练曲线
│   ├── requirements_gpu.txt    # CUDA 训练依赖
│   └── utils/                  # 数据校验、模型解析、TRL DPO 实现
├── llm_judge/
│   ├── evaluate_result.py      # 结果评价的短入口
│   ├── evaluate_question_reasoning.py # Question/reasoning 通用入口
│   ├── anchor.py, pairwise.py  # Anchor/Pairwise 比较
│   ├── client.py               # OpenAI-compatible 远程客户端
│   ├── prompts.py              # Judge prompt
│   ├── checkpoint.py           # 可恢复的逐病例进度
│   └── stats.py, viz.py        # 统计和图
├── utils/
│   ├── judge/                  # Question 质量流程使用的 HTTP 客户端
│   ├── quality_metrics/        # Markdown 解析和 QDR 实现
│   └── text.py                 # 文本清理
├── compute_quality_metrics.py  # parse/qdr 入口
├── fixtures/                   # 合成输入，仅用于离线验收
├── paper_reproduction/
│   ├── plot_data/              # 不含病例的论文聚合 CSV
│   └── scripts/render_figures.py
├── guide/                      # 对外运行说明
├── scripts/validate_fixture.py# 从零离线验收
├── requirements_eval.txt
└── .env.example
```

`data/`、`resources/`、`output/`、`results/` 和 `scratch/` 是运行时相对
路径。它们不承担版本库中的数据分发功能，默认全部被 `.gitignore` 忽略；
运行者可以通过命令行参数或环境变量把输入、模型和产物放到其他路径。

DPO 只负责偏好训练。模型实际生成、工作流编排、RAG/反思/修复和专家
评价不在本仓库实现。Judge 模块只接收已经生成的结果进行比较；Question
质量模块接收题目 Markdown；reasoning 模块接收 reasoning Markdown。
