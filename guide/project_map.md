# 项目结构

```text
Aiedu_reorganized/
├── dpo_main.py                    # DPO 根入口
├── question_llm_judge_main.py     # 结果评价 LLM Judge 根入口
├── question_quality_main.py        # Question 质量根入口
├── reasoning_quality_main.py       # reasoning 量化根入口
├── paper_reproduction_main.py      # 论文图复现根入口
├── dpo/
│   ├── main.py                 # DPO 实现（根入口调用）
│   ├── plot_training_metrics.py# 从 trainer_state.json 绘制训练曲线
│   ├── clean_dpo_chosen.py     # 显式输入/输出的 chosen 清洗工具
│   ├── requirements_gpu.txt    # CUDA 训练依赖
│   └── utils/                  # 数据校验、模型解析、TRL DPO 实现
├── llm_judge/
│   ├── evaluate_paper_question.py # 结果评价正式适配器
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
├── compute_quality_metrics.py  # Question parse/qdr 实现
├── fixtures/                   # 合成输入，仅用于离线验收
├── paper_reproduction/
│   ├── manifest.yaml           # 四类公开图的唯一输入映射
│   ├── plot_data/              # 不含病例的论文聚合 CSV
│   └── scripts/render_figures.py # 绘图实现
├── tools/paper_figures/        # 兼容包装，不保存第二份数据
├── tools/reasoning_hallucination/ # reasoning 审计实现
├── examples/                   # 可公开的最小输入示例
├── tests/                      # 公共合约测试
├── guide/                      # 对外运行说明
├── scripts/validate_fixture.py # 从零离线验收
├── pytest.ini
├── requirements_eval.txt
└── .env.example
```

`data/`、`resources/`、`output/`、`results/` 和 `scratch/` 是运行时相对
路径。它们不承担版本库中的数据分发功能，默认全部被 `.gitignore` 忽略；
运行者可以通过命令行参数或环境变量把输入、模型和产物放到其他路径。

DPO 只负责偏好训练。模型实际生成、工作流编排、RAG/反思/修复和专家
评价不在本仓库实现。Judge 模块只接收已经生成的结果进行比较；Question
质量模块接收题目 Markdown；reasoning 模块读取外部病例和 reasoning 文本，
只输出分类计数和汇总。`data/`、`resources/`、`output/`、`results/` 和
`scratch/` 只作为被忽略的本地运行目录。
