# 使用指南

按以下顺序阅读即可完成一次从数据检查到结果复现的运行：

| 文档 | 负责的问题 |
| --- | --- |
| [project_map.md](project_map.md) | 目录、模块边界和运行产物在哪里 |
| [environment.md](environment.md) | CUDA DPO 环境和评估环境如何安装 |
| [data_format.md](data_format.md) | DPO、Question、reasoning 和 Judge 输入的格式 |
| [dpo_training.md](dpo_training.md) | 如何训练、续训和保存模型 |
| [llm_judge.md](llm_judge.md) | 如何用远程 LLM Judge 评价模型结果 |
| [question_quality.md](question_quality.md) | 如何解析题目并运行 Question 质量审查 |
| [reasoning_evaluation.md](reasoning_evaluation.md) | 如何评价 reasoning 输出 |
| [paper_reproduction.md](paper_reproduction.md) | 如何使用聚合数据重建论文图 |
| [troubleshooting.md](troubleshooting.md) | 常见失败和定位顺序 |
| [validation_checklist.md](validation_checklist.md) | 交付前和运行前的可执行检查清单 |

指南只描述公开仓库中的接口。真实病历、训练数据、模型、模型输出、
Judge 原始理由和本地运行记录由运行者自行保管。
