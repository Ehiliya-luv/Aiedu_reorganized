# Reasoning 评估

reasoning 评估复用结果 Judge 的 Anchor/Pairwise 接口，不负责生成 reasoning，
也不负责工作流编排或专家评价。运行者把每个模型的 reasoning Markdown 放在：

```text
<package-root>/by_model/<model>/reasoning/case001.md
```

原始 case 放在独立目录后运行：

```bash
python -m llm_judge.evaluate_question_reasoning \
  --data-dir <package-root> \
  --cases-dir <case-dir> \
  --baseline baseline \
  --tasks reasoning \
  --repeats 3 --workers 4 \
  --output-dir <local-result-dir>
```

评估输出包含每个候选相对基线的胜负、有效票数、区间、checkpoint 和可选图表。
如果需要论文中的 reasoning 一致性汇总，应使用运行者受控环境中生成的摘要，
再按 [paper_reproduction.md](paper_reproduction.md) 的聚合格式绘图；公开交付
不包含逐病例 reasoning 或 Judge 理由。
