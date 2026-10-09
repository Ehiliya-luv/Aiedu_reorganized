# 结果评价 LLM Judge

Judge 只评价外部流程已经生成的 Question 或 reasoning 结果。它采用一个
基线模型和多个候选模型的 Anchor 比较，每次比较返回 A/B 胜负和理由，支持
重复投票、bootstrap 区间、checkpoint/resume 和图表。Judge 可连接任意提供
OpenAI-compatible Chat Completions 的远程服务。

## 输入和 dry-run

```bash
python -m llm_judge.evaluate_result \
  --data-dir <package-root> \
  --cases-dir <case-dir> \
  --baseline baseline \
  --tasks question,reasoning \
  --dry-run
```

`<package-root>` 必须含 `by_model/<model>/<task>/case*.md`。dry-run 只检查
目录和 case 数量，不调用远程服务。

## 在线运行

先在进程环境或未跟踪的 `.env` 中设置：

```text
AIEDU_JUDGE_API_BASE=<your-openai-compatible-endpoint>
AIEDU_JUDGE_API_KEY=<your-key>
AIEDU_JUDGE_MODEL=<your-judge-model>
```

再运行：

```bash
python -m llm_judge.evaluate_result \
  --data-dir <package-root> \
  --cases-dir <case-dir> \
  --baseline baseline \
  --tasks question,reasoning \
  --repeats 3 --workers 4 \
  --output-dir <local-result-dir>
```

中断后对同一输出目录加 `--resume` 可继续 checkpoint。远程 Judge 的随机性、
服务版本和重试顺序可能造成小幅数值变化；应比较有效票数、方向和置信区间，
不要把单次结果当作字节级确定性输出。

本仓库不包含模型输出、病例文本和原始 Judge 理由。运行者应把这些输入和
产物保存在受控目录，并只向外部交付必要的聚合统计。
