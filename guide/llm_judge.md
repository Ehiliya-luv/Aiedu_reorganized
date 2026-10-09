# 结果评价 LLM Judge

正式入口默认评价外部流程生成的 Question 结果。它采用一个基线模型和多个
候选模型的 Anchor 比较，每次比较返回 A/B 胜负和理由，支持重复投票、bootstrap
区间、checkpoint/resume 和图表。Judge 可连接任意提供 OpenAI-compatible Chat
Completions 的远程服务。reasoning 的量化一致性由
`reasoning_quality_main.py` 单独负责。

## 输入和 dry-run

```bash
python question_llm_judge_main.py \
  --data-dir <package-root> \
  --cases-dir <case-dir> \
  --baseline baseline \
  --tasks question \
  --dry-run
```

`<package-root>` 可以是标准化的 `by_model/<model>/<task>/case*.md` 根目录，
也可以是项目原有的 Question 输出根目录；后者按
`<source-group>/<model>/case*/question.md` 读取。模型目录名就是 `--baseline`
和候选模型名；例如原有 Baichuan 输出应传 `--baseline baichuan-m2`。程序也会
把 `case1` 规范化为 `case001`。dry-run 只检查目录和 case 数量，不调用远程服务。

需要复用通用比较器时可以显式传入其他 task；这不改变 reasoning 量化入口的
分类和汇总职责。

## 在线运行

先在进程环境或未跟踪的 `.env` 中设置：

```text
AIEDU_JUDGE_API_BASE=<your-openai-compatible-endpoint>
AIEDU_JUDGE_API_KEY=<your-key>
AIEDU_JUDGE_MODEL=<your-judge-model>
```

再运行：

```bash
python question_llm_judge_main.py \
  --data-dir <package-root> \
  --cases-dir <case-dir> \
  --baseline baseline \
  --tasks question \
  --prompt-variant prompt_v3 \
  --repeats 3 --workers 4 \
  --output-dir <local-result-dir>
```

中断后对同一输出目录加 `--resume` 可继续 checkpoint。远程 Judge 的随机性、
服务版本和重试顺序可能造成小幅数值变化；应比较有效票数、方向和置信区间，
不要把单次结果当作字节级确定性输出。

`--verify` 只验证模型输出目录和 case 覆盖，`--reproduce` 只验证冻结的四类
论文图输入，不会调用远程服务。`--round` 和 `--run-id` 仅用于隔离本地运行产物。

本仓库不包含模型输出、病例文本和原始 Judge 理由。运行者应把这些输入和
产物保存在受控目录，并只向外部交付必要的聚合统计。
