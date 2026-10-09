# 验证清单

## 从零离线验收

在一个新虚拟环境中执行：

```bash
python -m pip install -r requirements_eval.txt
python scripts/validate_fixture.py
python question_llm_judge_main.py \
  --data-dir fixtures/judge_outputs \
  --baseline baseline --tasks question --dry-run
python question_llm_judge_main.py --data-dir fixtures/judge_outputs --verify
python question_llm_judge_main.py --reproduce
python question_quality_main.py parse --group-dir demo=fixtures/question_inputs/demo --output-dir scratch/demo-quality
python question_quality_main.py qdr --parsed-dir scratch/demo-quality/parsed --output-dir scratch/demo-quality --mock
python reasoning_quality_main.py --mock --data-dir fixtures/reasoning_outputs --cases-dir fixtures/reasoning_cases --output-dir scratch/reasoning-quality
python paper_reproduction_main.py --verify
python paper_reproduction_main.py --only question-quality
```

应得到：DPO JSONL 校验通过、Question Markdown 能解析、QDR mock 产生摘要、
Judge dry-run 找到 baseline/candidate 和共同 case、reasoning fixture 产生汇总，
manifest 校验通过，且绘图目录生成三种格式。

## 自有数据运行验收

- [ ] DPO 先用少量自有数据和小模型完成一次 checkpoint 保存与 resume。
- [ ] 使用同一批外部模型输出做 Judge dry-run，再由受控 API 环境执行在线小样本。
- [ ] Question 先 parse，再进行小样本远程 QDR，确认缓存重跑不重复请求。
- [ ] reasoning 评估至少完成一个候选与基线的完整 case 交集，并核对四类声明计数。
- [ ] 汇总结果的有效票数、case 数、方向和论文图表数值与受控记录一致；允许
  远程 Judge 引入的小幅统计波动，但不能改变结论方向。
