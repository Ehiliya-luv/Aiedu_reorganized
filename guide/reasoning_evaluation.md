# Reasoning 评估

reasoning 评估不负责生成 reasoning、工作流编排或专家评价。它读取外部病例
与 reasoning Markdown，对每个模型汇总 supported、inferable、unsupported 和
contradicted 四类声明计数，并计算 case consistency 与 hallucination rate。

运行者把每个模型的 reasoning Markdown 放在：

```text
<package-root>/by_model/<model>/reasoning/case001.md
```

原始 case 放在独立目录后运行：

```bash
python reasoning_quality_main.py \
  --data-dir <package-root> \
  --cases-dir <case-dir> \
  --baseline baseline \
  --workers 4 \
  --output-dir <local-result-dir>
```

公开 fixture 使用 `--mock`，在线运行传入 `--judge-model` 并使用同一组远程
OpenAI-compatible 环境变量。输出只在本地保存逐病例 hash、分类计数和汇总，
不把病例或 reasoning 文本写回仓库。
