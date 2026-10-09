# 数据格式与路径约定

## DPO 偏好数据

入口接收 UTF-8 编码的 JSONL，每行一个对象，至少包含三个非空字符串字段：

```json
{"prompt":"任务上下文","chosen":"偏好答案","rejected":"被拒答案","meta":{"sample_id":"可选","question_type":"可选"}}
```

`chosen` 和 `rejected` 不能完全相同。程序严格检查 JSON、字段和空值，训练
使用文件中的全部有效样本，不自动划分验证集，也不自动检查训练集与评估集
是否重叠。外部用户负责保证数据独立性。

## Question Markdown

质量入口保持现有题目格式：按 `## A1型题`、`## A2型题`、`## A3型题`、
`## A4型题` 或 `## B1型题` 分段；选项使用 `A.` 至 `E.`；答案使用
`答案：A`。文件可以采用以下任一目录布局：

```text
flat/                 case001.md
nested_question/      case1/question.md
per_stage/            case001/per_stage/questions.md
```

使用 `--group-dir NAME=PATH` 指定每个组。解析结果是 `parsed/*.jsonl`，
包含 case、题型、题干、选项、答案和结构问题字段。

## Judge 输入

结果评价入口优先读取项目原有的 Question 输出布局：

```text
<question-output-root>/<source-group>/<model-name>/case001/question.md
```

也接受公开 fixture 和外部用户更方便使用的标准化布局：

```text
<data-dir>/by_model/<model-name>/question/case001.md
<data-dir>/by_model/<model-name>/reasoning/case001.md
```

Reasoning 量化入口使用以下布局：项目原有结果可以直接使用
`<reasoning-output-root>/<source-group>/<model-name>/case001/reasoning.md`，
也接受平铺的 `<reasoning-output-root>/<model-name>/case001.md`；标准化布局则放在
`by_model/<model-name>/reasoning/case001.md`。每个模型和任务应使用相同 case
编号；程序也会把 `case1` 规范化为 `case001`。原始病例可另放在 `--cases-dir`，
文件名为 `case*.md`、`case*.txt` 或 `case*.json`。
Judge 若没有显式病例目录，会把基线输出第一行作为兼容回退；正式实验应显式
提供独立病例目录。

## 产物

运行产物应写入 `output/`、`results/` 或 `scratch/`，并由 `.gitignore`
忽略。建议每次运行使用独立子目录，保留命令、环境版本和摘要 CSV/JSON；
不要把逐病例输入、输出或 Judge 理由复制到公开仓库。
