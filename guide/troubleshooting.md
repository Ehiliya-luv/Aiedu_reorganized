# 排查顺序

## DPO 启动失败

1. 先检查 Python 版本和 `dpo/requirements_gpu.txt` 是否安装在当前环境。
2. 用 `python -c "import torch; print(torch.cuda.is_available())"` 检查 CUDA。
3. 单独运行 JSONL 校验，确认每行都有非空 `prompt`、`chosen`、`rejected`，且
   两个答案不同。
4. 确认模型路径存在或 repo id 可由当前网络和权限下载；不要把模型目录复制
   到 Git。
5. 若是显存不足，先减小 batch、max length 或启用 QLoRA，再检查 checkpoint。

## Judge 失败

1. 先运行 `question_llm_judge_main.py --dry-run`，确认模型目录、任务和 case 编号交集不为空。
2. 检查 `AIEDU_JUDGE_API_BASE`、`AIEDU_JUDGE_API_KEY`、`AIEDU_JUDGE_MODEL`。
3. 用服务提供方的最小 Chat Completions 请求验证地址、权限和模型名。
4. 查看输出目录中的 checkpoint；网络超时和限流可通过 `--resume` 继续。
5. 如果只有少数 case 失败，检查对应输入是否为空、编码是否为 UTF-8，以及远程
   服务是否返回了可解析的 JSON。

## Question 解析或 QDR 失败

1. 先只运行 `parse --limit 1`，查看生成的 JSONL。
2. 检查标题、选项标签和答案行是否符合 [data_format.md](data_format.md)。
3. 确认每个组的 case 编号不重复，且没有把模型输出目录误当作题目目录。
4. 在线 QDR 若出现票数不齐，先查看 solver 模型可用性和服务限额，再继续
   同一输出目录；不要手工修改缓存中的判断。

## 绘图失败

确认 `requirements_eval.txt` 已安装，先运行
`python paper_reproduction_main.py --verify`，并确认 CSV 表头没有被改动。绘图
只依赖聚合 CSV，不需要真实数据或模型。
