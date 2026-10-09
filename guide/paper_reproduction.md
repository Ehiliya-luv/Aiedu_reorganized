# 论文图复现

`paper_reproduction/plot_data/` 保存当前论文图所需的最小聚合 CSV。它们只有
模型级或任务级数值，不包含病例、模型输出、Judge 理由、密钥或绝对路径。

安装评估依赖后，在仓库根目录运行：

```bash
python paper_reproduction_main.py --manifest paper_reproduction/manifest.yaml
```

图会写到被忽略的 `results/paper_figures/`，每张图同时输出 PDF、PNG、SVG。
也可以只重建一张：

```bash
python paper_reproduction_main.py --only question-quality
python paper_reproduction_main.py --only question-preference
python paper_reproduction_main.py --only reasoning-consistency
python paper_reproduction_main.py --only workflow-time
```

Question 图严格读取 `question_quality.csv` 中的三列正式指标；preference 图
读取远程 Judge 的聚合胜负和区间；reasoning 图读取 case consistency 汇总；
workflow-time 图读取任务级时间汇总。绘图脚本不尝试从公开仓库恢复真实输入，
也不会调用远程服务。

先用 `python paper_reproduction_main.py --verify` 检查 manifest、表头和四个
聚合输入。

如果后续公开版本不包含时间汇总，只需删除该 CSV 并不运行
`--only workflow-time`；其余图不依赖该文件。论文最终像素级验收应在含有论文
参考 PDF 的受控环境中完成，公开仓库只保证数值输入和绘图逻辑可复现。
