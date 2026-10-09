# Result evaluation LLM Judge

The reusable Anchor and pairwise implementations stay in this directory. Use
the root-level result entry point for a normal run:

```bash
python question_llm_judge_main.py --data-dir <private-by-model> --cases-dir <private-cases> --verify
python question_llm_judge_main.py --data-dir <private-by-model> --cases-dir <private-cases> --tasks question --judge-model <model>
```

`--dry-run`, `--verify`, and `--reproduce` make no remote calls. Online runs
read the provider URL, key and model from process environment variables or the
ignored local `.env`. Checkpoints and generated Judge reasons belong in the
ignored output directory and are never part of the public repository.
