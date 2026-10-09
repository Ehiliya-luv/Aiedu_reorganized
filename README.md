# Aiedu training and evaluation code

This repository contains the reusable DPO training, result-evaluation LLM
Judge, Question quality, reasoning consistency, and paper-figure code used by
the project. The five public commands are:

```text
python dpo_main.py
python question_llm_judge_main.py
python question_quality_main.py
python reasoning_quality_main.py
python paper_reproduction_main.py
```

Run any command with `--help`, then read [guide/README.md](guide/README.md) for
the corresponding input format and examples. The implementation continues to
use the original internal modules and data formats; the root files are thin
command wrappers.

The repository contains no medical records, preference data, model weights,
generated model outputs, raw Judge reasons, credentials, or private runtime
logs. Operators provide those paths and configuration locally. Synthetic
fixtures are included only for offline contract tests:

```bash
python scripts/validate_fixture.py
```
