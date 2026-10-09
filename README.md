# Aiedu training and evaluation code

This repository contains the reusable code for the Aiedu DPO training and
evaluation components. It supports:

- DPO training with LoRA, QLoRA, full fine-tuning, checkpoint saving and resume;
- remote OpenAI-compatible LLM Judge evaluation of generated results;
- Question parsing and the internal `qdr` quality workflow;
- Question and reasoning evaluation summaries and paper-facing figure rendering.

Medical records, preference data, model weights, generated outputs, Judge
reasons, credentials and local experiment artifacts are supplied by the
operator and are never included in this repository.

The shortest public smoke test is:

```bash
python scripts/validate_fixture.py
```

For a real run, start with [guide/README.md](guide/README.md). The DPO entry
point remains `python dpo/main.py`; result evaluation uses
`python -m llm_judge.evaluate_result`; Question quality uses
`python compute_quality_metrics.py`.
