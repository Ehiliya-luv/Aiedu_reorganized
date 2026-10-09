# Paper reproduction inputs and renderer

This directory contains only the code and aggregate numeric inputs needed to
rebuild the current paper-facing plots. It deliberately excludes cases,
model outputs, raw Judge records and timing logs.

Run from the repository root:

```bash
python paper_reproduction_main.py --manifest paper_reproduction/manifest.yaml
```

The generated files go to the ignored `results/paper_figures/` directory. The
CSV files in `plot_data/` are the single tracked source for the renderer; a
future release can remove an optional figure by removing its source CSV and
manifest entry.
