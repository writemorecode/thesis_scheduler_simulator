### Multidimensional Heterogeneous First-Fit Bin Packing

`main.py` contains a NumPy-based simulator for packing item types with multi-dimensional demands into heterogeneous bin types using the First-Fit heuristic. The solver accounts for per-bin purchase and opening costs and returns both the packing plan and the total cost.

#### Quick Start

```bash
uv run python main.py
```

The example in `main.py` demonstrates how to structure the input matrices:

- `C` (`K x M`): bin capacity columns (one column per bin type).
- `R` (`K x J`): item requirement columns (one column per item type).
- `L` (`J x 1`): counts of each item type.
- `purchase_costs`, `opening_costs`: one entry per bin type.
- `opened_bins` (optional): length-`M` vector with how many bins of each type are
  already open; their purchase + opening costs are charged immediately.

Import `first_fit_decreasing` to use the simulator inside your own scripts.

#### Evaluation Workflow

1. Run schedulers and write raw per-instance results:

```bash
uv run python src/eval.py --dataset dataset --schedulers ruin_recreate,ffd,bfd --output-dir eval_results --seed 123
```

2. Summarize per-scheduler performance:

```bash
uv run python src/eval_multi_summary.py --results-dir eval_results --output eval_summary_results.csv
```

3. Statistical analysis on a subset of schedulers:

```bash
uv run python src/analysis.py --results-dir eval_results --schedulers ffd,ruin_recreate --export-summary eval_log_ratio_summary.csv
```
