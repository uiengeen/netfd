# Experiment scripts

Each script is a thin orchestration layer that loads a YAML config from
`../configs/experiments/`, calls into the `netfd` library, and writes
outputs (figures, JSON summary, npz blob) to `outputs/<experiment_name>/`.

| File | YAML | Outputs |
|---|---|---|
| `01_topology_illustration.py` | `01_topology_illustration.yaml` | benchmark topology (plain/annotated), adjacency heatmap, three fault-class schematics |
| `02_blindness.py` | `02_blindness.yaml` | observation-sweep confusion matrices |
| `03_proximity.py` | `03_proximity.yaml` | symmetric-edge confusion matrices with red-boxed pairs |
| `04_single_shot_baseline.py` | `04_single_shot_baseline.yaml` | mixed-hypothesis single-shot confusion matrices |
| `05_ppo_node_fault.py` | `05_ppo_node_fault.yaml` | PPO vs single-shot, training curves, action usage |
| `06_ppo_edge_fault.py` | `06_ppo_edge_fault.yaml` | same as 05 for edge faults |
| `07_ppo_mixed_fault.py` | `07_ppo_mixed_fault.yaml` | same as 05 for mixed faults |

## Common CLI

All scripts share the same interface:

```
python experiments/<script.py> [--config PATH] [--outdir DIR] [--override KEY=VAL]
```

- `--config` / `-c`: alternative YAML path (defaults to the matching file under `configs/experiments/`)
- `--outdir` / `-o`: override the YAML's `outdir`
- `--override` / `-O`: set a scalar by dotted path; can be repeated

Examples:

```bash
# Use the default config
python experiments/05_ppo_node_fault.py

# Shorter PPO budget for a smoke test
python experiments/05_ppo_node_fault.py \
    --override ppo.total_timesteps=10000 \
    --override noise.n_mc_eval=20

# Different seed, different output dir
python experiments/05_ppo_node_fault.py \
    --override ppo.seed=42 \
    --outdir runs/exp05_seed42
```

## Shared helpers

`_common.py` — CLI parser, banner, path utilities.
`_ppo_pipeline.py` — the full PPO pipeline (build hypotheses, ν-gap
matrix, pair scenarios, single-shot baseline, PPO train/eval, plots,
summaries). Scripts 05/06/07 are ~50-line wrappers around it; the only
difference between them is the hypothesis kind, which is dispatched
inside `_ppo_pipeline.build_hypotheses_from_yaml`.

## Outputs

All experiments write to `outputs/<exp_name>/` (gitignored). Typical
contents:

- `*.png` — figures
- `summary.json` — human-readable per-SNR accuracy and configuration
- `results.npz` — raw arrays (confusion matrices, ν-gap matrix, PPO
  training log) for downstream analysis
