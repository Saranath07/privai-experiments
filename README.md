# privai-experiments

Angle A: Scaling laws of privacy-utility tradeoffs in differentially private preference alignment.
Implements SquareχPO (arXiv:2505.21395) at 4B/12B scale on real human preference data.

## GPU Machine Setup (run once)

```bash
git pull
pip install -r requirements.txt
huggingface-cli login
wandb login
export OPENAI_API_KEY=sk-...
```

## Workflow

### Step 1 — Build all noisy datasets (run once)
```bash
python scripts/prepare_datasets.py --output_dir data/
```

### Step 2 — Sanity check (3 quick runs before full grid)
```bash
python scripts/launch_grid.py --phase sanity
```
Check: loss decreases, win rate of clean DPO > 50%, LTC worse than CTL at ε=1.0.

### Step 3 — Full 4B ablation grid
```bash
python scripts/launch_grid.py --phase ablation
```

### Step 4 — 12B flagship runs
```bash
python scripts/launch_grid.py --phase flagship
```

### Step 5 — Dataset size scaling
```bash
python scripts/launch_grid.py --phase nscaling
```

## Single run
```bash
python scripts/train.py --eps 1.0 --alpha 0.0 --mode CTL --model_size 4b --wandb
```

## Dry run (print commands without executing)
```bash
python scripts/launch_grid.py --phase ablation --dry_run
```

## Key numbers to check after sanity runs
- c(ε=1.0) = 2.164 — verify with `python -c "from src.debiased_dpo import c_eps; print(c_eps(1.0))"`
- RR flip rate at ε=1.0 ≈ 26.9% — printed by prepare_datasets.py
- Clean DPO win rate > 50% vs SFT reference
