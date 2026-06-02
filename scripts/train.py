"""
Main training script for Angle A experiments.

Example usage (on GPU machine):
    # Clean baseline
    python scripts/train.py --eps inf --alpha 0.0 --mode CTL --model_size 4b

    # SquareχPO CTL, ε=1.0
    python scripts/train.py --eps 1.0 --alpha 0.0 --mode CTL --model_size 4b

    # SquareχPO LTC with corruption
    python scripts/train.py --eps 2.0 --alpha 0.1 --mode LTC --model_size 4b

    # 12B flagship
    python scripts/train.py --eps 1.0 --alpha 0.0 --mode CTL --model_size 12b
"""

import argparse
import json
import os
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig

from debiased_dpo import DebiasedSquareDPOTrainer
from training_config import get_training_config

MODEL_IDS = {
    "4b":  "google/gemma-3-4b-it",
    "12b": "google/gemma-3-12b-it",
}

LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


def run_name(eps, alpha, mode, model_size, seed):
    eps_str = "inf" if eps == float("inf") else str(eps)
    return f"gemma{model_size}_eps{eps_str}_alpha{alpha}_{mode}_seed{seed}"


def main(args):
    eps = float("inf") if args.eps == "inf" else float(args.eps)
    run = run_name(eps, args.alpha, args.mode, args.model_size, args.seed)
    print(f"\n{'='*60}\nRun: {run}\n{'='*60}")

    # Load dataset — always use the full saved dataset, slice after loading
    eps_str = "inf" if eps == float("inf") else str(eps)

    # Find the saved dataset for this (eps, alpha, mode) combo regardless of n suffix
    data_dir = Path(args.data_dir)
    prefix = f"hh_rlhf_eps{eps_str}_alpha{args.alpha}_{args.mode}_"
    matches = sorted(data_dir.glob(f"{prefix}*_seed{args.seed}"))

    if not matches:
        raise FileNotFoundError(
            f"No dataset found in {data_dir} matching '{prefix}*_seed{args.seed}'.\n"
            f"Run: python scripts/prepare_datasets.py --output_dir {args.data_dir}"
        )
    data_path = matches[-1]  # take the largest (full) dataset
    print(f"Loading dataset from {data_path}")

    dataset = load_from_disk(str(data_path))
    if args.n_samples:
        dataset = dataset.select(range(min(args.n_samples, len(dataset))))
    split = dataset.train_test_split(test_size=0.05, seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    # Load model + tokenizer
    model_id = MODEL_IDS[args.model_size]
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto"
    )
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    # Reference model (frozen base, no LoRA)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto"
    )

    # Training config
    train_cfg = get_training_config(eps, model_size=args.model_size)
    output_dir = Path(args.output_dir) / run

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        run_name=run,
        report_to="wandb" if args.wandb else "none",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        seed=args.seed,
        max_length=1024,
        **train_cfg,
    )

    trainer = DebiasedSquareDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        max_prompt_length=512,  # trainer arg, not config
        eps=eps,
        R_max=args.R_max,
    )

    print("Training...")
    trainer.train()

    # Save results summary
    results = {
        "run": run,
        "eps": str(eps),
        "alpha": args.alpha,
        "mode": args.mode,
        "model_size": args.model_size,
        "seed": args.seed,
        "train_samples": len(train_ds),
        "output_dir": str(output_dir),
    }
    results_path = Path(args.results_dir) / f"{run}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps",        default="inf", help="Privacy budget ε (or 'inf')")
    parser.add_argument("--alpha",      type=float, default=0.0, help="Corruption rate α")
    parser.add_argument("--mode",       default="CTL", choices=["CTL", "LTC", "clean"])
    parser.add_argument("--model_size", default="4b",  choices=["4b", "12b"])
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--n_samples",  type=int, default=None, help="Dataset size cap")
    parser.add_argument("--R_max",      type=float, default=1.0)
    parser.add_argument("--data_dir",   default="data")
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--results_dir",default="results")
    parser.add_argument("--wandb",      action="store_true")
    parser.add_argument("--gpu",        default=None, help="CUDA device id(s), e.g. '1' or '1,2'")
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)
