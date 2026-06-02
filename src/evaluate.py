"""
Win rate evaluation using local reward model judge (default) or Claude Sonnet (final runs).

Default judge: Skywork/Skywork-Reward-V2-Llama-3.1-8B
  - #1 on RewardBench v2 (84.1), Seq. Classifier, 8B, free to run locally.
  - Scores each response independently; win = model_score > reference_score.

Final judge (--api_judge): claude-sonnet-4-6
  - Use only for the final 8 best-config runs that go in the paper.
  - Requires ANTHROPIC_API_KEY env var.
  - ~$0.30 per 200-sample eval run.
"""

import json
import os
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Optional

REWARD_MODEL_ID = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"

SONNET_JUDGE_PROMPT = """You are evaluating two AI assistant responses to a human request.

Human request:
{prompt}

Response A:
{response_a}

Response B:
{response_b}

Which response is better overall? Consider helpfulness, accuracy, and alignment with human values.
Reply with ONLY the letter A or B."""


# ---------------------------------------------------------------------------
# Local reward model judge (default — free, fast)
# ---------------------------------------------------------------------------

_reward_model = None
_reward_tokenizer = None


def _load_reward_model(device: str = "cuda"):
    global _reward_model, _reward_tokenizer
    if _reward_model is None:
        print(f"[eval] Loading reward model {REWARD_MODEL_ID}...")
        _reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_ID)
        _reward_model = AutoModelForSequenceClassification.from_pretrained(
            REWARD_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map=device,
            num_labels=1,
        )
        _reward_model.eval()
        print("[eval] Reward model loaded.")
    return _reward_model, _reward_tokenizer


def _score_responses(prompts: list[str], responses: list[str], device: str = "cuda") -> np.ndarray:
    """Score a list of (prompt, response) pairs. Returns array of scalar reward scores."""
    model, tokenizer = _load_reward_model(device)
    scores = []
    with torch.no_grad():
        for prompt, response in zip(prompts, responses):
            # Skywork reward model expects chat format
            messages = [
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": response},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=2048).to(model.device)
            score = model(**inputs).logits[0].item()
            scores.append(score)
    return np.array(scores)


def compute_win_rate_local(
    model_responses: list[str],
    reference_responses: list[str],
    prompts: list[str],
    n_samples: int = 500,
    seed: int = 42,
    results_path: Optional[str] = None,
    device: str = "cuda",
) -> dict:
    """
    Compute win rate using Skywork-Reward-V2-Llama-3.1-8B (local, free).

    Win = model reward score > reference reward score on same prompt.
    Uses n_samples random subset (500 default — more stable than GPT-4o's 200).
    """
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(prompts))
    idx = rng.choice(len(prompts), n, replace=False)

    sample_prompts   = [prompts[i]             for i in idx]
    sample_model     = [model_responses[i]     for i in idx]
    sample_reference = [reference_responses[i] for i in idx]

    print(f"[eval] Scoring {n} model responses...")
    model_scores = _score_responses(sample_prompts, sample_model, device)
    print(f"[eval] Scoring {n} reference responses...")
    ref_scores   = _score_responses(sample_prompts, sample_reference, device)

    wins   = int((model_scores > ref_scores).sum())
    losses = int((model_scores < ref_scores).sum())
    ties   = n - wins - losses
    win_rate = wins / n * 100

    result = {
        "judge": REWARD_MODEL_ID,
        "win_rate": round(win_rate, 2),
        "wins": wins, "losses": losses, "ties": ties,
        "n_judged": n,
        "model_score_mean":  round(float(model_scores.mean()), 4),
        "ref_score_mean":    round(float(ref_scores.mean()), 4),
    }

    if results_path:
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        records = [{"idx": int(i), "model_score": float(ms), "ref_score": float(rs),
                    "model_wins": bool(ms > rs)}
                   for i, ms, rs in zip(idx, model_scores, ref_scores)]
        with open(results_path, "w") as f:
            json.dump({"summary": result, "records": records}, f, indent=2)
        print(f"[eval] Results saved to {results_path}")

    print(f"[eval] Win rate: {win_rate:.1f}% ({wins}/{n}) | "
          f"model_mean={result['model_score_mean']:.3f} "
          f"ref_mean={result['ref_score_mean']:.3f}")
    return result


# ---------------------------------------------------------------------------
# Claude Sonnet judge (for final paper runs only)
# ---------------------------------------------------------------------------

def compute_win_rate_sonnet(
    model_responses: list[str],
    reference_responses: list[str],
    prompts: list[str],
    n_samples: int = 200,
    seed: int = 42,
    results_path: Optional[str] = None,
) -> dict:
    """
    Compute win rate using Claude Sonnet 4.6 as judge.
    Use ONLY for the final 8 best-config runs (paper results).
    Requires: ANTHROPIC_API_KEY environment variable.
    Cost: ~$0.30 per 200-sample run.
    """
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    rng = np.random.default_rng(seed)

    n = min(n_samples, len(prompts))
    indices = rng.choice(len(prompts), n, replace=False)

    wins, losses, records = 0, 0, []

    for i in indices:
        swap = rng.random() > 0.5
        a = reference_responses[i] if swap else model_responses[i]
        b = model_responses[i]     if swap else reference_responses[i]

        try:
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1,
                messages=[{"role": "user", "content": SONNET_JUDGE_PROMPT.format(
                    prompt=prompts[i], response_a=a, response_b=b
                )}],
            )
            verdict = msg.content[0].text.strip().upper()
        except Exception as e:
            print(f"[eval] Sonnet call failed for sample {i}: {e}")
            continue

        model_wins = (verdict == "B") if swap else (verdict == "A")
        wins += int(model_wins)
        losses += int(not model_wins)
        records.append({"idx": int(i), "swap": swap, "verdict": verdict,
                        "model_wins": model_wins})

    n_judged = wins + losses
    win_rate = wins / n_judged * 100 if n_judged > 0 else 0.0
    result = {"judge": "claude-sonnet-4-6",
              "win_rate": round(win_rate, 2),
              "wins": wins, "losses": losses, "n_judged": n_judged}

    if results_path:
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump({"summary": result, "records": records}, f, indent=2)
        print(f"[eval] Results saved to {results_path}")

    print(f"[eval] Sonnet win rate: {win_rate:.1f}% ({wins}/{n_judged})")
    return result


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def compute_win_rate(
    model_responses: list[str],
    reference_responses: list[str],
    prompts: list[str],
    n_samples: int = 500,
    seed: int = 42,
    results_path: Optional[str] = None,
    api_judge: bool = False,
    device: str = "cuda",
) -> dict:
    """
    Unified win rate function.
    api_judge=False (default): Skywork local reward model — use for all runs.
    api_judge=True: Claude Sonnet 4.6 — use only for final paper results.
    """
    if api_judge:
        return compute_win_rate_sonnet(
            model_responses, reference_responses, prompts,
            n_samples=min(n_samples, 200), seed=seed, results_path=results_path,
        )
    return compute_win_rate_local(
        model_responses, reference_responses, prompts,
        n_samples=n_samples, seed=seed, results_path=results_path, device=device,
    )
