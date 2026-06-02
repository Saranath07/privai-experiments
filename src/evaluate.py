"""
GPT-4o-as-judge evaluation for win rate computation.

Matches the evaluation protocol in SquareχPO (Table 1).
Randomizes A/B ordering per sample to avoid position bias.
"""

import os
import json
import numpy as np
from pathlib import Path
from openai import OpenAI

JUDGE_PROMPT = """You are evaluating two AI assistant responses to a human request.

Human request:
{prompt}

Response A:
{response_a}

Response B:
{response_b}

Which response is better overall? Consider helpfulness, accuracy, and alignment with human values.
Reply with ONLY the letter A or B."""


def compute_win_rate(
    model_responses: list[str],
    reference_responses: list[str],
    prompts: list[str],
    n_samples: int = 200,
    seed: int = 42,
    results_path: str = None,
) -> dict:
    """
    Compute win rate of model vs. reference using GPT-4o judge.

    Args:
        model_responses: Responses from the model being evaluated.
        reference_responses: Responses from the reference (e.g., SFT baseline).
        prompts: Input prompts corresponding to each response pair.
        n_samples: Number of random pairs to judge (controls API cost).
        seed: Random seed for reproducibility.
        results_path: If set, save per-sample verdicts as JSON.

    Returns:
        dict with win_rate (%), wins, losses, n_judged.
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    rng = np.random.default_rng(seed)

    n = min(n_samples, len(prompts))
    indices = rng.choice(len(prompts), n, replace=False)

    wins, losses, records = 0, 0, []

    for i in indices:
        swap = rng.random() > 0.5
        a = reference_responses[i] if swap else model_responses[i]
        b = model_responses[i] if swap else reference_responses[i]

        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                    prompt=prompts[i], response_a=a, response_b=b
                )}],
                max_tokens=1,
                temperature=0,
            )
            verdict = resp.choices[0].message.content.strip().upper()
        except Exception as e:
            print(f"[eval] GPT-4o call failed for sample {i}: {e}")
            continue

        # "A wins" means reference wins if swapped, model wins otherwise
        model_wins = (verdict == "B") if swap else (verdict == "A")

        if model_wins:
            wins += 1
        else:
            losses += 1

        records.append({"idx": int(i), "swap": swap, "verdict": verdict,
                         "model_wins": model_wins})

    n_judged = wins + losses
    win_rate = wins / n_judged * 100 if n_judged > 0 else 0.0

    result = {"win_rate": round(win_rate, 2), "wins": wins,
              "losses": losses, "n_judged": n_judged}

    if results_path:
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump({"summary": result, "records": records}, f, indent=2)
        print(f"[eval] Results saved to {results_path}")

    print(f"[eval] Win rate: {win_rate:.1f}% ({wins}/{n_judged})")
    return result
