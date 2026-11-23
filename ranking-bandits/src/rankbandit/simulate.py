"""Benchmark ranking policies on a simulated catalogue.

    python -m rankbandit.simulate                 # stationary + popularity-shift scenarios, figures + JSON

Each policy is run on the same catalogues and click draws seeds; curves show the mean over seeds
with a 95 % confidence band.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rankbandit.environment import CascadeEnvironment  # noqa: E402
from rankbandit.policies import ALL_POLICIES  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
COLORS = ["#9AA5B1", "#E0A33B", "#5B8DB8", "#7A5BB8", "#167D9A", "#C0392B"]


def run(policy_cls, horizon: int, seed: int, n_items: int, k: int, shift_at: int | None = None) -> dict:
    env = CascadeEnvironment.catalogue(n_items, k, seed=seed)
    policy = policy_cls(n_items, k, np.random.default_rng(seed + 10_000))
    regret, clicks = np.zeros(horizon), np.zeros(horizon)
    for t in range(horizon):
        if shift_at is not None and t == shift_at:
            env.shift()
        ranking = policy.rank()
        click, regret[t] = env.step(ranking)
        clicks[t] = click is not None
        policy.update(ranking, click)
    return {"cum_regret": np.cumsum(regret), "clicks": clicks, "best_ctr": env.best_click_probability}


def benchmark(horizon: int, seeds: int, n_items: int, k: int, shift_at: int | None) -> dict:
    results = {}
    for cls in ALL_POLICIES:
        runs = [run(cls, horizon, s, n_items, k, shift_at) for s in range(seeds)]
        regrets = np.stack([r["cum_regret"] for r in runs])
        last = slice(-horizon // 10, None)  # last 10 % of rounds = converged behaviour
        ctr = np.array([r["clicks"][last].mean() for r in runs])
        best = np.array([r["best_ctr"] for r in runs])
        results[cls.name] = {
            "regret_mean": regrets.mean(0),
            "regret_ci": 1.96 * regrets.std(0) / np.sqrt(seeds),
            "final_regret": round(float(regrets[:, -1].mean()), 1),
            "final_ctr": round(float(ctr.mean()), 4),
            "share_of_optimal_ctr": round(float((ctr / best).mean()), 3),
        }
        print(
            f"{cls.name:32s} regret {results[cls.name]['final_regret']:8.1f}   CTR {results[cls.name]['final_ctr']:.3f}"
            f"   ({results[cls.name]['share_of_optimal_ctr']:.0%} of optimal)",
            flush=True,
        )
    return results


def plot(results: dict, title: str, path: Path, shift_at: int | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for color, (name, r) in zip(COLORS, results.items(), strict=False):
        x = np.arange(1, len(r["regret_mean"]) + 1)
        ax.plot(x, r["regret_mean"], color=color, lw=2 if "Thompson" in name else 1.5, label=name)
        ax.fill_between(
            x, r["regret_mean"] - r["regret_ci"], r["regret_mean"] + r["regret_ci"], color=color, alpha=0.15
        )
    if shift_at:
        ax.axvline(shift_at, color="#52616B", ls="--", lw=1)
        ax.annotate(
            "popularity shift",
            (shift_at, ax.get_ylim()[1] * 0.92),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8.5,
            color="#52616B",
        )
    ax.set(xlabel="rounds (users served)", ylabel="cumulative expected regret (lost clicks)", title=title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=20_000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--items", type=int, default=100)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    figures = ROOT / "reports" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    summary = {"config": vars(args)}
    for scenario, shift_at in [("stationary", None), ("popularity_shift", args.horizon // 2)]:
        print(f"\n== {scenario}")
        results = benchmark(args.horizon, args.seeds, args.items, args.k, shift_at)
        title = (
            "Stationary catalogue" if shift_at is None else "Popularity shift at mid-horizon"
        ) + f" ({args.items} items, top-{args.k})"
        plot(results, title, figures / f"regret_{scenario}.png", shift_at)
        summary[scenario] = {
            name: {k: v for k, v in r.items() if not k.startswith("regret_")} for name, r in results.items()
        }
    (ROOT / "reports" / "benchmark.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
