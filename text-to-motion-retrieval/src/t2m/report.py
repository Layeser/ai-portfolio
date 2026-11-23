"""Figure: training loss vs held-out retrieval, from artifacts/metrics.json.

    python -m t2m.report
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from t2m.data import ROOT  # noqa: E402
from t2m.train import ARTIFACTS  # noqa: E402

LEAKED_VAL_R1 = 0.933  # original notebook, on "val" folders whose motions were in the training set


def main() -> None:
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    history = metrics["history"]
    evals = [h for h in history if "holdout_challenge" in h]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot([h["epoch"] for h in history], [h["loss"] for h in history], color="#9AA5B1", lw=2, label="train loss")
    ax.set(xlabel="epoch", ylabel="contrastive train loss")
    ax2 = ax.twinx()
    ax2.plot([h["epoch"] for h in evals], [h["holdout_challenge"] for h in evals], "o-", color="#167D9A", lw=2,
             label="held-out weighted recall (32 candidates)")
    r1 = metrics["holdout"]["challenge_32"]["R@1"]
    ax2.axhline(r1, color="#167D9A", ls=":", lw=1.2, label=f"held-out R@1, best checkpoint ({r1:.3f})")
    ax2.axhline(LEAKED_VAL_R1, color="#C0392B", ls="--", lw=1.2, label="R@1 reported on leaked val (0.933)")
    ax2.set(ylabel="held-out score", ylim=(0, 1))
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax2.legend(lines, labels, frameon=False, fontsize=8, loc="center right")
    ax.set_title("Held-out score plateaus at epoch 20 while train loss keeps falling")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = ROOT / "reports" / "figures" / "training_vs_holdout.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print("wrote", out)


if __name__ == "__main__":
    main()
