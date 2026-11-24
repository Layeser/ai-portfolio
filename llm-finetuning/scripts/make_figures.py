"""Figures for the README: base vs fine-tuned scores, recall per field, training curve."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REP, FIG = ROOT / "reports", ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
base = json.loads((REP / "eval_base_test.json").read_text())
lora = json.loads((REP / "eval_lora_test.json").read_text())

metrics = [("field_f1", "field F1"), ("valid_json", "valid JSON"), ("doc_accuracy", "receipts fully correct")]
fig, ax = plt.subplots(figsize=(7, 3.4))
x = range(len(metrics))
for off, (res, label, color) in zip((-0.2, 0.2), ((base, "Qwen2.5-1.5B zero-shot", "#9aa5ad"),
                                                   (lora, "+ LoRA fine-tuning", "#167d9a")), strict=True):
    vals = [res[k] for k, _ in metrics]
    bars = ax.bar([i + off for i in x], vals, 0.4, label=label, color=color)
    ax.bar_label(bars, [f"{v:.2f}" for v in vals], fontsize=9)
ax.set_xticks(list(x), [n for _, n in metrics])
ax.set_ylim(0, 1.1)
ax.set_title("CORD v2 test set (100 receipts)")
ax.legend(frameon=False, loc="upper right", fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "scores.png", dpi=150)

fields = list(lora["recall_by_field"])[:12]  # the 12 most frequent fields (metrics orders them by count)
fig, ax = plt.subplots(figsize=(7, 4))
y = range(len(fields))
ax.barh([i + 0.2 for i in y], [base["recall_by_field"].get(f, 0) for f in fields], 0.4, color="#9aa5ad", label="zero-shot")
ax.barh([i - 0.2 for i in y], [lora["recall_by_field"][f] for f in fields], 0.4, color="#167d9a", label="LoRA")
ax.set_yticks(list(y), fields)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("recall per field (test)")
ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "fields.png", dpi=150)
print("figures ->", FIG)
