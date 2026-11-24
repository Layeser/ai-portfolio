"""README figures: WRMSSE by model, accuracy by aggregation level, example forecasts."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from salesfc.data import calendar, sales_wide  # noqa: E402

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
R = json.loads((ROOT / "reports" / "test_results.json").read_text())["scores"]
NAMES = {"naive": "Naive", "seasonal_naive": "Seasonal naive", "tide": "TiDE (PyTorch)", "lightgbm": "LightGBM",
         "blend": "0.8 LightGBM + 0.2 TiDE"}
COLORS = {"naive": "#C9D1D9", "seasonal_naive": "#9AA5B1", "tide": "#E0A33B", "lightgbm": "#3E9BB3", "blend": "#167D9A"}

fig, ax = plt.subplots(figsize=(7, 3.6))
order = ["naive", "seasonal_naive", "tide", "lightgbm", "blend"]
bars = ax.barh([NAMES[k] for k in order], [R[k]["WRMSSE"] for k in order], color=[COLORS[k] for k in order])
ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
ax.axvline(0.520, color="#C0392B", ls="--", lw=1)
ax.text(0.525, -0.45, "M5 winner 0.520", color="#C0392B", fontsize=8)
ax.set(xlabel="WRMSSE on the M5 evaluation period (lower is better)", xlim=(0, 1.95),
       title="Official M5 metric, 30,490 series, 28-day horizon")
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(FIG / "wrmsse.png", dpi=160)
plt.close(fig)

levels = ["Total", "State", "Store", "Category", "Department", "State×cat.", "State×dept.", "Store×cat.",
          "Store×dept.", "Item", "Item×state", "Item×store"]
fig, ax = plt.subplots(figsize=(8, 3.6))
x = np.arange(12)
for k, off in (("seasonal_naive", -0.27), ("lightgbm", 0), ("blend", 0.27)):
    ax.bar(x + off, R[k]["per_level"], 0.27, color=COLORS[k], label=NAMES[k])
ax.set_xticks(x, levels, rotation=35, ha="right", fontsize=8)
ax.set(ylabel="RMSSE", title="Error by aggregation level: aggregates are easy, single items are noisy")
ax.legend(frameon=False, fontsize=8)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "levels.png", dpi=160)
plt.close(fig)

ids, sales = sales_wide()
blend = np.load(ROOT / "artifacts" / "forecast_blend_test.npy")
dates = calendar()["date"].to_numpy()
hist = slice(1941 - 84, 1941)
fut = slice(1941, 1969)
panels = [("All 30,490 series (total units)", np.ones(len(ids), bool)),
          ("Store CA_3, FOODS", (ids["store_id"] == "CA_3").to_numpy() & (ids["cat_id"] == "FOODS").to_numpy()),
          ("Item FOODS_3_090, store CA_3", (ids["id"] == "FOODS_3_090_CA_3").to_numpy())]
fig, axes = plt.subplots(3, 1, figsize=(9, 7.5))
for ax, (title, m) in zip(axes, panels, strict=True):
    ax.plot(dates[hist], sales[m][:, hist].sum(0), color="#9AA5B1", lw=1, label="history")
    ax.plot(dates[fut], sales[m][:, fut].sum(0), color="#17324D", lw=1.4, label="actual")
    ax.plot(dates[fut], blend[m].sum(0), color="#E0A33B", lw=1.8, label="forecast")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.25)
axes[0].legend(frameon=False, fontsize=8, ncol=3)
fig.suptitle("28-day forecasts on the M5 evaluation period (never seen during training or tuning)", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "forecasts.png", dpi=150)
plt.close(fig)
print("figures written")
