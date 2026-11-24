"""Score every model on the M5 evaluation period (d_1942-1969) and write reports/test_results.json.

The blend weight (0.8 LightGBM + 0.2 TiDE) was chosen on the validation fold, never on these days.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from salesfc.data import sales_wide  # noqa: E402
from salesfc.metrics import WRMSSE  # noqa: E402

WEIGHT = 0.8

metric = WRMSSE(1942)
sales = sales_wide()[1]
lgbm = np.load(ROOT / "artifacts" / "forecast_lgbm_test.npy")
tide = np.load(ROOT / "artifacts" / "forecast_tide_test.npy")
blend = WEIGHT * lgbm + (1 - WEIGHT) * tide
np.save(ROOT / "artifacts" / "forecast_blend_test.npy", blend)
forecasts = {"naive": np.repeat(sales[:, 1940:1941], 28, 1), "seasonal_naive": np.tile(sales[:, 1934:1941], 4),
             "tide": tide, "lightgbm": lgbm, "blend": blend}
scores = {}
for name, f in forecasts.items():
    total, levels = metric(f, per_level=True)
    scores[name] = {"WRMSSE": round(total, 4), "per_level": [round(x, 4) for x in levels]}
    print(f"{name:15s} {total:.4f}")
(ROOT / "reports" / "test_results.json").write_text(json.dumps(
    {"period": "d_1942-d_1969 (M5 evaluation, private leaderboard)", "blend_weight_lightgbm": WEIGHT,
     "blend_weight_chosen_on": "validation fold d_1914-d_1941", "scores": scores}, indent=2))
