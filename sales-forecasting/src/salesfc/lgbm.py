"""Global LightGBM with a Tweedie objective (the model family that dominated the M5 competition).

    python -m salesfc.lgbm --fold validation      # choose the number of rounds
    python -m salesfc.lgbm --fold test --rounds 1200
"""

from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np

from salesfc.data import FOLDS, HORIZON, ROOT, sales_wide
from salesfc.features import CATEGORICAL, build, feature_columns
from salesfc.metrics import WRMSSE

PARAMS = {
    "objective": "tweedie", "tweedie_variance_power": 1.1, "learning_rate": 0.03, "num_leaves": 127,
    "min_data_in_leaf": 100, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 0.1, "max_bin": 127, "verbose": -1, "num_threads": 10, "seed": 0,
}


def to_matrix(forecast_df, values: np.ndarray, start: int) -> np.ndarray:
    out = np.zeros((len(sales_wide()[0]), HORIZON), dtype=np.float32)
    out[forecast_df["series"].to_numpy(), forecast_df["d"].to_numpy() - start] = values
    return out


def run(fold: str, rounds: int, train_days: int, checkpoints=()) -> dict:
    start = FOLDS[fold]
    t0 = time.time()
    train, forecast = build(start, train_days)
    cols = feature_columns(train)
    data = lgb.Dataset(train[cols], train["sales"], categorical_feature=[c for c in CATEGORICAL if c in cols],
                       params={"max_bin": PARAMS["max_bin"], "verbose": -1}, free_raw_data=True).construct()
    print(f"{fold}: {len(train):,} training rows, {len(cols)} features, built in {time.time() - t0:.0f}s", flush=True)
    del train
    metric = WRMSSE(start)
    history = {}

    def callback(env):
        it = env.iteration + 1
        if it in checkpoints:
            pred = to_matrix(forecast, env.model.predict(forecast[cols], num_iteration=it), start)
            history[it] = round(metric(pred), 4)
            print(f"  round {it}: WRMSSE {history[it]}", flush=True)

    model = lgb.train(PARAMS, data, num_boost_round=rounds, callbacks=[callback])
    pred = to_matrix(forecast, model.predict(forecast[cols]), start)
    score, per_level = metric(pred, per_level=True)
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / f"forecast_lgbm_{fold}.npy", pred)
    model.save_model(str(out_dir / f"lgbm_{fold}.txt"))
    importance = dict(sorted(zip(cols, model.feature_importance("gain").round().tolist(), strict=True),
                             key=lambda kv: -kv[1]))
    return {"fold": fold, "WRMSSE": round(score, 4), "per_level": [round(s, 4) for s in per_level],
            "rounds": rounds, "train_days": train_days, "history": history, "minutes": round((time.time() - t0) / 60, 1),
            "importance_gain": importance}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", choices=list(FOLDS), required=True)
    parser.add_argument("--rounds", type=int, default=800)
    parser.add_argument("--train-days", type=int, default=548)
    args = parser.parse_args()
    checkpoints = (100, 200, 300, 400, 500, 600, 800, 1000) if args.fold == "validation" else ()
    result = run(args.fold, args.rounds, args.train_days, checkpoints)
    print(json.dumps({k: v for k, v in result.items() if k != "importance_gain"}), flush=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / f"lgbm_{args.fold}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
