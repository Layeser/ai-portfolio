"""Forecast service: the 28-day forecast of any of the 30,490 item x store series.

    GET /forecast/{item_id}/{store_id}   e.g. /forecast/FOODS_3_090/CA_1
    GET /health

Forecasts are produced offline (batch, as in most retail planning systems) and served from memory.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from salesfc.data import FOLDS, HORIZON, ROOT, calendar, sales_wide


class DayForecast(BaseModel):
    date: str
    units: float


class Forecast(BaseModel):
    item_id: str
    store_id: str
    model: str
    last_28_days_mean: float
    forecast: list[DayForecast]


class Engine:
    def __init__(self, ids: list[str], forecasts: np.ndarray, dates: list[str], recent_mean: np.ndarray, model: str):
        self.index = {k: i for i, k in enumerate(ids)}
        self.forecasts, self.dates, self.recent_mean, self.model = forecasts, dates, recent_mean, model

    @classmethod
    def from_artifacts(cls, fold: str = "test") -> Engine:
        ids, sales = sales_wide()
        start = FOLDS[fold]
        forecasts = np.load(ROOT / "artifacts" / f"forecast_blend_{fold}.npy")
        dates = calendar()["date"].dt.strftime("%Y-%m-%d").to_list()[start - 1 : start - 1 + HORIZON]
        recent = sales[:, start - 1 - HORIZON : start - 1].mean(1)
        return cls(ids["id"].to_list(), forecasts, dates, recent, "0.8 LightGBM + 0.2 TiDE")

    def get(self, item_id: str, store_id: str) -> Forecast:
        i = self.index.get(f"{item_id}_{store_id}")
        if i is None:
            raise KeyError(item_id)
        return Forecast(item_id=item_id, store_id=store_id, model=self.model,
                        last_28_days_mean=round(float(self.recent_mean[i]), 3),
                        forecast=[DayForecast(date=d, units=round(float(v), 3))
                                  for d, v in zip(self.dates, self.forecasts[i], strict=True)])


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "engine", None) is None:
        app.state.engine = Engine.from_artifacts()
    yield


app = FastAPI(title="M5 sales forecasting", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "series": len(request.app.state.engine.index), "model": request.app.state.engine.model}


@app.get("/forecast/{item_id}/{store_id}", response_model=Forecast)
def forecast(item_id: str, store_id: str, request: Request) -> Forecast:
    try:
        return request.app.state.engine.get(item_id, store_id)
    except KeyError as exc:
        raise HTTPException(404, f"unknown series {item_id} / {store_id}") from exc
