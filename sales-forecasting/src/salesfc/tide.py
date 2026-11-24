"""TiDE (Das et al., "Long-term Forecasting with TiDE: Time-series Dense Encoder", TMLR 2023) in PyTorch,
trained as one global model over the 30,490 M5 series.

Inputs per window: 112 days of sales history (divided by the series' recent mean), dynamic covariates
known in advance for history + horizon (calendar, events, SNAP, relative price), static embeddings
(item, department, category, store, state). Output: 28 daily forecasts. Loss: Tweedie deviance
(power 1.1) on each series' scaled target (as in DeepAR), which suits intermittent counts; the model
predicts a log-mean that is multiplied back by the series scale.

    python -m salesfc.tide --fold validation
    python -m salesfc.tide --fold test --epochs <best epoch on validation>
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

from salesfc.data import FOLDS, HORIZON, ID_COLS, ROOT, calendar, first_sale_day, price_matrix, sales_wide
from salesfc.metrics import WRMSSE

LOOKBACK = 112


def covariates() -> tuple[np.ndarray, np.ndarray]:
    """(calendar features (days, C), relative price (series, days)) for all 1969 days."""
    cal = calendar()
    wday = np.eye(7, dtype=np.float32)[cal["wday"].to_numpy() - 1]
    month = cal["month"].to_numpy()
    mday = cal["date"].dt.day.to_numpy()
    seasonal = np.stack([np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
                         np.sin(2 * np.pi * mday / 31), np.cos(2 * np.pi * mday / 31)], 1).astype(np.float32)
    events = np.eye(5, dtype=np.float32)[cal["event_type_1"].astype("category").cat.codes.to_numpy() + 1]
    snap = cal[["snap_CA", "snap_TX", "snap_WI"]].to_numpy(np.float32)
    price = price_matrix()
    rel = price / np.nanmean(price, axis=1, keepdims=True)
    return np.concatenate([wday, seasonal, events, snap], 1), np.nan_to_num(rel, nan=1.0).astype(np.float32)


class ResBlock(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_out), nn.Dropout(dropout))
        self.skip = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x):
        return self.norm(self.net(x) + self.skip(x))


class TiDE(nn.Module):
    def __init__(self, n_cov: int, cardinalities: list[int], hidden: int = 512, proj: int = 8, dec: int = 16,
                 layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(n, min(32, (n + 1) // 2)) for n in cardinalities])
        d_static = sum(e.embedding_dim for e in self.embeds)
        self.cov_proj = ResBlock(n_cov, hidden // 4, proj, dropout)
        d_in = LOOKBACK + (LOOKBACK + HORIZON) * proj + d_static
        enc = [ResBlock(d_in, hidden, hidden, dropout)] + [ResBlock(hidden, hidden, hidden, dropout)
                                                            for _ in range(layers - 1)]
        dec_layers = [ResBlock(hidden, hidden, hidden, dropout) for _ in range(layers - 1)]
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec_layers, ResBlock(hidden, hidden, HORIZON * dec, dropout))
        self.temporal = ResBlock(dec + proj, hidden // 8, 1, dropout)
        self.lookback_skip = nn.Linear(LOOKBACK, HORIZON)
        self.dec = dec

    def forward(self, y, cov, static):
        """y (B, L) scaled history, cov (B, L+H, C), static (B, n_static) -> log-mean (B, H) on the scaled level."""
        c = self.cov_proj(cov)
        s = torch.cat([e(static[:, i]) for i, e in enumerate(self.embeds)], 1)
        h = self.encoder(torch.cat([y, c.flatten(1), s], 1))
        d = self.decoder(h).view(len(y), HORIZON, self.dec)
        out = self.temporal(torch.cat([d, c[:, LOOKBACK:]], 2)).squeeze(-1)
        return out + self.lookback_skip(torch.log1p(y))


def tweedie_loss(log_mu: torch.Tensor, y: torch.Tensor, p: float = 1.1) -> torch.Tensor:
    return (-y * torch.exp(log_mu * (1 - p)) / (1 - p) + torch.exp(log_mu * (2 - p)) / (2 - p)).mean()


class Windows:
    """Builds (history, covariates, static, target, scale) batches on the fly from the wide matrices."""

    def __init__(self, start: int):
        ids, sales = sales_wide()
        self.sales = sales[:, : start - 1]
        self.cal, self.price = covariates()
        self.static = np.stack([ids[c].astype("category").cat.codes.to_numpy() for c in ID_COLS], 1)
        self.cardinalities = [int(self.static[:, i].max()) + 1 for i in range(len(ID_COLS))]
        self.first = first_sale_day(self.sales)
        self.start = start
        # sample series like WRMSSE weighs them: by recent dollar sales (square root to keep small items in play)
        dollars = (self.sales[:, -HORIZON:] * np.nan_to_num(price_matrix()[:, start - 1 - HORIZON : start - 1])).sum(1)
        w = np.sqrt(np.maximum(dollars, 0)) + 1e-3
        self.sampling = w / w.sum()

    def batch(self, series: np.ndarray, cutoff: np.ndarray, device: str):
        """cutoff = 0-based index of the first forecast day for each sample."""
        hist_idx = cutoff[:, None] + np.arange(-LOOKBACK, 0)[None]
        full_idx = cutoff[:, None] + np.arange(-LOOKBACK, HORIZON)[None]
        y = self.sales[series[:, None], hist_idx]
        scale = y[:, -56:].mean(1, keepdims=True) + 1.0
        cov = np.concatenate([self.cal[full_idx], self.price[series[:, None], full_idx][..., None]], 2)
        to = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)  # noqa: E731
        target = None
        if cutoff.max() + HORIZON <= self.sales.shape[1]:
            target = to(self.sales[series[:, None], cutoff[:, None] + np.arange(HORIZON)[None]])
        return to(y / scale), to(cov), to(self.static[series], torch.long), target, to(scale)

    def sample(self, n: int, rng: np.random.Generator, train_days: int):
        """Random training windows whose target lies entirely in the known history after product launch."""
        last = self.sales.shape[1] - HORIZON
        series = rng.choice(len(self.sales), n, p=self.sampling)
        low = np.maximum(self.first[series] + 28, last - train_days)
        ok = low <= last
        series, low = series[ok], low[ok]
        cutoff = low + (rng.random(len(series)) * (last - low + 1)).astype(int)
        return series, np.maximum(cutoff, LOOKBACK)


@torch.no_grad()
def predict(model: TiDE, win: Windows, device: str) -> np.ndarray:
    model.eval()
    out = []
    for k in range(0, len(win.sales), 4096):
        series = np.arange(k, min(k + 4096, len(win.sales)))
        y, cov, static, _, scale = win.batch(series, np.full(len(series), win.start - 1), device)
        out.append((torch.exp(model(y, cov, static)) * scale).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", choices=list(FOLDS), required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps", type=int, default=400, help="batches per epoch")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--train-days", type=int, default=730)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = FOLDS[args.fold]
    win = Windows(start)
    model = TiDE(win.cal.shape[1] + 1, win.cardinalities).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * args.steps)
    metric = WRMSSE(start)
    history, best, t0 = [], (9.0, 0), time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for _ in range(args.steps):
            series, cutoff = win.sample(args.batch, rng, args.train_days)
            y, cov, static, target, scale = win.batch(series, cutoff, device)
            loss = tweedie_loss(model(y, cov, static), target / scale)  # on each series' own scale
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item()
        record = {"epoch": epoch, "loss": round(total / args.steps, 4)}
        if args.fold == "validation":  # the test score is computed once, after the last epoch
            pred = predict(model, win, device)
            score = record["WRMSSE"] = round(metric(pred), 4)
        else:
            pred, score = None, 9.0
        history.append(record)
        print(json.dumps(record), f"{(time.time() - t0) / 60:.1f} min", flush=True)
        if args.fold == "validation" and score < best[0]:
            best = (score, epoch)
            np.save(ROOT / "artifacts" / f"forecast_tide_{args.fold}.npy", pred)
            torch.save(model.state_dict(), ROOT / "artifacts" / f"tide_{args.fold}.pt")
    if args.fold == "test":
        pred = predict(model, win, device)
        best = (metric(pred), args.epochs)
        np.save(ROOT / "artifacts" / "forecast_tide_test.npy", pred)
        torch.save(model.state_dict(), ROOT / "artifacts" / "tide_test.pt")
    result = {"fold": args.fold, "WRMSSE": round(best[0], 4), "best_epoch": best[1],
              "history": history, "args": vars(args), "minutes": round((time.time() - t0) / 60, 1)}
    (ROOT / "reports" / f"tide_{args.fold}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "history"}), flush=True)


if __name__ == "__main__":
    main()
