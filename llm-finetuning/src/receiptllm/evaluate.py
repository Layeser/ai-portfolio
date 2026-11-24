"""Score a model on a split: the base model zero-shot, or the base model + LoRA adapter.

    python -m receiptllm.evaluate --split test --name base
    python -m receiptllm.evaluate --split test --name lora --adapter artifacts/lora
"""

from __future__ import annotations

import argparse
import json
import time

from receiptllm.data import ROOT, load
from receiptllm.metrics import parse_json, score
from receiptllm.model import BASE, generate
from receiptllm.model import load as load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--adapter")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    records = load(args.split)
    tok, model = load_model(args.base, str(ROOT / args.adapter) if args.adapter else None)
    t0 = time.time()
    outputs = generate(tok, model, [r["text"] for r in records], batch_size=args.batch)
    seconds = time.time() - t0
    result = score([parse_json(o) for o in outputs], [r["target"] for r in records])
    result.update(name=args.name, split=args.split, base=args.base, adapter=args.adapter,
                  seconds_per_receipt=round(seconds / len(records), 3))
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / f"eval_{args.name}_{args.split}.json").write_text(json.dumps(result, indent=2))
    with open(out / f"predictions_{args.name}_{args.split}.jsonl", "w") as f:
        for r, o in zip(records, outputs, strict=True):
            f.write(json.dumps({"id": r["id"], "output": o, "target": r["target"]}, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "recall_by_field"}), flush=True)


if __name__ == "__main__":
    main()
