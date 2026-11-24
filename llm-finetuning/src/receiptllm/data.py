"""CORD v2 receipts (Park et al., 2019, CC-BY-4.0) turned into a text-to-JSON extraction task.

Input: the receipt's OCR text, rebuilt from the annotated word boxes (words grouped by row, rows
sorted top to bottom, words left to right; the field labels are never shown). Target: the official
CORD parse (`gt_parse`) as compact JSON, with `menu` always a list.
Only the `ground_truth` column is downloaded (~1.5 MB), not the 2.3 GB of images.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
SPLITS = ("train", "validation", "test")

SYSTEM = (
    "You extract structured data from receipt OCR text. Answer with one JSON object only, no comments. "
    "Use only these keys, and omit a key when the receipt does not show it:\n"
    '- "menu": list of purchased items, each with "nm" (item name), "cnt" (quantity as printed), '
    '"price" (line price), and when printed "unitprice", "discountprice", "num" (item code), '
    '"sub" (sub-items, same keys), "etc";\n'
    '- "sub_total": {"subtotal_price", "discount_price", "service_price", "tax_price", "etc"};\n'
    '- "total": {"total_price", "cashprice", "changeprice", "creditcardprice", "emoneyprice", '
    '"menuqty_cnt" (number of items), "menutype_cnt" (number of item types), "total_etc"}.\n'
    "Copy every value exactly as printed (keep separators such as 75,000)."
)


def download() -> None:
    """Fetch only the annotation column of the three CORD v2 splits from the Hugging Face Hub."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files = fs.glob("datasets/naver-clova-ix/cord-v2/data/*.parquet")
    RAW.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        tables = []
        for f in sorted(x for x in files if f"/{split}-" in x):
            with fs.open(f, block_size=1 << 20) as fh:
                tables.append(pq.ParquetFile(fh).read(columns=["ground_truth"]))
        pq.write_table(pa.concat_tables(tables), RAW / f"{split}.parquet")


def ocr_text(valid_line: list[dict]) -> str:
    """Receipt text from the annotated words: one line per row, rows top to bottom, words left to right."""
    rows: dict[int, list[tuple[float, float, str]]] = defaultdict(list)
    for line in valid_line:
        for w in line["words"]:
            q = w["quad"]
            rows[w["row_id"]].append(((q["y1"] + q["y3"]) / 2, q["x1"], w["text"]))
    ordered = sorted(rows.values(), key=lambda ws: sum(y for y, _, _ in ws) / len(ws))
    return "\n".join(" ".join(t for _, _, t in sorted(ws, key=lambda w: w[1])) for ws in ordered)


def normalise(parse: dict) -> dict:
    out = dict(parse)
    if isinstance(out.get("menu"), dict):
        out["menu"] = [out["menu"]]
    return out


def load(split: str) -> list[dict]:
    """[{"id", "text", "target"}] for one split."""
    import pyarrow.parquet as pq

    records = []
    for i, row in enumerate(pq.read_table(RAW / f"{split}.parquet").to_pylist()):
        gt = json.loads(row["ground_truth"])
        records.append({"id": f"{split}-{i}", "text": ocr_text(gt["valid_line"]), "target": normalise(gt["gt_parse"])})
    return records


def messages(text: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Receipt OCR text:\n{text}"}]


def target_json(target: dict) -> str:
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    download()
