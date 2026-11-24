"""Fast CPU tests: no model download."""

import json

from fastapi.testclient import TestClient

from receiptllm.data import normalise, ocr_text, target_json
from receiptllm.metrics import leaves, parse_json, score
from receiptllm.train import collate, encode


def word(text, x, y, row):
    return {"text": text, "row_id": row, "quad": {"x1": x, "y1": y, "x3": x + 10, "y3": y + 10}}


def test_ocr_text_orders_rows_and_words():
    lines = [{"words": [word("75,000", 200, 50, 2), word("Tea", 10, 50, 2)]},
             {"words": [word("TOTAL", 10, 100, 3), word("75,000", 200, 101, 3)]},
             {"words": [word("CAFE", 10, 5, 1)]}]
    assert ocr_text(lines) == "CAFE\nTea 75,000\nTOTAL 75,000"


def test_parse_json_tolerates_text_around_and_rejects_garbage():
    assert parse_json('Here: {"a": 1} done') == {"a": 1}
    assert parse_json("no json") is None
    assert parse_json("[1, 2]") is None
    assert parse_json('{"a": ') is None


def test_score_perfect_partial_and_empty():
    t = normalise({"menu": {"nm": "Tea", "price": "18,000"}, "total": {"total_price": "18,000"}})
    assert t["menu"] == [{"nm": "Tea", "price": "18,000"}]
    perfect = score([t], [t])
    assert perfect["field_f1"] == 1.0 and perfect["doc_accuracy"] == 1.0
    wrong_price = json.loads(json.dumps(t))
    wrong_price["total"]["total_price"] = "18.000"
    s = score([wrong_price], [t])
    assert s["recall"] == round(2 / 3, 4) and s["doc_accuracy"] == 0.0
    assert score([None], [t])["field_f1"] == 0.0 and score([None], [t])["valid_json"] == 0.0


def test_leaves_ignores_nulls_and_counts_duplicates():
    obj = {"menu": [{"nm": "Tea"}, {"nm": "Tea"}], "sub_total": {"tax_price": None}}
    assert leaves(obj) == {("menu.nm", "Tea"): 2}


class FakeTok:
    eos_token = "<eos>"
    pad_token_id = 0

    def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
        return "PROMPT " + msgs[-1]["content"]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_encode_masks_prompt_and_learns_answer():
    rec = {"text": "Tea 18,000", "target": {"total": {"total_price": "18,000"}}}
    enc = encode(FakeTok(), rec, max_len=10_000)
    answer = target_json(rec["target"]) + "<eos>"
    n_prompt = len(enc["input_ids"]) - len(answer)
    assert enc["labels"][:n_prompt] == [-100] * n_prompt
    assert "".join(chr(i) for i in enc["labels"][n_prompt:]) == answer
    batch = collate([enc, encode(FakeTok(), {"text": "x", "target": {}}, 10_000)], pad_id=0)
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape


def test_api_contract_with_fake_model():
    from receiptllm import api

    api.state["extract"] = lambda text: '{"total": {"total_price": "18,000"}}'
    with TestClient(api.app) as client:
        r = client.post("/extract", json={"text": "TOTAL 18,000"})
        assert r.status_code == 200
        assert r.json()["data"] == {"total": {"total_price": "18,000"}} and r.json()["valid_json"]
        assert client.post("/extract", json={"text": ""}).status_code == 422
