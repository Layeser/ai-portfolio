"""Two classifiers behind one interface: fit / predict_proba / save / load.

* TfidfModel        word + character n-grams, logistic regression. Seconds to train, strong baseline.
* TransformerModel  fine-tuned XLM-RoBERTa (multilingual: titles are FR, descriptions FR/DE/EN).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import joblib
import numpy as np


class TfidfModel:
    name = "tfidf"

    def __init__(self, C: float = 0.5):
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_pipeline, make_union
        from sklearn.svm import LinearSVC

        # Memory-conscious choice: lbfgs logistic regression on 500k features x 100 classes kept
        # several GB of gradient history and was OOM-killed on a 7 GB WSL. A linear SVM (liblinear,
        # one-vs-rest) with sigmoid calibration gives probabilities with a fraction of the memory.
        features = make_union(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True,
                            dtype=np.float32),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=150_000,
                            sublinear_tf=True, dtype=np.float32),
        )
        classifier = CalibratedClassifierCV(LinearSVC(C=C), method="sigmoid", cv=3)
        self.pipeline = make_pipeline(features, classifier)

    @property
    def classes_(self) -> list[str]:
        return list(self.pipeline.classes_)

    def fit(self, texts, labels) -> TfidfModel:
        self.pipeline.fit(list(texts), list(labels))
        return self

    def predict_proba(self, texts) -> np.ndarray:
        return self.pipeline.predict_proba(list(texts))

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path / "tfidf.joblib", compress=3)

    @classmethod
    def load(cls, path: Path) -> TfidfModel:
        model = cls.__new__(cls)
        model.pipeline = joblib.load(path / "tfidf.joblib")
        return model


class TransformerModel:
    name = "xlmr"

    def __init__(self, base: str = "xlm-roberta-base", max_length: int = 192, epochs: int = 3,
                 batch_size: int = 32, lr: float = 3e-5, device: str | None = None):
        import torch

        self.base, self.max_length, self.epochs = base, max_length, epochs
        self.batch_size, self.lr = batch_size, lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = self.model = None
        self._classes: list[str] = []
        self.history: list[dict] = []

    @property
    def classes_(self) -> list[str]:
        return self._classes

    def _batches(self, texts: list[str], labels: list[int] | None = None, shuffle: bool = False):
        import torch

        order = np.random.permutation(len(texts)) if shuffle else np.arange(len(texts))
        for start in range(0, len(texts), self.batch_size):
            idx = order[start:start + self.batch_size]
            enc = self.tokenizer([texts[i] for i in idx], truncation=True, max_length=self.max_length,
                                 padding=True, return_tensors="pt").to(self.device)
            if labels is not None:
                enc["labels"] = torch.tensor([labels[i] for i in idx], device=self.device)
            yield enc

    def fit(self, texts, labels, valid_texts=None, valid_labels=None) -> TransformerModel:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

        torch.manual_seed(42)
        np.random.seed(42)
        texts, labels = list(texts), list(labels)
        self._classes = sorted(set(labels))
        index = {c: i for i, c in enumerate(self._classes)}
        y = [index[label] for label in labels]

        self.tokenizer = AutoTokenizer.from_pretrained(self.base)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base, num_labels=len(self._classes)).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        steps = self.epochs * math.ceil(len(texts) / self.batch_size)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps), steps)

        best_acc, best_state = -1.0, None
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            start, total, n_steps = time.perf_counter(), 0.0, 0
            for batch in self._batches(texts, y, shuffle=True):
                with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
                    loss = self.model(**batch).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total += loss.item()
                n_steps += 1
            record = {"epoch": epoch, "train_loss": round(total / n_steps, 4),
                      "minutes": round((time.perf_counter() - start) / 60, 1)}
            if valid_texts is not None:
                pred = np.array(self._classes)[self.predict_proba(valid_texts).argmax(1)]
                record["valid_accuracy"] = round(float((pred == np.array(list(valid_labels))).mean()), 4)
                if record["valid_accuracy"] > best_acc:  # keep the best epoch (early stopping)
                    best_acc = record["valid_accuracy"]
                    # copy to CPU: a GPU copy (+1.1 GB) overflowed the 8 GB card and doubled epoch time
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            self.history.append(record)
            print(record, flush=True)
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, texts) -> np.ndarray:
        import torch

        self.model.eval()
        out = []
        with torch.no_grad(), torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
            for batch in self._batches(list(texts)):
                out.append(torch.softmax(self.model(**batch).logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (path / "prodcat.json").write_text(json.dumps({"classes": self._classes, "max_length": self.max_length,
                                                       "history": self.history}))

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> TransformerModel:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        meta = json.loads((path / "prodcat.json").read_text())
        model = cls(base=str(path), max_length=meta["max_length"], device=device)
        model._classes, model.history = meta["classes"], meta["history"]
        model.tokenizer = AutoTokenizer.from_pretrained(path)
        model.model = AutoModelForSequenceClassification.from_pretrained(path).to(model.device)
        return model


def load_model(artifacts_dir: Path):
    """Serve the trained model with the best *validation* macro-F1 (never chosen on the test set)."""
    scores = {}
    for name in ("tfidf", "xlmr"):
        metrics_path = artifacts_dir / name / "metrics.json"
        if metrics_path.exists():
            scores[name] = json.loads(metrics_path.read_text())["valid"]["f1_macro"]
    best = max(scores, key=scores.get) if scores else "tfidf"
    return TransformerModel.load(artifacts_dir / "xlmr") if best == "xlmr" else TfidfModel.load(artifacts_dir / "tfidf")
