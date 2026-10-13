"""Dataset loading, cleaning, label encoding, and stratified splitting."""
from __future__ import annotations

import html
import re
import unicodedata
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
HTML_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = html.unescape(unicodedata.normalize("NFKC", str(value or "")))
    text = HTML_RE.sub(" ", text)
    text = URL_RE.sub(" URL ", text)
    text = MENTION_RE.sub(" USER ", text)
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
    # Keep hashtag words, punctuation, and emoji because they carry useful signal.
    text = text.replace("#", "")
    return SPACE_RE.sub(" ", text).strip()


def encode_label(value: object) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"negative", "neg", "0"}:
            return 0
        if normalized in {"positive", "pos", "4", "1"}:
            return 1
    else:
        number = int(value)
        if number == 0:
            return 0
        if number in (1, 4):
            return 1
    raise ValueError(f"Unsupported sentiment label: {value!r}")


def load_source(csv_path: str | None = None) -> pd.DataFrame:
    if csv_path:
        frame = pd.read_csv(csv_path)
        if not {"text", "sentiment"}.issubset(frame.columns):
            raise ValueError("CSV must contain 'text' and 'sentiment' columns")
        return frame[["text", "sentiment"]]
    dataset = load_dataset("stanfordnlp/sentiment140", split="train")
    frame = dataset.to_pandas()
    text_col = "text" if "text" in frame else "tweet"
    label_col = "sentiment" if "sentiment" in frame else "label"
    return frame[[text_col, label_col]].rename(columns={text_col: "text", label_col: "sentiment"})


def prepare_data(data_dir: str | Path, csv_path: str | None = None,
                 sample_size: int | None = None, seed: int = 42):
    data_dir = Path(data_dir)
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    frame = load_source(csv_path)
    frame["text"] = frame["text"].map(clean_text)
    frame = frame[frame["text"].str.len() > 0].copy()
    frame["label"] = frame["sentiment"].map(encode_label)
    frame = frame[["text", "label"]].drop_duplicates(subset="text").reset_index(drop=True)
    if sample_size and len(frame) > sample_size:
        frame, _ = train_test_split(frame, train_size=sample_size, stratify=frame.label, random_state=seed)
        frame = frame.reset_index(drop=True)
    train, validation = train_test_split(frame, test_size=0.1, stratify=frame.label, random_state=seed)
    train.to_csv(processed_dir / "train.csv", index=False)
    validation.to_csv(processed_dir / "validation.csv", index=False)
    return train.reset_index(drop=True), validation.reset_index(drop=True)
