"""Dataset loading, cleaning, label encoding, and stratified splitting."""
from __future__ import annotations

import html
import csv
import io
import re
import shutil
import unicodedata
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

import pandas as pd
from sklearn.model_selection import train_test_split

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
HTML_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
SENTIMENT140_URL = "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
SENTIMENT140_TRAIN_FILE = "training.1600000.processed.noemoticon.csv"


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


def _download_sentiment140(data_dir: str | Path) -> Path:
    """Download and cache Stanford's raw archive; no Hugging Face code is run."""
    raw_dir = Path(data_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / "sentiment140.zip"
    if archive_path.exists():
        try:
            with ZipFile(archive_path) as archive:
                if any(Path(name).name == SENTIMENT140_TRAIN_FILE for name in archive.namelist()):
                    return archive_path
        except BadZipFile:
            pass

    temporary_path = archive_path.with_suffix(".zip.part")
    request = Request(SENTIMENT140_URL, headers={"User-Agent": "sentiment-analysis-engine/1.0"})
    print(f"Downloading raw Sentiment140 archive from Stanford to {archive_path}...")
    with urlopen(request, timeout=60) as response, temporary_path.open("wb") as destination:
        shutil.copyfileobj(response, destination)
    try:
        with ZipFile(temporary_path) as archive:
            if not any(Path(name).name == SENTIMENT140_TRAIN_FILE for name in archive.namelist()):
                raise ValueError(f"Downloaded archive does not contain {SENTIMENT140_TRAIN_FILE}")
    except BadZipFile as exc:
        temporary_path.unlink(missing_ok=True)
        raise ValueError("Downloaded Sentiment140 archive is not a valid ZIP file") from exc
    temporary_path.replace(archive_path)
    return archive_path


def load_source(csv_path: str | None = None, data_dir: str | Path = "data") -> pd.DataFrame:
    if csv_path:
        frame = pd.read_csv(csv_path)
        if not {"text", "sentiment"}.issubset(frame.columns):
            raise ValueError("CSV must contain 'text' and 'sentiment' columns")
        return frame[["text", "sentiment"]]

    archive_path = _download_sentiment140(data_dir)
    with ZipFile(archive_path) as archive:
        train_member = next(
            name for name in archive.namelist()
            if Path(name).name == SENTIMENT140_TRAIN_FILE
        )
        records = []
        with archive.open(train_member) as binary_file:
            text_file = io.TextIOWrapper(binary_file, encoding="ISO-8859-1", newline="")
            for row_number, row in enumerate(csv.reader(text_file, delimiter=",", quotechar='"'), start=1):
                if len(row) != 6:
                    raise ValueError(
                        f"Malformed Sentiment140 row {row_number}: expected 6 columns, found {len(row)}"
                    )
                # Stanford's raw layout: polarity, tweet ID, date, query, user, tweet text.
                records.append({"sentiment": int(row[0]), "text": row[5]})
    return pd.DataFrame.from_records(records, columns=["text", "sentiment"])


def prepare_data(data_dir: str | Path, csv_path: str | None = None,
                 sample_size: int | None = None, seed: int = 42):
    data_dir = Path(data_dir)
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    frame = load_source(csv_path, data_dir)
    frame["text"] = frame["text"].map(clean_text)
    frame = frame[frame["text"].str.len() > 0].copy()
    frame["label"] = frame["sentiment"].map(encode_label)
    frame = frame[["text", "label"]].drop_duplicates(subset="text").reset_index(drop=True)
    if sample_size and len(frame) > sample_size:
        frame, _ = train_test_split(frame, train_size=sample_size, stratify=frame.label, random_state=seed)
        frame = frame.reset_index(drop=True)
    # Keep the test partition untouched until the final evaluation: 80/10/10 overall.
    train_validation, test = train_test_split(
        frame, test_size=0.1, stratify=frame.label, random_state=seed
    )
    train, validation = train_test_split(
        train_validation,
        test_size=1 / 9,
        stratify=train_validation.label,
        random_state=seed,
    )
    train.to_csv(processed_dir / "train.csv", index=False)
    validation.to_csv(processed_dir / "validation.csv", index=False)
    test.to_csv(processed_dir / "test.csv", index=False)
    return tuple(part.reset_index(drop=True) for part in (train, validation, test))
