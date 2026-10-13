"""Fine-tune BERT on Sentiment140 or a compatible local CSV."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

try:
    from .preprocessing import prepare_data
except ImportError:
    from preprocessing import prepare_data


class TweetDataset:
    def __init__(self, frame, tokenizer, max_length):
        self.frame = frame.reset_index(drop=True)
        self.tokens = tokenizer(self.frame.text.tolist(), truncation=True, padding="max_length", max_length=max_length)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        return {**{k: np.asarray(v[idx]) for k, v in self.tokens.items()}, "labels": int(self.frame.label.iloc[idx])}


def metrics(prediction):
    logits, labels = prediction
    predicted = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predicted),
            "precision": precision_score(labels, predicted, zero_division=0),
            "recall": recall_score(labels, predicted, zero_division=0),
            "f1": f1_score(labels, predicted, zero_division=0)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Optional CSV with text,sentiment columns")
    parser.add_argument("--sample-size", type=int, default=int(os.getenv("TRAIN_SAMPLE_SIZE", "0")) or None)
    parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", "data"))
    parser.add_argument("--output-dir", default=os.getenv("MODEL_DIR", "artifacts/model"))
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=int(os.getenv("MAX_LENGTH", "128")))
    parser.add_argument("--epochs", type=float, default=float(os.getenv("EPOCHS", "2")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "16")))
    parser.add_argument("--learning-rate", type=float, default=float(os.getenv("LEARNING_RATE", "2e-5")))
    args = parser.parse_args()

    train_df, val_df, test_df = prepare_data(args.data_dir, args.csv, args.sample_size)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2,
                                                                 id2label={0: "negative", 1: "positive"},
                                                                 label2id={"negative": 0, "positive": 1})
    train_set = TweetDataset(train_df, tokenizer, args.max_length)
    val_set = TweetDataset(val_df, tokenizer, args.max_length)
    test_set = TweetDataset(test_df, tokenizer, args.max_length)
    training_args = TrainingArguments(
        output_dir=str(Path(args.output_dir).parent / "checkpoints"),
        num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size, learning_rate=args.learning_rate,
        weight_decay=0.01, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
        logging_steps=100, report_to="none", seed=42,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_set,
                      eval_dataset=val_set, compute_metrics=metrics)
    trainer.train()
    validation_metrics = trainer.evaluate(eval_dataset=val_set, metric_key_prefix="validation")
    test_results = trainer.predict(test_set, metric_key_prefix="test")
    test_metrics = test_results.metrics
    print("Validation metrics:", validation_metrics)
    print("Held-out test metrics:", test_metrics)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump({key: float(value) if isinstance(value, np.generic) else value
                   for key, value in test_metrics.items()}, metrics_file, indent=2)
    print(f"Saved fine-tuned model to {args.output_dir}")


if __name__ == "__main__":
    main()
