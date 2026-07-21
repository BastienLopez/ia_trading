"""Préparation et fine-tuning reproductible d'un modèle de sentiment crypto."""

import hashlib
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


LABELS = {"negative": 0, "neutral": 1, "positive": 2}
DEFAULT_DATASET_GLOB = "ai_trading/examples/data/sentiment/analyzed/*.csv"
DLT_DATASET_NAME = "ExponentialScience/DLT-Sentiment-News"
DLT_LICENSE = "CC-BY-NC-4.0"


def _normalise_training_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Valide, déduplique et ordonne le contrat commun d'entraînement."""
    required = {"text", "label_name", "published_at"}
    if not required.issubset(dataset.columns):
        missing = ", ".join(sorted(required - set(dataset.columns)))
        raise ValueError(f"Dataset incomplet: colonne(s) manquante(s) {missing}")

    prepared = dataset.copy()
    prepared["text"] = prepared["text"].fillna("").astype(str).str.strip()
    prepared["label_name"] = prepared["label_name"].astype(str).str.lower().str.strip()
    prepared["published_at"] = pd.to_datetime(prepared["published_at"], errors="coerce")
    prepared = prepared[
        prepared["text"].ne("")
        & prepared["label_name"].isin(LABELS)
        & prepared["published_at"].notna()
    ]
    prepared = prepared.drop_duplicates(subset=["text"], keep="last").sort_values(
        "published_at", kind="stable"
    )
    prepared["label"] = prepared["label_name"].map(LABELS).astype("int64")
    return prepared.reset_index(drop=True)


def validate_training_dataset(dataset: pd.DataFrame) -> Dict[str, int]:
    """Refuse les corpus déséquilibrés ou trop petits avant un entraînement coûteux."""
    counts = dataset["label_name"].value_counts().to_dict()
    missing = set(LABELS) - set(counts)
    if missing:
        raise ValueError(f"Classes absentes du dataset: {', '.join(sorted(missing))}")
    if len(dataset) < 1_000 or min(counts.values()) < 100:
        raise ValueError("Dataset insuffisant: 1 000 exemples et 100 exemples par classe requis")
    return {label: int(counts[label]) for label in LABELS}


def load_crypto_sentiment_dataset(paths: Iterable[Path]) -> pd.DataFrame:
    """Charge les exports crypto, déduplique et normalise le contrat d'entraînement."""
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        required = {"title", "global_sentiment_label"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Dataset incomplet: {path}")
        body = frame["body"].fillna("") if "body" in frame else pd.Series("", index=frame.index)
        published_at = (
            frame["published_at"]
            if "published_at" in frame
            else pd.Series(pd.NaT, index=frame.index)
        )
        text = frame["title"].fillna("") + " " + body
        prepared = pd.DataFrame(
            {
                "text": text.str.strip(),
                "label_name": frame["global_sentiment_label"].str.lower().str.strip(),
                "published_at": published_at,
            }
        )
        frames.append(prepared)

    if not frames:
        raise ValueError("Aucun fichier de sentiment crypto fourni")
    return _normalise_training_dataset(pd.concat(frames, ignore_index=True))


def load_dlt_sentiment_news() -> pd.DataFrame:
    """Charge le corpus crypto DLT sous licence CC-BY-NC-4.0 pour usage personnel."""
    from datasets import load_dataset

    source = load_dataset(DLT_DATASET_NAME, split="train").to_pandas()
    required = {"text", "market_direction", "timestamp"}
    if not required.issubset(source.columns):
        missing = ", ".join(sorted(required - set(source.columns)))
        raise ValueError(f"Schéma DLT invalide: colonne(s) manquante(s) {missing}")
    label_map = {0: "neutral", 1: "negative", 2: "positive"}
    dataset = pd.DataFrame(
        {
            "text": source["text"].fillna("").str.strip(),
            "label_name": source["market_direction"].map(label_map),
            "published_at": pd.to_datetime(source["timestamp"], errors="coerce"),
        }
    )
    return _normalise_training_dataset(dataset)


def dataset_fingerprint(dataset: pd.DataFrame) -> str:
    """Empreinte stable à enregistrer avec chaque modèle entraîné."""
    payload = dataset[["text", "label"]].to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def chronological_split(dataset: pd.DataFrame, validation_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare train/validation sans fuite temporelle."""
    if not 0 < validation_ratio < 0.5:
        raise ValueError("validation_ratio doit être compris entre 0 et 0.5")
    if len(dataset) < 10:
        raise ValueError("Au moins 10 exemples sont requis pour le fine-tuning")
    dataset = dataset.sort_values("published_at", kind="stable").reset_index(drop=True)
    split_at = int(len(dataset) * (1 - validation_ratio))
    return dataset.iloc[:split_at].copy(), dataset.iloc[split_at:].copy()


def fine_tune_crypto_sentiment(
    output_dir: str,
    dataset_glob: str = DEFAULT_DATASET_GLOB,
    use_dlt_dataset: bool = True,
    model_name: str = "finiteautomata/bertweet-base-sentiment-analysis",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    seed: int = 42,
    class_weighted_loss: bool = True,
    max_steps: int = -1,
    max_eval_samples: int = -1,
    use_fp16: Optional[bool] = None,
) -> dict:
    """Entraîne un modèle crypto en CUDA/FP16 si le GPU Docker est disponible."""
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    dataset = (
        load_dlt_sentiment_news()
        if use_dlt_dataset
        else load_crypto_sentiment_dataset(sorted(Path().glob(dataset_glob)))
    )
    cuda_available = torch.cuda.is_available()
    fp16_enabled = cuda_available if use_fp16 is None else bool(use_fp16 and cuda_available)
    class_distribution = validate_training_dataset(dataset)
    train_frame, validation_frame = chronological_split(dataset)
    class_weights = np.ones(len(LABELS), dtype=np.float32)
    if class_weighted_loss:
        train_counts = train_frame["label"].value_counts().reindex(range(len(LABELS)), fill_value=0)
        class_weights = (len(train_frame) / (len(LABELS) * train_counts.to_numpy())).astype(np.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    max_length = min(256, tokenizer.model_max_length)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    train_dataset = Dataset.from_pandas(train_frame[["text", "label"]], preserve_index=False).map(tokenize, batched=True)
    validation_dataset = Dataset.from_pandas(validation_frame[["text", "label"]], preserve_index=False).map(tokenize, batched=True)
    if max_eval_samples > 0:
        validation_dataset = validation_dataset.select(range(min(max_eval_samples, len(validation_dataset))))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label={value: key.upper() for key, value in LABELS.items()},
        label2id={key.upper(): value for key, value in LABELS.items()},
        ignore_mismatched_sizes=True,
    )

    def compute_metrics(prediction) -> Dict[str, float]:
        """Mesures de qualité indépendantes de la perte d'entraînement."""
        logits, labels = prediction
        predictions = np.argmax(logits, axis=-1)
        accuracy = float((predictions == labels).mean())
        f1_scores = []
        for label in range(len(LABELS)):
            true_positive = np.sum((predictions == label) & (labels == label))
            false_positive = np.sum((predictions == label) & (labels != label))
            false_negative = np.sum((predictions != label) & (labels == label))
            precision = true_positive / max(1, true_positive + false_positive)
            recall = true_positive / max(1, true_positive + false_negative)
            f1_scores.append(2 * precision * recall / max(1e-12, precision + recall))
        return {"accuracy": accuracy, "macro_f1": float(np.mean(f1_scores))}

    arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        max_steps=max_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=fp16_enabled,
        dataloader_pin_memory=cuda_available,
        report_to=[],
    )
    class WeightedTrainer(Trainer):
        """Équilibre les classes sans modifier le split temporel."""

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits,
                labels,
                weight=torch.as_tensor(class_weights, device=outputs.logits.device),
            )
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    evaluation = trainer.evaluate()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    metadata = {
        "model_name": model_name,
        "dataset_fingerprint": dataset_fingerprint(dataset),
        "samples": len(dataset),
        "train_samples": len(train_frame),
        "validation_samples": len(validation_frame),
        "evaluated_samples": len(validation_dataset),
        "dataset_source": DLT_DATASET_NAME if use_dlt_dataset else dataset_glob,
        "dataset_license": DLT_LICENSE if use_dlt_dataset else "local",
        "class_distribution": class_distribution,
        "device": torch.cuda.get_device_name(0) if cuda_available else "cpu",
        "fp16": fp16_enabled,
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seed": seed,
            "class_weighted_loss": class_weighted_loss,
            "class_weights": class_weights.tolist(),
        },
        "evaluation": evaluation,
    }
    Path(output_dir, "training_metadata.json").write_text(pd.Series(metadata).to_json(indent=2), encoding="utf-8")
    return metadata
