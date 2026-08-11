from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .data import TARGET_FIELDS, EventWindowDataset, collate_windows
from .io import read_json, sha256_file, write_json
from .model import build_model, configured_model_variant
from .prepare import UNK_TOKEN
from .runtime_metadata import collect_runtime_metadata
from .representation_schema import checkpoint_schema


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    observed_dir: str | Path,
    prepared_dir: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    # Fail before dataset reads, output writes, or accelerator initialization.
    model_variant = configured_model_variant(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 0))
    seed_everything(seed)
    device = resolve_device(str(config["training"].get("device", "auto")))

    train_dataset = EventWindowDataset(observed_dir, prepared_dir, "train", config)
    validation_dataset = EventWindowDataset(observed_dir, prepared_dir, "validation", config)
    loader_settings = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["training"].get("num_workers", 0)),
        "collate_fn": collate_windows,
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_settings)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_settings)

    vocabularies = read_json(Path(prepared_dir) / "vocabularies.json")
    prepared_metadata_path = Path(prepared_dir) / "prepared_metadata.json"
    prepared_metadata = read_json(prepared_metadata_path)
    categorical_fields = list(train_dataset.categorical_fields)
    model = build_model(
        vocabularies,
        len(train_dataset.continuous_fields),
        config,
        categorical_fields=categorical_fields,
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    history: list[dict[str, Any]] = []
    best_validation = math.inf
    checkpoint_path = output_dir / "best_model.pt"
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_metrics = _run_epoch(model, train_loader, config, device, optimizer)
        with torch.no_grad():
            validation_metrics = _run_epoch(model, validation_loader, config, device, None)
        epoch_record = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        history.append(epoch_record)
        if validation_metrics["loss"] < best_validation:
            best_validation = validation_metrics["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_variant": model_variant,
                    "representation_schema": checkpoint_schema(
                        model_variant=model_variant,
                        component_dimensions={name: int(config["model"]["user_embedding_dim"])
                                              for name in ("persistent", "context", "combined")},
                        categorical_fields=categorical_fields,
                        continuous_fields=list(train_dataset.continuous_fields),
                        preparation_hash=sha256_file(prepared_metadata_path),
                        source_files=dict(prepared_metadata["source_files"]),
                        train_end=str(prepared_metadata["train_end"]),
                        validation_end=str(prepared_metadata["validation_end"]),
                    ),
                    "config": config,
                    "vocabularies": vocabularies,
                    "categorical_fields": categorical_fields,
                    "continuous_fields": train_dataset.continuous_fields,
                    "epoch": epoch,
                    "validation_metrics": validation_metrics,
                },
                checkpoint_path,
            )
        print(
            f"epoch={epoch:02d} train_loss={train_metrics['loss']:.4f} "
            f"validation_loss={validation_metrics['loss']:.4f}"
        )

    report = {
        "runtime_metadata": collect_runtime_metadata(
            duration_seconds=time.perf_counter() - started, seed=seed, device=device
        ).to_dict(),
        "device": str(device),
        "model_variant": model_variant,
        "categorical_fields": categorical_fields,
        "continuous_fields": list(train_dataset.continuous_fields),
        "train_windows": len(train_dataset),
        "validation_windows": len(validation_dataset),
        "best_validation_loss": best_validation,
        "checkpoint": str(checkpoint_path.resolve()),
        "history": history,
    }
    write_json(report, output_dir / "training_report.json")
    return report


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    optimizer: AdamW | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    correct: dict[str, int] = {}
    top5_correct: dict[str, int] = {}
    objective_weights = config["objectives"]

    for batch_number, batch in enumerate(loader, start=1):
        _validate_batch_values(batch, model, batch_number)
        batch = _move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        _, logits = model(batch["categorical"], batch["continuous"], batch["lengths"])
        losses = []
        for name, prediction in logits.items():
            weight = float(objective_weights.get(name, 0.0))
            if weight <= 0:
                continue
            target = batch["targets"][name]
            losses.append(weight * F.cross_entropy(prediction, target))
            correct[name] = correct.get(name, 0) + int((prediction.argmax(dim=1) == target).sum())
            k = min(5, prediction.shape[1])
            top5 = prediction.topk(k, dim=1).indices
            top5_correct[name] = top5_correct.get(name, 0) + int((top5 == target[:, None]).any(dim=1).sum())

        consistency_weight = float(objective_weights.get("cross_window_consistency", 0.0))
        if consistency_weight > 0:
            early = model.encode_components(
                batch["early_categorical"],
                batch["early_continuous"],
                batch["early_lengths"],
                augment=True,
            )
            late = model.encode_components(
                batch["late_categorical"],
                batch["late_continuous"],
                batch["late_lengths"],
                augment=True,
            )
            losses.append(
                consistency_weight
                * (1.0 - F.cosine_similarity(early.combined, late.combined).mean())
            )

        if not losses:
            raise ValueError("No active training objectives matched the model heads")
        loss = torch.stack(losses).sum()
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["gradient_clip_norm"])
            )
            optimizer.step()
        batch_size = len(batch["user_id"])
        total_loss += float(loss.detach()) * batch_size
        total_examples += batch_size

    metrics = {"loss": total_loss / max(total_examples, 1)}
    for name in sorted(correct):
        metrics[f"{name}_accuracy"] = correct[name] / max(total_examples, 1)
        metrics[f"{name}_top5"] = top5_correct[name] / max(total_examples, 1)
    return metrics


def _validate_batch_values(
    batch: dict[str, Any],
    model: torch.nn.Module,
    batch_number: int,
) -> None:
    """Fail on malformed prepared data before an accelerator sees the batch."""
    encoder = model.module if hasattr(model, "module") else model
    user_preview = ", ".join(str(value) for value in batch.get("user_id", [])[:5])

    for prefix in ("", "early_", "late_"):
        categorical = batch[f"{prefix}categorical"]
        continuous = batch[f"{prefix}continuous"]
        lengths = batch[f"{prefix}lengths"]
        if categorical.shape[-1] != len(encoder.categorical_fields):
            raise ValueError(
                f"Batch {batch_number} ({user_preview}) has {categorical.shape[-1]} "
                f"categorical fields; model expects {len(encoder.categorical_fields)}"
            )
        invalid_lengths = (lengths <= 0) | (lengths > categorical.shape[1])
        if bool(invalid_lengths.any()):
            raise ValueError(
                f"Batch {batch_number} ({user_preview}) has invalid {prefix}lengths: "
                f"{lengths[invalid_lengths].tolist()} for padded width {categorical.shape[1]}"
            )
        if not bool(torch.isfinite(continuous).all()):
            raise ValueError(
                f"Batch {batch_number} ({user_preview}) contains non-finite "
                f"values in {prefix}continuous"
            )
        for field_index, field in enumerate(encoder.categorical_fields):
            values = categorical[:, :, field_index]
            minimum = int(values.min())
            maximum = int(values.max())
            vocabulary_size = encoder.categorical_embeddings[field].num_embeddings
            if minimum < 0 or maximum >= vocabulary_size:
                raise ValueError(
                    f"Batch {batch_number} ({user_preview}) has out-of-range IDs for "
                    f"{prefix}{field}: min={minimum}, max={maximum}, "
                    f"vocabulary_size={vocabulary_size}"
                )

    for objective, target in batch["targets"].items():
        if objective not in encoder.heads:
            continue
        minimum = int(target.min())
        maximum = int(target.max())
        class_count = encoder.heads[objective].out_features
        if minimum < 0 or maximum >= class_count:
            raise ValueError(
                f"Batch {batch_number} ({user_preview}) has out-of-range targets for "
                f"{objective}: min={minimum}, max={maximum}, classes={class_count}"
            )


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            # Sequence lengths are control metadata, not model features. Keep
            # them on CPU to avoid unnecessary MPS synchronization/copies.
            moved[key] = value if key.endswith("lengths") else value.to(device)
        elif isinstance(value, dict):
            moved[key] = {
                nested_key: nested_value.to(device) if isinstance(nested_value, torch.Tensor) else nested_value
                for nested_key, nested_value in value.items()
            }
        else:
            moved[key] = value
    return moved


def evaluate_next_event(
    observed_dir: str | Path,
    prepared_dir: str | Path,
    checkpoint_path: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    device = resolve_device(str(config["training"].get("device", "auto")))
    train_dataset = EventWindowDataset(observed_dir, prepared_dir, "train", config)
    dataset = EventWindowDataset(observed_dir, prepared_dir, "test", config)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
        collate_fn=collate_windows,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(
        checkpoint["vocabularies"],
        len(checkpoint["continuous_fields"]),
        checkpoint["config"],
        categorical_fields=_checkpoint_categorical_fields(checkpoint, dataset.categorical_fields),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    with torch.no_grad():
        metrics = _run_epoch(model, loader, checkpoint["config"], device, None)
        predictions: dict[str, list[int]] = {}
        truths: dict[str, list[int]] = {}
        for batch_number, batch in enumerate(loader, start=1):
            _validate_batch_values(batch, model, batch_number)
            moved = _move_batch(batch, device)
            _, logits = model(moved["categorical"], moved["continuous"], moved["lengths"])
            for objective, values in logits.items():
                if objective not in moved["targets"]:
                    continue
                predictions.setdefault(objective, []).extend(values.argmax(dim=1).cpu().tolist())
                truths.setdefault(objective, []).extend(moved["targets"][objective].cpu().tolist())

    diagnostics: dict[str, Any] = {}
    for objective, field in TARGET_FIELDS.items():
        if objective not in predictions or field not in train_dataset.vocabularies:
            continue
        train_targets = [
            train_dataset._token_id(field, train_dataset.events.iloc[reference.target_index][field])
            for reference in train_dataset.references
        ]
        vocabulary = train_dataset.vocabularies[field]
        diagnostics[objective] = next_event_classification_diagnostics(
            train_targets=np.asarray(train_targets, dtype=np.int64),
            truths=np.asarray(truths[objective], dtype=np.int64),
            predictions=np.asarray(predictions[objective], dtype=np.int64),
            class_count=len(vocabulary),
            unknown_label_id=int(vocabulary[UNK_TOKEN]),
        )
    return {
        "schema_version": "geoembeddings-next-event-evaluation/2.0",
        "test_windows": len(dataset),
        **metrics,
        "predictive_diagnostics": diagnostics,
        "interpretation": (
            "Predictive diagnostics only; improvement over a train-fitted majority baseline "
            "is not evidence of embedding quality or disentanglement."
        ),
    }


def next_event_classification_diagnostics(
    *,
    train_targets: np.ndarray,
    truths: np.ndarray,
    predictions: np.ndarray,
    class_count: int,
    unknown_label_id: int,
) -> dict[str, Any]:
    """Compare predictions with a majority control fitted only on train targets."""
    train_targets = np.asarray(train_targets, dtype=np.int64)
    truths = np.asarray(truths, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if truths.shape != predictions.shape:
        raise ValueError("truths and predictions must have identical shapes")
    if class_count <= 0 or not 0 <= unknown_label_id < class_count:
        raise ValueError("class_count and unknown_label_id do not define a valid vocabulary")

    known_train = train_targets[train_targets != unknown_label_id]
    train_counts = np.bincount(known_train, minlength=class_count)
    train_counts[unknown_label_id] = 0
    majority_id = (
        int(np.flatnonzero(train_counts == train_counts.max())[0])
        if train_counts.sum()
        else None
    )
    known = truths != unknown_label_id
    known_count = int(known.sum())
    total = int(truths.size)
    unknown_count = total - known_count
    evaluation_counts = np.bincount(truths[known], minlength=class_count)

    def scores(values: np.ndarray) -> dict[str, Any]:
        correct_known = int(np.sum(values[known] == truths[known])) if known_count else 0
        recalls: list[float] = []
        f1s: list[float] = []
        per_class: dict[str, Any] = {}
        for label in range(class_count):
            if label == unknown_label_id:
                continue
            support = int(evaluation_counts[label])
            tp = int(np.sum((values[known] == label) & (truths[known] == label)))
            fp = int(np.sum((values[known] == label) & (truths[known] != label)))
            fn = support - tp
            recall = tp / support if support else None
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
            per_class[str(label)] = {"support": support, "recall": recall, "f1": f1}
            if support:
                recalls.append(float(recall))
                f1s.append(float(f1 if f1 is not None else 0.0))
        return {
            "known_label_accuracy": correct_known / known_count if known_count else 0.0,
            "coverage_aware_accuracy": correct_known / total if total else 0.0,
            "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
            "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
            "per_class": per_class,
        }

    learned = scores(predictions)
    naive_predictions = np.full(
        truths.shape, majority_id if majority_id is not None else unknown_label_id
    )
    naive = scores(naive_predictions)
    return {
        "status": "ok" if known_count else "zero_known_label_coverage",
        "fit_split": "train",
        "majority_label_id": majority_id,
        "train_class_counts": {str(i): int(train_counts[i]) for i in range(class_count)},
        "evaluation_known_class_counts": {
            str(i): int(evaluation_counts[i]) for i in range(class_count)
        },
        "empty_evaluation_class_count": int(
            np.sum(evaluation_counts[np.arange(class_count) != unknown_label_id] == 0)
        ),
        "known_label_count": known_count,
        "unknown_label_count": unknown_count,
        "known_label_coverage": known_count / total if total else 0.0,
        "learned": learned,
        "naive": naive,
        "deltas": {
            "known_label_accuracy": learned["known_label_accuracy"] - naive["known_label_accuracy"],
            "coverage_aware_accuracy": (
                learned["coverage_aware_accuracy"] - naive["coverage_aware_accuracy"]
            ),
            "macro_f1": learned["macro_f1"] - naive["macro_f1"],
            "balanced_accuracy": learned["balanced_accuracy"] - naive["balanced_accuracy"],
        },
    }


def _checkpoint_categorical_fields(
    checkpoint: dict[str, Any],
    prepared_fields: list[str],
) -> list[str]:
    """Resolve and verify the positional categorical schema for a checkpoint."""
    checkpoint_fields = checkpoint.get("categorical_fields")
    if checkpoint_fields is None:
        # Compatibility for checkpoints created before the field order was
        # stored explicitly. Their resolved training config is authoritative.
        data_config = checkpoint.get("config", {}).get("data", {})
        checkpoint_fields = list(data_config.get("categorical_fields", []))
        if bool(data_config.get("include_object_id", False)):
            checkpoint_fields.append("object_id")
    checkpoint_fields = list(checkpoint_fields)
    if checkpoint_fields != list(prepared_fields):
        raise ValueError(
            "Checkpoint categorical field order does not match prepared data: "
            f"checkpoint={checkpoint_fields}, prepared={list(prepared_fields)}. "
            "Rerun prepare and train with the same embedding configuration."
        )
    return checkpoint_fields
