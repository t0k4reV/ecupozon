"""Shared, category-agnostic utilities for Gemma LoRA training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "google/gemma-4-E4B-it"
TRAINING_SEED = 2026
MINORITY_CLASS_TARGET_SHARE = 0.30
MAX_DESCRIPTION_CHARS = 8000
TRAINING_EPOCHS = 2
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-4
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
TARGET_MODULES = "all-linear"
SUPPORTED_CATEGORIES = {"БАД", "Легковоспламеняющиеся"}
REQUIRED_MANIFEST_COLUMNS = {
    "id",
    "name",
    "description",
    "category",
    "label",
    "image_paths",
}

ImageSelector = Callable[[dict[str, Any], Path], list[Path]]
ImageLoader = Callable[[Path], Any]
ContentBuilder = Callable[[dict[str, Any], "TrainingProfile", list[Path]], list[dict[str, str]]]
PostTrainingEvaluator = Callable[[Any, Any, Any, Path, Path], dict[str, Any]]


@dataclass(frozen=True)
class TrainingProfile:
    category: str
    experiment_slug: str
    adapter_slug: str
    classification_rule: str
    validation_fraction: float
    evaluate_each_epoch: bool
    checkpoint_limit: int
    training_image_profile: dict[str, Any]
    default_inference: dict[str, Any] | None = None
    expected_split_sizes: dict[str, int] | None = None


def get_default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_training_arguments(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--project-root", type=Path, default=get_default_project_root())
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Load the model and run one forward pass without training or writing adapters.",
    )
    return parser.parse_args()


def calculate_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(json_path: Path, payload: object) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = json_path.with_suffix(json_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(json_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def build_training_prompt(product: dict[str, Any], profile: TrainingProfile) -> str:
    description = str(product.get("description", ""))[:MAX_DESCRIPTION_CHARS]
    return (
        f"Категория: {profile.category}\n"
        f"Правила: {profile.classification_rule}\n"
        f"Название: {product.get('name', '')}\n"
        f"Описание: {description}\n"
        "Определи соответствие правилам. Ответь только цифрой 0 или 1."
    )


def build_default_training_content(
    product: dict[str, Any],
    profile: TrainingProfile,
    _selected_image_paths: list[Path],
) -> list[dict[str, str]]:
    return [{"type": "text", "text": build_training_prompt(product, profile)}]


def oversample_minority_class(products: Any) -> Any:
    import pandas as pd

    label_counts = products["label"].value_counts()
    minority_label = label_counts.idxmin()
    majority_count = int(label_counts.max())
    target_minority_count = math.ceil(
        MINORITY_CLASS_TARGET_SHARE * majority_count / (1 - MINORITY_CLASS_TARGET_SHARE)
    )
    minority_products = products.loc[products["label"].eq(minority_label)]
    if len(minority_products) < target_minority_count:
        extra_products = minority_products.sample(
            n=target_minority_count - len(minority_products),
            replace=True,
            random_state=TRAINING_SEED,
        )
        products = pd.concat([products, extra_products], ignore_index=True)
    return products.sample(frac=1, random_state=TRAINING_SEED).reset_index(drop=True)


class ProductTrainingDataset:
    def __init__(self, products: Any) -> None:
        self.products = products.to_dict(orient="records")

    def __len__(self) -> int:
        return len(self.products)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.products[index]


def find_existing_product_images(
    product: dict[str, Any],
    images_directory: Path,
    limit: int | None = None,
) -> list[Path]:
    relative_paths = product.get("image_paths")
    if not isinstance(relative_paths, list):
        return []
    selected_paths = relative_paths if limit is None else relative_paths[:limit]
    image_paths: list[Path] = []
    for relative_path in selected_paths:
        image_path = (images_directory / str(relative_path)).resolve()
        if images_directory != image_path and images_directory not in image_path.parents:
            raise ValueError(f"Unsafe image path in manifest: {relative_path}")
        if image_path.is_file():
            image_paths.append(image_path)
    return image_paths


def tokenize_conversations(
    processor: Any,
    conversations: list[list[dict[str, Any]]],
    *,
    training: bool,
) -> Any:
    common_arguments = {
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "add_generation_prompt": not training,
        "enable_thinking": False,
    }
    try:
        return processor.apply_chat_template(
            conversations,
            processor_kwargs={"padding": True},
            **common_arguments,
        )
    except TypeError as error:
        if "processor_kwargs" not in str(error):
            raise
        return processor.apply_chat_template(
            conversations,
            padding=True,
            **common_arguments,
        )


def get_label_token_ids(processor: Any) -> tuple[int, int]:
    zero = processor.tokenizer("0", add_special_tokens=False).input_ids
    one = processor.tokenizer("1", add_special_tokens=False).input_ids
    if len(zero) != 1 or len(one) != 1:
        raise RuntimeError(f"Expected single-token labels, got 0={zero}, 1={one}")
    return int(zero[0]), int(one[0])


def score_validation_images(
    *,
    model: Any,
    processor: Any,
    validation_products: Any,
    images_directory: Path,
    profile: TrainingProfile,
    load_image: ImageLoader,
    build_content: ContentBuilder = build_default_training_content,
    max_images: int,
    batch_size: int = 2,
) -> Any:
    """Score each validation image separately and return per-product probabilities."""
    import pandas as pd
    import torch

    label_token_ids = get_label_token_ids(processor)
    variants: list[tuple[dict[str, Any], int, Path | None]] = []
    for product in validation_products.to_dict(orient="records"):
        image_paths = find_existing_product_images(product, images_directory, limit=max_images)
        variants.extend(
            (product, image_index, image_path) for image_index, image_path in enumerate(image_paths)
        )
        if not image_paths:
            variants.append((product, 0, None))

    probabilities_by_id: dict[str, dict[int, float]] = {}
    next_variant = 0
    current_batch_size = max(1, batch_size)
    while next_variant < len(variants):
        chunk = variants[next_variant : next_variant + current_batch_size]
        conversations: list[list[dict[str, Any]]] = []
        for product, _, image_path in chunk:
            selected_paths = [image_path] if image_path is not None else []
            content: list[dict[str, Any]] = []
            if image_path is not None:
                try:
                    content.append({"type": "image", "image": load_image(image_path)})
                except (OSError, ValueError) as error:
                    print(f"Skipping validation image {image_path}: {error}", flush=True)
            content.extend(build_content(product, profile, selected_paths))
            conversations.append([{"role": "user", "content": content}])

        model_inputs = generation_output = None
        try:
            model_inputs = tokenize_conversations(processor, conversations, training=False).to(
                model.device
            )
            with torch.inference_mode():
                generation_output = model.generate(
                    **model_inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            label_logits = generation_output.scores[0][:, list(label_token_ids)].float()
            probabilities = label_logits.softmax(dim=-1)[:, 1].detach().cpu().tolist()
        except RuntimeError as error:
            if "out of memory" not in str(error).lower() or current_batch_size == 1:
                raise
            del model_inputs, generation_output
            torch.cuda.empty_cache()
            current_batch_size = 1
            continue

        for (product, image_index, _), probability in zip(chunk, probabilities, strict=True):
            probabilities_by_id.setdefault(str(product["id"]), {})[image_index] = float(probability)
        next_variant += len(chunk)
        del model_inputs, generation_output, label_logits

    rows: list[dict[str, Any]] = []
    for product in validation_products.to_dict(orient="records"):
        scores = probabilities_by_id[str(product["id"])]
        row: dict[str, Any] = {"id": str(product["id"]), "label": int(product["label"])}
        row.update(
            {f"p_img{index + 1}": scores.get(index, math.nan) for index in range(max_images)}
        )
        row["p_max"] = max(scores.values())
        rows.append(row)
    return pd.DataFrame(rows)


def create_training_collator(
    processor: Any,
    profile: TrainingProfile,
    images_directory: Path,
    select_images: ImageSelector,
    load_image: ImageLoader,
    build_content: ContentBuilder = build_default_training_content,
):
    import torch

    def collate_batch(products: list[dict[str, Any]]) -> dict[str, Any]:
        conversations: list[list[dict[str, Any]]] = []
        targets: list[list[int]] = []
        for product in products:
            content: list[dict[str, Any]] = []
            selected_image_paths = select_images(product, images_directory)
            for image_path in selected_image_paths:
                try:
                    content.append({"type": "image", "image": load_image(image_path)})
                except (OSError, ValueError) as error:
                    print(f"Skipping image {image_path}: {error}", flush=True)
            content.extend(build_content(product, profile, selected_image_paths))
            target_text = str(int(product["label"]))
            conversations.append(
                [
                    {"role": "user", "content": content},
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": target_text}],
                    },
                ]
            )
            targets.append(processor.tokenizer(target_text, add_special_tokens=False).input_ids)

        batch = tokenize_conversations(processor, conversations, training=True)
        labels = torch.full_like(batch["input_ids"], -100)
        for row_index, target in enumerate(targets):
            token_ids = batch["input_ids"][row_index].tolist()
            target_length = len(target)
            starts = [
                index
                for index in range(len(token_ids) - target_length + 1)
                if token_ids[index : index + target_length] == target
            ]
            if not starts:
                raise ValueError(f"Target tokens not found for batch row {row_index}")
            start = starts[-1]
            labels[row_index, start : start + target_length] = batch["input_ids"][
                row_index, start : start + target_length
            ]
        batch["labels"] = labels
        return batch

    return collate_batch


def load_validated_inputs(
    project_root: Path,
    images_directory: Path,
    profile: TrainingProfile,
) -> tuple[Any, str, dict[str, str]]:
    import pandas as pd
    import torch

    project_root = project_root.resolve()
    images_directory = images_directory.resolve()
    manifest_path = project_root / "artifacts" / "gemma_lora" / "data" / "products_manifest.jsonl"
    metadata_path = manifest_path.with_name("products_manifest.meta.json")
    model_path = project_root / "models" / "google" / "gemma-4-E4B-it"
    model_manifest_path = model_path / "download_manifest.json"

    for path in (images_directory, model_path):
        if not path.is_dir():
            raise FileNotFoundError(path)
    for path in (manifest_path, metadata_path, model_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Gemma LoRA training")
    if profile.category not in SUPPORTED_CATEGORIES:
        raise ValueError(f"Unsupported training category: {profile.category}")

    products = pd.read_json(manifest_path, lines=True, dtype={"id": "string"})
    missing_columns = REQUIRED_MANIFEST_COLUMNS - set(products.columns)
    if missing_columns:
        raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")
    if products.empty or products["id"].isna().any() or products["id"].eq("").any():
        raise ValueError("Training manifest contains no usable products")
    if products["id"].duplicated().any():
        raise ValueError("Training manifest contains duplicate ids")
    labels = pd.to_numeric(products["label"], errors="raise")
    if not labels.isin({0, 1}).all():
        raise ValueError("Training manifest labels must be binary 0/1")
    products["label"] = labels.astype(int)
    if not products["image_paths"].map(lambda value: isinstance(value, list)).all():
        raise ValueError("Training manifest image_paths must contain JSON arrays")
    if set(products["category"]) != SUPPORTED_CATEGORIES:
        raise ValueError("Manifest categories do not match the supported training categories")

    category_products = products.loc[products["category"].eq(profile.category)].copy()
    label_counts = category_products["label"].value_counts()
    if set(label_counts.index) != {0, 1} or int(label_counts.min()) < 2:
        raise ValueError(
            f"Insufficient stratified samples for {profile.category}: {label_counts.to_dict()}"
        )
    validation_count = math.ceil(len(category_products) * profile.validation_fraction)
    if validation_count < 2 or len(category_products) - validation_count < 2:
        raise ValueError(f"Not enough rows for {profile.category} split")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("schema") != "e_cup.products_manifest"
        or metadata.get("schema_version") != 1
        or metadata.get("manifest_sha256") != calculate_sha256(manifest_path)
    ):
        raise ValueError("Products manifest metadata is missing, unsupported or stale")

    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    if model_manifest.get("model_id") != MODEL_ID:
        raise ValueError("Model manifest uses another base model")
    resolved_revision = model_manifest.get("resolved_revision")
    if (
        not isinstance(resolved_revision, str)
        or len(resolved_revision) != 40
        or not all(character in "0123456789abcdef" for character in resolved_revision)
    ):
        raise ValueError("Model manifest has no valid resolved revision")

    data_provenance = {
        "manifest": str(manifest_path.relative_to(project_root)),
        "manifest_sha256": calculate_sha256(manifest_path),
        "metadata_sha256": calculate_sha256(metadata_path),
    }
    return category_products, resolved_revision, data_provenance


def create_data_split(products: Any, profile: TrainingProfile) -> tuple[Any, Any, Any, dict]:
    from sklearn.model_selection import train_test_split

    training_products_raw, validation_products = train_test_split(
        products,
        test_size=profile.validation_fraction,
        random_state=TRAINING_SEED,
        stratify=products["label"],
    )
    training_products = oversample_minority_class(training_products_raw)
    for split_name, split_products in (
        ("train", training_products_raw),
        ("validation", validation_products),
    ):
        if set(split_products["label"]) != {0, 1}:
            raise ValueError(f"{profile.category} {split_name} split lacks a label")
    split = {
        "seed": TRAINING_SEED,
        "validation_fraction": profile.validation_fraction,
        "train_raw": len(training_products_raw),
        "train_oversampled": len(training_products),
        "validation": len(validation_products),
        "train_labels": {
            str(label): int(count)
            for label, count in training_products["label"].value_counts().items()
        },
        "validation_labels": {
            str(label): int(count)
            for label, count in validation_products["label"].value_counts().items()
        },
    }
    if profile.expected_split_sizes is not None:
        actual_sizes = {
            "train_raw": split["train_raw"],
            "train_oversampled": split["train_oversampled"],
            "validation": split["validation"],
        }
        if actual_sizes != profile.expected_split_sizes:
            raise ValueError(
                f"Unexpected split sizes for {profile.category}: "
                f"expected {profile.expected_split_sizes}, got {actual_sizes}"
            )
    return training_products_raw, training_products, validation_products, split


def load_lora_model(project_root: Path) -> tuple[Any, Any, Any, Any]:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    model_path = project_root / "models" / "google" / "gemma-4-E4B-it"
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    processor.tokenizer.padding_side = "right"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base_model = AutoModelForMultimodalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map="auto",
        local_files_only=True,
    )
    base_model.config.use_cache = False
    base_model.enable_input_require_grads()
    model = get_peft_model(
        base_model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            target_modules=TARGET_MODULES,
        ),
    )
    return processor, base_model, model, dtype


def run_forward_check(
    model: Any,
    collator: Any,
    training_products: Any,
    images_directory: Path,
) -> None:
    import torch

    sample = next(
        (
            product
            for product in training_products.to_dict(orient="records")
            if find_existing_product_images(product, images_directory)
        ),
        None,
    )
    if sample is None:
        raise ValueError("No training product with an existing image was found")
    batch = collator([sample])
    if hasattr(batch, "to"):
        batch = batch.to(model.device)
    else:
        batch = {
            key: value.to(model.device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
    model.eval()
    with torch.inference_mode():
        output = model(**batch)
    if output.loss is None or not bool(torch.isfinite(output.loss).item()):
        raise RuntimeError("LoRA forward check returned a non-finite loss")
    if not bool(torch.isfinite(output.logits[:, -1, :]).all().item()):
        raise RuntimeError("LoRA forward check returned non-finite logits")
    print(f"Forward check passed: loss={float(output.loss):.6f}", flush=True)


def build_training_parameters() -> dict[str, Any]:
    return {
        "epochs": TRAINING_EPOCHS,
        "batch_size": 1,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": TARGET_MODULES,
        "minority_train_share": MINORITY_CLASS_TARGET_SHARE,
    }


def run_category_training(
    profile: TrainingProfile,
    project_root: Path,
    images_directory: Path,
    *,
    check_only: bool,
    select_images: ImageSelector,
    load_image: ImageLoader,
    build_content: ContentBuilder = build_default_training_content,
    post_training_evaluator: PostTrainingEvaluator | None = None,
    additional_inputs: dict[str, Any] | None = None,
) -> None:
    import torch
    from transformers import Trainer, TrainingArguments

    project_root = project_root.expanduser().resolve()
    images_directory = images_directory.expanduser().resolve()
    random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    torch.cuda.manual_seed_all(TRAINING_SEED)

    category_products, resolved_revision, data_provenance = load_validated_inputs(
        project_root, images_directory, profile
    )
    training_raw, training_products, validation_products, split = create_data_split(
        category_products, profile
    )
    print(
        f"{profile.category}: train={len(training_raw):,}, "
        f"oversampled={len(training_products):,}, validation={len(validation_products):,}",
        flush=True,
    )
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    output_directory = (
        project_root
        / "artifacts"
        / "gemma_lora"
        / profile.experiment_slug
        / "adapters"
        / f"gemma_e4b_{profile.adapter_slug}"
    )
    final_directory = output_directory / "final"
    if not check_only and output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_directory}")

    processor = base_model = model = trainer = None
    try:
        processor, base_model, model, dtype = load_lora_model(project_root)
        model.print_trainable_parameters()
        collator = create_training_collator(
            processor,
            profile,
            images_directory,
            select_images,
            load_image,
            build_content,
        )
        if check_only:
            run_forward_check(model, collator, training_products, images_directory)
            print(f"{profile.category}: check-only complete; no training performed", flush=True)
            return

        output_directory.mkdir(parents=True, exist_ok=False)
        write_json_atomic(output_directory / "split.json", split)
        validation_products[["id", "label"]].to_csv(
            output_directory / "validation_ids.csv", index=False
        )
        training_arguments = TrainingArguments(
            output_dir=str(output_directory),
            num_train_epochs=TRAINING_EPOCHS,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            warmup_steps=30,
            lr_scheduler_type="cosine",
            bf16=dtype == torch.bfloat16,
            fp16=dtype == torch.float16,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_strategy="epoch" if profile.evaluate_each_epoch else "no",
            save_strategy="epoch",
            save_total_limit=profile.checkpoint_limit,
            prediction_loss_only=profile.evaluate_each_epoch,
            eval_accumulation_steps=1 if profile.evaluate_each_epoch else None,
            logging_steps=10,
            torch_empty_cache_steps=10,
            remove_unused_columns=False,
            report_to="none",
            seed=TRAINING_SEED,
            data_seed=TRAINING_SEED,
        )
        trainer = Trainer(
            model=model,
            args=training_arguments,
            data_collator=collator,
            train_dataset=ProductTrainingDataset(training_products),
            eval_dataset=(
                ProductTrainingDataset(validation_products) if profile.evaluate_each_epoch else None
            ),
        )
        trainer.train()
        model.save_pretrained(final_directory)
        processor.save_pretrained(final_directory)

        inference = profile.default_inference
        validation_result: dict[str, Any] | None = None
        if post_training_evaluator is not None:
            validation_result = post_training_evaluator(
                model,
                processor,
                validation_products,
                images_directory,
                output_directory,
            )
            inference = validation_result["inference"]
        if inference is None:
            raise RuntimeError(f"No inference contract produced for {profile.category}")

        required_adapter_files = (
            final_directory / "adapter_config.json",
            final_directory / "adapter_model.safetensors",
        )
        missing = [str(path) for path in required_adapter_files if not path.is_file()]
        if missing:
            raise RuntimeError(f"Incomplete adapter output: {missing}")
        adapter_hashes = {
            path.name: calculate_sha256(path)
            for path in sorted(final_directory.iterdir())
            if path.is_file()
        }
        category_result = {
            "category": profile.category,
            "experiment": profile.experiment_slug,
            "adapter": str(final_directory.relative_to(project_root)),
            "files": adapter_hashes,
            "split": split,
            "inference": inference,
            "training_image_profile": profile.training_image_profile,
        }
        if additional_inputs:
            category_result["additional_inputs"] = additional_inputs
        if validation_result is not None:
            category_result["validation"] = validation_result["validation"]
            evaluation_files = (
                output_directory / "selection.json",
                output_directory / "validation_predictions.csv",
            )
            missing_evaluation_files = [
                str(path) for path in evaluation_files if not path.is_file()
            ]
            if missing_evaluation_files:
                raise RuntimeError(f"Incomplete validation output: {missing_evaluation_files}")
            category_result["validation_artifacts"] = {
                path.name: calculate_sha256(path) for path in evaluation_files
            }
        result_manifest = {
            "schema": "e_cup.gemma_lora_category_training",
            "schema_version": 1,
            "model_id": MODEL_ID,
            "base_model_revision": resolved_revision,
            "data": data_provenance,
            "seed": TRAINING_SEED,
            "parameters": build_training_parameters(),
            "result": category_result,
        }
        write_json_atomic(output_directory / "training_result.json", result_manifest)
        print(f"Saved adapter: {final_directory}", flush=True)
    finally:
        del trainer, model, base_model, processor
        torch.cuda.empty_cache()
