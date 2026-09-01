"""Train the Gemma LoRA adapter for the supplements category."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from scripts.common.training.evaluation import add_prediction_columns, calculate_binary_metrics
from scripts.common.training.lora_training import (
    TrainingProfile,
    find_existing_product_images,
    parse_training_arguments,
    run_category_training,
    score_validation_images,
    write_json_atomic,
)

EVALUATION_BATCH_SIZE = 2
PRODUCTION_THRESHOLD = 0.7

PROFILE = TrainingProfile(
    category="БАД",
    experiment_slug="shared",
    adapter_slug="supplements",
    classification_rule=(
        "1: есть прямое указание БАД или dietary supplement. "
        "0: спортивное питание, явное отрицание принадлежности к БАД "
        "или отсутствие маркировки БАД."
    ),
    validation_fraction=0.10,
    evaluate_each_epoch=True,
    checkpoint_limit=2,
    training_image_profile={
        "images_per_example": 1,
        "selection": "random_from_all_available",
        "exif_transpose": False,
        "max_image_side": None,
    },
    default_inference={
        "max_images": 1,
        "aggregation": "single",
        "threshold": PRODUCTION_THRESHOLD,
    },
)


def select_training_image(product: dict[str, Any], images_directory: Path) -> list[Path]:
    image_paths = find_existing_product_images(product, images_directory)
    return [random.choice(image_paths)] if image_paths else []


def load_training_image(image_path: Path) -> Any:
    from PIL import Image

    with Image.open(image_path) as source_image:
        return source_image.convert("RGB")


def evaluate_supplements_adapter(
    model: Any,
    processor: Any,
    validation_products: Any,
    images_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Evaluate the fixed production policy on the supplements holdout."""
    model.eval()
    model.config.use_cache = True
    processor.tokenizer.padding_side = "left"
    predictions = score_validation_images(
        model=model,
        processor=processor,
        validation_products=validation_products,
        images_directory=images_directory,
        profile=PROFILE,
        load_image=load_training_image,
        max_images=1,
        batch_size=EVALUATION_BATCH_SIZE,
    )
    predictions = add_prediction_columns(predictions, "p_img1", PRODUCTION_THRESHOLD)
    predictions.to_csv(output_directory / "validation_predictions.csv", index=False)
    metrics = calculate_binary_metrics(
        predictions["label"],
        predictions["probability"],
        PRODUCTION_THRESHOLD,
    )
    validation = {
        "selected_mode": "single_first_image",
        "threshold_policy": "fixed",
        "metrics": metrics,
    }
    write_json_atomic(output_directory / "selection.json", validation)
    return {
        "inference": dict(PROFILE.default_inference or {}),
        "validation": validation,
    }


def main() -> None:
    args = parse_training_arguments(__doc__ or "Train supplements LoRA")
    run_category_training(
        PROFILE,
        args.project_root,
        args.images_dir,
        check_only=args.check_only,
        select_images=select_training_image,
        load_image=load_training_image,
        post_training_evaluator=evaluate_supplements_adapter,
    )


if __name__ == "__main__":
    main()
