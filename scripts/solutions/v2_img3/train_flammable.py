"""Train and calibrate the v2-img3 Gemma LoRA adapter for flammable goods."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from scripts.common.training.evaluation import add_prediction_columns
from scripts.common.training.lora_training import (
    TrainingProfile,
    find_existing_product_images,
    parse_training_arguments,
    run_category_training,
    score_validation_images,
    select_best_threshold,
    write_json_atomic,
)

MAX_IMAGE_SIDE = 1024
EVALUATION_BATCH_SIZE = 2
HISTORICAL_THRESHOLD = 0.005

PROFILE = TrainingProfile(
    category="Легковоспламеняющиеся",
    experiment_slug="v2_img3",
    adapter_slug="flammable",
    classification_rule=(
        "1: самостоятельный источник воспламенения, горючее вещество/газ "
        "или такой товар входит в комплект. 0: пустое устройство для работы "
        "с огнём, встроенный поджиг, горючий компонент другого изделия или "
        "легковоспламеняющийся предмет не входит в комплект."
    ),
    validation_fraction=0.20,
    evaluate_each_epoch=False,
    checkpoint_limit=2,
    training_image_profile={
        "images_per_example": 1,
        "selection": "first_with_probability_0.5_else_random_from_first_three",
        "first_image_probability": 0.5,
        "exif_transpose": True,
        "max_image_side": MAX_IMAGE_SIDE,
    },
    expected_split_sizes={
        "train_raw": 3160,
        "train_oversampled": 4352,
        "validation": 791,
    },
)


def select_training_image(product: dict[str, Any], images_directory: Path) -> list[Path]:
    image_paths = find_existing_product_images(product, images_directory, limit=3)
    if not image_paths:
        return []
    selected = image_paths[0] if random.random() < 0.5 else random.choice(image_paths)
    return [selected]


def load_training_image(image_path: Path) -> Any:
    from PIL import Image, ImageOps

    with Image.open(image_path) as source_image:
        image = ImageOps.exif_transpose(source_image).convert("RGB")
    if max(image.size) > MAX_IMAGE_SIDE:
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
    return image


def evaluate_flammable_adapter(
    model: Any,
    processor: Any,
    validation_products: Any,
    images_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
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
        max_images=3,
        batch_size=EVALUATION_BATCH_SIZE,
    )
    first_image_metrics = select_best_threshold(predictions["label"], predictions["p_img1"])
    max_three_metrics = select_best_threshold(predictions["label"], predictions["p_max"])
    predictions = add_prediction_columns(
        predictions,
        "p_max",
        float(max_three_metrics["threshold"]),
    )
    predictions.to_csv(output_directory / "validation_predictions.csv", index=False)
    selection = {
        "selected_mode": "max_first_three",
        "threshold_policy": "validation_f1_then_precision_then_threshold",
        "first_image": first_image_metrics,
        "max_first_three": max_three_metrics,
        "historical_threshold": HISTORICAL_THRESHOLD,
        "matches_historical_threshold": (max_three_metrics["threshold"] == HISTORICAL_THRESHOLD),
    }
    write_json_atomic(output_directory / "selection.json", selection)
    return {
        "inference": {
            "max_images": 3,
            "aggregation": "max",
            "threshold": max_three_metrics["threshold"],
        },
        "validation": selection,
    }


def main() -> None:
    args = parse_training_arguments(__doc__ or "Train flammable-goods LoRA")
    run_category_training(
        PROFILE,
        args.project_root,
        args.images_dir,
        check_only=args.check_only,
        select_images=select_training_image,
        load_image=load_training_image,
        post_training_evaluator=evaluate_flammable_adapter,
    )


if __name__ == "__main__":
    main()
