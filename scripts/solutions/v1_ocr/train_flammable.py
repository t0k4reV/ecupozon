"""Train and calibrate the v1 OCR Gemma LoRA adapter for flammable goods."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from scripts.common.training.evaluation import add_prediction_columns
from scripts.common.training.lora_training import (
    TrainingProfile,
    calculate_sha256,
    find_existing_product_images,
    get_default_project_root,
    run_category_training,
    score_validation_images,
    select_best_threshold,
    write_json_atomic,
)
from scripts.common.training.ocr_training import build_ocr_training_content, load_ocr_cache

MAX_IMAGE_SIDE = 1024
EVALUATION_BATCH_SIZE = 2

PROFILE = TrainingProfile(
    category="Легковоспламеняющиеся",
    experiment_slug="v1_ocr",
    adapter_slug="flammable_ocr",
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
        "ocr": "easyocr_ru_en_for_selected_image",
        "max_ocr_chars": 2000,
    },
    expected_split_sizes={
        "train_raw": 3160,
        "train_oversampled": 4352,
        "validation": 791,
    },
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=get_default_project_root())
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--ocr-cache", type=Path, required=True)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run one forward pass without training or writing an adapter.",
    )
    return parser.parse_args()


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


def main() -> None:
    args = parse_arguments()
    project_root = args.project_root.expanduser().resolve()
    ocr_cache_path = args.ocr_cache.expanduser().resolve()
    ocr_cache = load_ocr_cache(ocr_cache_path)

    def build_content(
        product: dict[str, Any],
        profile: TrainingProfile,
        selected_image_paths: list[Path],
    ) -> list[dict[str, str]]:
        return build_ocr_training_content(
            product,
            profile,
            selected_image_paths,
            ocr_cache,
        )

    def evaluate_adapter(
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
            build_content=build_content,
            max_images=1,
            batch_size=EVALUATION_BATCH_SIZE,
        )
        selection = select_best_threshold(predictions["label"], predictions["p_img1"])
        predictions = add_prediction_columns(
            predictions,
            "p_img1",
            float(selection["threshold"]),
        )
        predictions.to_csv(output_directory / "validation_predictions.csv", index=False)
        validation = {
            "selected_mode": "single_first_image_with_ocr",
            "threshold_policy": "validation_f1_then_precision_then_threshold",
            **selection,
        }
        write_json_atomic(output_directory / "selection.json", validation)
        return {
            "inference": {
                "max_images": 1,
                "aggregation": "single",
                "threshold": selection["threshold"],
            },
            "validation": validation,
        }

    run_category_training(
        PROFILE,
        project_root,
        args.images_dir,
        check_only=args.check_only,
        select_images=select_training_image,
        load_image=load_training_image,
        build_content=build_content,
        post_training_evaluator=evaluate_adapter,
        additional_inputs={
            "ocr_cache": (
                str(ocr_cache_path.relative_to(project_root))
                if project_root in ocr_cache_path.parents
                else ocr_cache_path.name
            ),
            "ocr_cache_sha256": calculate_sha256(ocr_cache_path),
            "ocr_records": len(ocr_cache),
        },
    )


if __name__ == "__main__":
    main()
