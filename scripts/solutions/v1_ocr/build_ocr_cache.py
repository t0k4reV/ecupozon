"""Build the deterministic EasyOCR cache used by the v1 OCR training profile."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.common.solution_profile import load_solution_profile
from scripts.common.training.lora_training import get_default_project_root
from scripts.common.training.ocr_training import normalize_ocr_text

FLAMMABLE_CATEGORY = "Легковоспламеняющиеся"
EASYOCR_MODEL_HASHES = {
    "craft_mlt_25k.pth": "4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17",
    "cyrillic_g2.pth": "48d0f3b58f28aa64651ab1032cc2d498c4de25135829668e87c14e7a07529f29",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=get_default_project_root())
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run EasyOCR on one image without writing a cache.",
    )
    return parser.parse_args()


def calculate_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_detections(
    detections: list[Any], min_confidence: float = 0.25
) -> tuple[str, list[float]]:
    recognized_lines: list[tuple[float, float, str, float]] = []
    for detection in detections:
        if not isinstance(detection, (list, tuple)) or len(detection) < 3:
            continue
        bounding_box, text, confidence = (
            detection[0],
            normalize_ocr_text(detection[1]),
            float(detection[2]),
        )
        if not text or confidence < min_confidence:
            continue
        try:
            left = min(float(point[0]) for point in bounding_box)
            top = min(float(point[1]) for point in bounding_box)
        except (TypeError, ValueError, IndexError):
            left = top = 0.0
        recognized_lines.append((top, left, text, confidence))
    recognized_lines.sort(key=lambda detection: (round(detection[0] / 20), detection[1]))
    return (
        " ".join(detection[2] for detection in recognized_lines),
        [detection[3] for detection in recognized_lines],
    )


def main() -> None:
    args = parse_arguments()
    import easyocr
    import pandas as pd
    import torch

    project_root = args.project_root.expanduser().resolve()
    images_directory = args.images_dir.expanduser().resolve()
    manifest_path = project_root / "artifacts" / "gemma_lora" / "data" / "products_manifest.jsonl"
    models_directory = project_root / "artifacts" / "easyocr"
    profile_path = Path(__file__).resolve().with_name("profile.json")
    solution_profile = load_solution_profile(profile_path)
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else project_root / solution_profile["training"]["ocr_cache"]
    )
    if not images_directory.is_dir():
        raise FileNotFoundError(images_directory)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for EasyOCR cache generation")
    for filename, expected_hash in EASYOCR_MODEL_HASHES.items():
        model_path = models_directory / filename
        if not model_path.is_file() or calculate_sha256(model_path) != expected_hash:
            raise ValueError(f"EasyOCR model is missing or invalid: {model_path}")

    products = pd.read_json(manifest_path, lines=True, dtype={"id": "string"})
    products = products.loc[products["category"].eq(FLAMMABLE_CATEGORY)]
    image_records: list[tuple[str, str, Path]] = []
    for product in products.itertuples(index=False):
        relative_paths = product.image_paths if isinstance(product.image_paths, list) else []
        for relative_path in relative_paths[:3]:
            image_path = (images_directory / str(relative_path)).resolve()
            if images_directory != image_path and images_directory not in image_path.parents:
                raise ValueError(f"Unsafe image path in manifest: {relative_path}")
            if image_path.is_file():
                image_records.append((str(product.id), str(relative_path), image_path))
    if not image_records:
        raise ValueError("No flammable-goods images were found for OCR")

    reader = easyocr.Reader(
        ["ru", "en"],
        gpu="cuda",
        verbose=False,
        model_storage_directory=str(models_directory),
        download_enabled=False,
        cudnn_benchmark=False,
    )
    if args.check_only:
        product_id, relative_path, image_path = image_records[0]
        detections = reader.readtext(
            str(image_path),
            detail=1,
            paragraph=False,
            decoder="greedy",
            batch_size=128,
            workers=4,
            canvas_size=2560,
            mag_ratio=1.0,
        )
        text, _ = normalize_detections(detections)
        print(
            json.dumps(
                {"id": product_id, "image_path": relative_path, "ocr_chars": len(text)},
                ensure_ascii=False,
            )
        )
        return

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if output_path.exists() or temporary_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_path} / {temporary_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary_path.open("w", encoding="utf-8") as stream:
            for index, (product_id, relative_path, image_path) in enumerate(image_records, start=1):
                error_message = ""
                try:
                    detections = reader.readtext(
                        str(image_path),
                        detail=1,
                        paragraph=False,
                        decoder="greedy",
                        batch_size=128,
                        workers=4,
                        canvas_size=2560,
                        mag_ratio=1.0,
                    )
                    ocr_text, scores = normalize_detections(detections)
                except Exception as error:
                    ocr_text, scores = "", []
                    error_message = f"{type(error).__name__}: {error}"
                stream.write(
                    json.dumps(
                        {
                            "id": product_id,
                            "image_path": relative_path,
                            "ocr": ocr_text,
                            "scores": scores,
                            "error": error_message,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if index % 100 == 0 or index == len(image_records):
                    print(f"EasyOCR: {index:,}/{len(image_records):,}", flush=True)
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    print(f"OCR cache: {output_path} ({len(image_records):,} images)")


if __name__ == "__main__":
    main()
