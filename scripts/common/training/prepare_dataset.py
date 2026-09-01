"""Deduplicate and validate data.csv, then create the training dataset manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

REQUIRED_SOURCE_COLUMNS = {"id", "name", "description", "category", "label"}
DEDUPLICATION_COLUMNS = ["name", "description", "category"]
SUPPORTED_CATEGORIES = {"БАД", "Легковоспламеняющиеся"}
BINARY_LABELS = {0, 1}
TRAINING_MANIFEST_COLUMNS = [
    "id",
    "name",
    "description",
    "category",
    "label",
    "image_paths",
    "image_count",
    "has_images",
]


def get_default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=get_default_project_root())
    parser.add_argument("--images-dir", type=Path, required=True)
    return parser.parse_args()


def calculate_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_deduplicate_dataset(
    source_csv: Path,
    deduplicated_csv: Path,
) -> tuple[pd.DataFrame, int]:
    import pandas as pd

    if not source_csv.is_file():
        raise FileNotFoundError(f"CSV not found: {source_csv}")
    if source_csv.resolve() == deduplicated_csv.resolve():
        raise ValueError("Source and deduplicated CSV paths must differ")
    if deduplicated_csv.exists() and not deduplicated_csv.is_file():
        raise ValueError(f"CSV output path is not a file: {deduplicated_csv}")

    try:
        products = pd.read_csv(source_csv)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"CSV is empty: {source_csv}") from error

    missing_columns = REQUIRED_SOURCE_COLUMNS - set(products.columns)
    if missing_columns:
        raise ValueError(f"Missing CSV columns: {sorted(missing_columns)}")

    deduplicated_products = products.drop_duplicates(
        subset=DEDUPLICATION_COLUMNS,
        keep="first",
    ).copy()
    deduplicated_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = deduplicated_csv.with_suffix(deduplicated_csv.suffix + ".tmp")
    temporary_csv.unlink(missing_ok=True)
    try:
        deduplicated_products.to_csv(temporary_csv, index=False, encoding="utf-8-sig")
        temporary_csv.replace(deduplicated_csv)
    except Exception:
        temporary_csv.unlink(missing_ok=True)
        raise
    print(
        f"Deduplicated data: {len(products):,} -> {len(deduplicated_products):,} rows; "
        f"output: {deduplicated_csv}"
    )
    return deduplicated_products, len(products)


def main() -> None:
    args = parse_arguments()
    import pandas as pd

    project_root = args.project_root.expanduser().resolve()
    images_directory = args.images_dir.expanduser().resolve()
    source_csv_path = project_root / "data.csv"
    deduplicated_csv_path = project_root / "data_no_duplicates.csv"
    manifest_path = project_root / "artifacts" / "gemma_lora" / "data" / "products_manifest.jsonl"

    if not images_directory.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_directory}")

    products, source_row_count = load_and_deduplicate_dataset(
        source_csv_path,
        deduplicated_csv_path,
    )
    products["id"] = (
        products["id"].astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
    )
    products["name"] = products["name"].fillna("").astype(str).str.strip()
    products["description"] = products["description"].fillna("").astype(str).str.strip()
    products["category"] = products["category"].astype(str).str.strip()
    labels = pd.to_numeric(products["label"], errors="raise")

    if products["id"].isna().any() or products["id"].eq("").any():
        raise ValueError("CSV contains empty ids")
    unsafe_ids = products.loc[
        products["id"].isin({".", ".."}) | products["id"].str.contains(r"[/\\]", regex=True),
        "id",
    ].tolist()[:10]
    if unsafe_ids:
        raise ValueError(f"CSV contains ids unsafe for filesystem paths: {unsafe_ids}")
    if products["id"].duplicated().any():
        duplicates = products.loc[products["id"].duplicated(), "id"].tolist()[:10]
        raise ValueError(f"CSV contains duplicate ids: {duplicates}")
    if labels.isna().any() or not labels.isin(BINARY_LABELS).all():
        invalid = labels.loc[~labels.isin(BINARY_LABELS)].unique().tolist()
        raise ValueError(f"Only binary labels 0/1 are allowed: {invalid}")
    products["label"] = labels.astype(int)

    categories = set(products["category"])
    if categories != SUPPORTED_CATEGORIES:
        raise ValueError(
            f"Expected exactly {sorted(SUPPORTED_CATEGORIES)}, got {sorted(categories)}"
        )
    for category, category_products in products.groupby("category"):
        if set(category_products["label"]) != BINARY_LABELS:
            raise ValueError(f"Category {category!r} must contain both labels")

    def find_product_images(product_id: str) -> list[str]:
        product_image_directory = images_directory / product_id
        if not product_image_directory.is_dir():
            return []
        return sorted(
            image_path.relative_to(images_directory).as_posix()
            for image_path in product_image_directory.iterdir()
            if image_path.is_file() and image_path.suffix.lower() == ".jpg"
        )

    products["image_paths"] = products["id"].map(find_product_images)
    products["image_count"] = products["image_paths"].str.len()
    products["has_images"] = products["image_count"].gt(0)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_suffix(".jsonl.tmp")
    temporary_manifest.unlink(missing_ok=True)
    products[TRAINING_MANIFEST_COLUMNS].to_json(
        temporary_manifest,
        orient="records",
        lines=True,
        force_ascii=False,
    )
    temporary_manifest.replace(manifest_path)

    metadata_path = manifest_path.with_name("products_manifest.meta.json")
    metadata = {
        "schema": "e_cup.products_manifest",
        "schema_version": 1,
        "source_csv": source_csv_path.name,
        "source_csv_sha256": calculate_sha256(source_csv_path),
        "deduplicated_csv": deduplicated_csv_path.name,
        "deduplicated_csv_sha256": calculate_sha256(deduplicated_csv_path),
        "source_rows": source_row_count,
        "manifest_sha256": calculate_sha256(manifest_path),
        "rows": len(products),
        "products_with_images": int(products["has_images"].sum()),
        "images": int(products["image_count"].sum()),
        "categories": {
            category: {
                str(label): int(count)
                for label, count in category_products["label"].value_counts().sort_index().items()
            }
            for category, category_products in products.groupby("category")
        },
    }
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_metadata.replace(metadata_path)
    print(f"Manifest: {manifest_path}")
    print(f"Rows: {len(products):,}; images: {int(products['image_count'].sum()):,}")


if __name__ == "__main__":
    main()
