"""Run inference and produce a competition-compatible submission CSV."""

from __future__ import annotations

import argparse
import gc
import os
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from .comments import build_submission_comments, validate_comment
from .core import (
    ADAPTER_NAME_BY_CATEGORY,
    FLAMMABLE_CATEGORY,
    SUBMISSION_RESULT_PATTERN,
    SUPPLEMENTS_CATEGORY,
    SUPPORTED_CATEGORIES,
    get_submission_root,
    load_submission_config,
    load_test_data,
    log_event,
    max_images_for_category,
    resolve_shared_model_path,
)
from .inference import get_label_token_ids, predict_category_labels
from .ocr import extract_ocr_texts


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test_data_path",
        "--test-data-path",
        "-i",
        dest="test_data_path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        "--output-path",
        "-o",
        dest="output_path",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def format_submission_result(label: int, comment: str) -> str:
    comment = validate_comment(comment)
    verdict = "не бан" if int(label) == 1 else "бан"
    result = f"<комментарий>{comment}<вердикт>{verdict}"
    match = SUBMISSION_RESULT_PATTERN.fullmatch(result)
    if match is None or not 50 <= len(match.group(1)) <= 300:
        raise ValueError(f"Invalid competition result: {result!r}")
    return result


def write_csv_atomically(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            dataframe.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def validate_submission_output(source_products: pd.DataFrame, submission: pd.DataFrame) -> None:
    if submission.columns.tolist() != ["id", "result"]:
        raise ValueError("Output must contain exactly id,result columns")
    if (
        submission["id"].duplicated().any()
        or submission["id"].tolist() != source_products["id"].tolist()
    ):
        raise ValueError("Output ids must match input ids and order")
    for row in submission.itertuples(index=False):
        match = SUBMISSION_RESULT_PATTERN.fullmatch(str(row.result))
        if match is None or not 50 <= len(match.group(1)) <= 300:
            raise ValueError(f"Invalid submission for id={row.id}")
        if any(symbol in match.group(1) for symbol in ("\n", "\r", "<", ">")):
            raise ValueError(f"Forbidden comment character for id={row.id}")


def get_inference_batch_size(submission_config: Mapping[str, Any]) -> int:
    return max(1, int(os.environ.get("ECUP_BATCH_SIZE", submission_config.get("batch_size", 2))))


def run_submission_pipeline(
    test_data_path: Path,
    output_path: Path,
) -> None:
    started_at = time.perf_counter()
    submission_directory = get_submission_root()
    submission_config = load_submission_config(submission_directory)
    test_products = load_test_data(test_data_path)
    if test_products.empty:
        submission = pd.DataFrame(
            {
                "id": test_products["id"].astype("string"),
                "result": pd.Series([], dtype="string"),
            }
        )
        write_csv_atomically(submission, output_path)
        return

    import torch
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    torch.manual_seed(int(submission_config.get("seed", 2026)))
    torch.backends.cudnn.benchmark = False

    images_directory = test_data_path.parent / "images"
    ocr_text_by_path = extract_ocr_texts(
        test_products=test_products,
        images_directory=images_directory,
        submission_directory=submission_directory,
        submission_config=submission_config,
    )

    model_directory = resolve_shared_model_path(str(submission_config["model_id"]))
    adapter_directory_by_category = {
        category: submission_directory / relative
        for category, relative in submission_config["adapters"].items()
    }
    for adapter_directory in adapter_directory_by_category.values():
        if not (adapter_directory / "adapter_config.json").is_file():
            raise FileNotFoundError(f"LoRA adapter is incomplete: {adapter_directory}")
    processor = AutoProcessor.from_pretrained(model_directory, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    log_event(
        "model_loading", expert="gemma", model_directory=str(model_directory), dtype=str(dtype)
    )
    base_model = AutoModelForMultimodalLM.from_pretrained(
        model_directory,
        dtype=dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    base_model.config.use_cache = True
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_directory_by_category[SUPPLEMENTS_CATEGORY]),
        adapter_name=ADAPTER_NAME_BY_CATEGORY[SUPPLEMENTS_CATEGORY],
        local_files_only=True,
    )
    model.load_adapter(
        str(adapter_directory_by_category[FLAMMABLE_CATEGORY]),
        adapter_name=ADAPTER_NAME_BY_CATEGORY[FLAMMABLE_CATEGORY],
        local_files_only=True,
    )
    model.eval()
    label_token_ids = get_label_token_ids(processor)
    predicted_labels_by_index: dict[int, int] = {}
    for category in SUPPORTED_CATEGORIES:
        category_products = test_products.loc[test_products["category"].eq(category)]
        if category_products.empty:
            continue
        category_labels = predict_category_labels(
            category_products=category_products,
            category=category,
            adapter_name=ADAPTER_NAME_BY_CATEGORY[category],
            threshold=float(submission_config["thresholds"][category]),
            model=model,
            processor=processor,
            label_token_ids=label_token_ids,
            images_directory=images_directory,
            batch_size=get_inference_batch_size(submission_config),
            submission_config=submission_config,
            max_images=max_images_for_category(submission_config, category),
            ocr_text_by_path=ocr_text_by_path,
        )
        predicted_labels_by_index.update(
            zip(category_products.index.tolist(), category_labels, strict=True)
        )

    del model, base_model, processor
    gc.collect()
    torch.cuda.empty_cache()

    comments_started_at = time.perf_counter()
    comments_by_index, comment_stats = build_submission_comments(
        test_products=test_products,
        predicted_labels_by_index=predicted_labels_by_index,
        images_directory=images_directory,
        ocr_text_by_path=ocr_text_by_path,
        submission_config=submission_config,
    )
    comment_generation_seconds = time.perf_counter() - comments_started_at

    formatted_results = [
        format_submission_result(predicted_labels_by_index[index], comments_by_index[index])
        for index in test_products.index
    ]
    submission = pd.DataFrame(
        {"id": test_products["id"].astype("string"), "result": formatted_results}
    )
    validate_submission_output(test_products, submission)
    write_csv_atomically(submission, output_path)
    log_event(
        "comment_generation_complete",
        **comment_stats,
        seconds=round(comment_generation_seconds, 3),
    )
    log_event(
        "run_complete",
        rows=len(test_products),
        ban=sum(value == 0 for value in predicted_labels_by_index.values()),
        not_ban=sum(value == 1 for value in predicted_labels_by_index.values()),
        seconds=round(time.perf_counter() - started_at, 3),
    )


def main() -> None:
    args = parse_arguments()
    run_submission_pipeline(args.test_data_path, args.output_path)
