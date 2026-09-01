"""Gemma LoRA classification. This module is the only source of verdict labels."""

from __future__ import annotations

import gc
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from .core import CLASSIFICATION_RULES, find_product_image_paths, log_event


def build_classification_prompt(
    product: Any,
    max_description_chars: int,
    *,
    include_ocr_warning: bool,
) -> str:
    ocr_warning = "OCR каждой фотографии может содержать ошибки. " if include_ocr_warning else ""
    return (
        f"Категория: {product.category}\n"
        f"Правила: {CLASSIFICATION_RULES[product.category]}\n"
        f"Название: {product.name}\n"
        f"Описание: {product.description[:max_description_chars]}\n"
        f"{ocr_warning}Определи соответствие правилам. Ответь только цифрой 0 или 1."
    )


def load_product_image(image_path: Path | None, product_id: str, max_image_side: int):
    from PIL import Image, ImageOps

    if image_path is None:
        return None
    try:
        with Image.open(image_path) as source_image:
            image = ImageOps.exif_transpose(source_image).convert("RGB")
        if max_image_side > 0 and max(image.size) > max_image_side:
            image.thumbnail((max_image_side, max_image_side), Image.Resampling.LANCZOS)
        return image
    except (OSError, ValueError) as error:
        log_event(
            "image_skipped",
            id=product_id,
            image_path=image_path.name,
            error=f"{type(error).__name__}: {error}",
        )
        return None


def build_model_conversations(
    *,
    image_variants: list[tuple[int, Any, int | None, Path | None]],
    max_description_chars: int,
    max_image_side: int,
    ocr_text_by_path: Mapping[str, str],
    ocr_classification_categories: set[str],
) -> list[list[dict[str, Any]]]:
    conversations: list[list[dict[str, Any]]] = []
    for _, product, photo_number, image_path in image_variants:
        content: list[dict[str, Any]] = []
        include_ocr = str(product.category) in ocr_classification_categories
        image = load_product_image(image_path, str(product.id), max_image_side)
        if image is not None:
            content.append({"type": "image", "image": image})
            if include_ocr and image_path is not None:
                ocr_text = str(ocr_text_by_path.get(str(image_path.resolve()), "")).strip()
                content.append(
                    {
                        "type": "text",
                        "text": (
                            f"OCR изображения {photo_number}: {ocr_text or 'текст не распознан'}"
                        ),
                    }
                )
        content.append(
            {
                "type": "text",
                "text": build_classification_prompt(
                    product,
                    max_description_chars,
                    include_ocr_warning=include_ocr,
                ),
            }
        )
        conversations.append([{"role": "user", "content": content}])
    return conversations


def prepare_model_inputs(processor: Any, conversations: list[list[dict[str, Any]]]):
    template_arguments = {
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        return processor.apply_chat_template(
            conversations,
            processor_kwargs={"padding": True},
            **template_arguments,
        )
    except TypeError as error:
        if "processor_kwargs" not in str(error):
            raise
        return processor.apply_chat_template(conversations, padding=True, **template_arguments)


def get_label_token_ids(processor: Any) -> tuple[int, int]:
    zero_token_ids = processor.tokenizer("0", add_special_tokens=False).input_ids
    one_token_ids = processor.tokenizer("1", add_special_tokens=False).input_ids
    if len(zero_token_ids) != 1 or len(one_token_ids) != 1:
        raise RuntimeError(f"Labels must be single tokens: 0={zero_token_ids}, 1={one_token_ids}")
    return int(zero_token_ids[0]), int(one_token_ids[0])


def predict_category_probabilities(
    *,
    category_products: pd.DataFrame,
    category: str,
    adapter_name: str,
    model: Any,
    processor: Any,
    label_token_ids: tuple[int, int],
    images_directory: Path,
    batch_size: int,
    submission_config: Mapping[str, Any],
    max_images: int,
    ocr_text_by_path: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Return production probabilities and per-photo scores in input row order."""
    import torch

    products = list(category_products.itertuples(index=False))
    image_variants: list[tuple[int, Any, int | None, Path | None]] = []
    for row_index, product in enumerate(products):
        image_paths = find_product_image_paths(images_directory, str(product.id), max_images)
        if image_paths:
            image_variants.extend(
                (row_index, product, number, image_path) for number, image_path in image_paths
            )
        else:
            image_variants.append((row_index, product, None, None))
    probabilities_by_product: list[list[tuple[int | None, float]]] = [[] for _ in products]
    ocr_settings = submission_config.get("ocr", {})
    ocr_classification_categories = {
        str(value)
        for value in (
            ocr_settings.get("classification_categories", [])
            if isinstance(ocr_settings, Mapping)
            else []
        )
    }
    model.set_adapter(adapter_name)
    current_batch_size = max(1, batch_size)
    next_variant_index = 0

    while next_variant_index < len(image_variants):
        batch_variants = image_variants[
            next_variant_index : next_variant_index + current_batch_size
        ]
        conversations = None
        model_inputs = None
        generation_output = None
        try:
            conversations = build_model_conversations(
                image_variants=batch_variants,
                max_description_chars=int(submission_config.get("max_description_chars", 8000)),
                max_image_side=int(submission_config.get("max_image_side", 1024)),
                ocr_text_by_path=ocr_text_by_path,
                ocr_classification_categories=ocr_classification_categories,
            )
            model_inputs = prepare_model_inputs(processor, conversations).to(model.device)
            with torch.inference_mode():
                generation_output = model.generate(
                    **model_inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            class_logits = generation_output.scores[0][:, list(label_token_ids)].float()
            positive_class_probabilities = (
                class_logits.softmax(dim=-1)[:, 1].detach().cpu().tolist()
            )
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            del conversations, model_inputs, generation_output
            gc.collect()
            torch.cuda.empty_cache()
            if current_batch_size == 1:
                raise
            current_batch_size = max(1, current_batch_size // 2)
            log_event(
                "batch_reduced_after_oom",
                category=category,
                batch_size=current_batch_size,
            )
            continue

        for (row_index, _, photo_number, _), probability in zip(
            batch_variants, positive_class_probabilities, strict=True
        ):
            probabilities_by_product[row_index].append((photo_number, float(probability)))
        next_variant_index += len(batch_variants)
        log_event(
            "prediction_progress",
            category=category,
            done=next_variant_index,
            total=len(image_variants),
            batch_size=current_batch_size,
        )
        del conversations, model_inputs, generation_output, class_logits

    results: list[dict[str, Any]] = []
    for image_probabilities in probabilities_by_product:
        selected_photo, probability = max(image_probabilities, key=lambda item: item[1])
        results.append(
            {
                "probability": probability,
                "selected_photo": selected_photo,
                "image_probabilities": image_probabilities,
            }
        )
    return results


def predict_category_labels(
    *,
    category_products: pd.DataFrame,
    category: str,
    adapter_name: str,
    threshold: float,
    model: Any,
    processor: Any,
    label_token_ids: tuple[int, int],
    images_directory: Path,
    batch_size: int,
    submission_config: Mapping[str, Any],
    max_images: int,
    ocr_text_by_path: Mapping[str, str],
) -> list[int]:
    """Return verdict labels while keeping probability scoring reusable for audits."""
    probability_results = predict_category_probabilities(
        category_products=category_products,
        category=category,
        adapter_name=adapter_name,
        model=model,
        processor=processor,
        label_token_ids=label_token_ids,
        images_directory=images_directory,
        batch_size=batch_size,
        submission_config=submission_config,
        max_images=max_images,
        ocr_text_by_path=ocr_text_by_path,
    )
    return [int(result["probability"] >= threshold) for result in probability_results]
