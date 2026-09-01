"""Build validated, deterministic comments without changing model verdicts."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .core import (
    SUPPLEMENTS_CATEGORY,
    CommentEvidence,
    find_product_image_paths,
    max_images_for_category,
    normalize_text,
)
from .rules import (
    COMMENT_SUPPLEMENTS_OCR_MARKER_PATTERNS,
    EVIDENCE_DESCRIPTION_BY_REASON,
    FLAMMABLE_ACCESSORY_ONLY_PATTERNS,
    FLAMMABLE_INCLUDED_PATTERNS,
    FLAMMABLE_NOT_INCLUDED_PATTERNS,
    OCR_SUPPLEMENTS_MARKER_PATTERNS,
    SPORT_NUTRITION_PATTERNS,
    SUPPLEMENTS_NEGATION_PATTERNS,
    detect_comment_evidence,
    expected_label_for_reason,
    extract_matching_text,
    load_raw_ocr_texts,
    matches_any_pattern,
    select_evidence_matching_prediction,
)


def _format_photo_numbers(photos: Sequence[int]) -> str:
    values = [str(value) for value in photos]
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} и {values[1]}"
    return f"{', '.join(values[:-1])} и {values[-1]}"


def _describe_evidence_location(evidence: CommentEvidence) -> str:
    if evidence.source == "IMAGE":
        return f"На фото {_format_photo_numbers(evidence.photos)}"
    if evidence.source == "BOTH":
        return f"В тексте карточки и на фото {_format_photo_numbers(evidence.photos)}"
    return "В названии или описании"


def get_deterministic_template_index(product_id: str, template_count: int) -> int:
    if template_count <= 0:
        raise ValueError("template_count must be positive")
    digest = hashlib.sha256(str(product_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % template_count


def validate_comment(comment: str) -> str:
    comment = re.sub(r"\s+", " ", str(comment)).strip()
    if any(symbol in comment for symbol in ("\n", "\r", "<", ">")):
        raise ValueError("Comment contains forbidden characters")
    if not 50 <= len(comment) <= 290:
        raise ValueError(f"Comment length is outside 50..290: {len(comment)}")
    return comment


def _truncate_product_name(name: str, max_chars: int = 78) -> str:
    value = normalize_text(name).replace("<", " ").replace(">", " ")
    value = value.replace("«", '"').replace("»", '"').strip(" .,:;—-")
    if not value:
        return "товар без указанного названия"
    if len(value) <= max_chars:
        return value
    cut_at = max(1, max_chars - 1)
    shortened = value[:cut_at].rsplit(" ", 1)[0].rstrip(" .,:;—-")
    return f"{shortened or value[:cut_at].rstrip()}…"


def _truncate_text(value: str, max_chars: int) -> str:
    """Shorten an exact text fragment at a word boundary."""
    text = normalize_text(value).replace("<", " ").replace(">", " ")
    text = text.replace("«", '"').replace("»", '"').strip(" .,:;—-")
    if len(text) <= max_chars:
        return text
    cut_at = max(1, max_chars - 1)
    shortened = text[:cut_at].rsplit(" ", 1)[0].rstrip(" .,:;—-")
    return f"{shortened or text[:cut_at].rstrip()}…"


_OCR_QUOTE_STOP_WORDS = {
    "для",
    "или",
    "при",
    "как",
    "это",
    "без",
    "под",
    "над",
    "все",
    "его",
    "она",
    "они",
    "the",
    "and",
    "with",
    "from",
    "this",
    "that",
    "made",
    "набор",
    "товар",
    "продукт",
    "штук",
    "упаковка",
}


def _extract_ocr_token_stems(value: str) -> set[str]:
    stems: set[str] = set()
    for token in re.findall(r"[A-Za-zА-Яа-яЁё]{3,}", value.casefold()):
        normalized = token.replace("ё", "е")
        if normalized in _OCR_QUOTE_STOP_WORDS:
            continue
        stems.add(normalized[:6])
    return stems


def _contains_category_ocr_signal(category: str, value: str) -> bool:
    if category == SUPPLEMENTS_CATEGORY:
        patterns = OCR_SUPPLEMENTS_MARKER_PATTERNS + SPORT_NUTRITION_PATTERNS
        return matches_any_pattern(patterns, value) or _matches_regex(
            r"\b(?:витамин\w*|минерал\w*|магни\w*|омега[-\s]?3|"
            r"коллаген\w*|капсул\w*|таблет\w*|дозиров\w*)\b",
            value,
        )
    patterns = (
        FLAMMABLE_NOT_INCLUDED_PATTERNS
        + FLAMMABLE_INCLUDED_PATTERNS
        + FLAMMABLE_ACCESSORY_ONLY_PATTERNS
    )
    if matches_any_pattern(patterns, value):
        return True
    return _matches_regex(
        r"\b(?:спич\w*|зажигалк\w*|газ\w*|баллон\w*|бензин\w*|керосин\w*|"
        r"топлив\w*|горюч\w*|розжиг\w*|свеч\w*|дым\w*|фитил\w*)\b",
        value,
    )


def select_relevant_ocr_excerpt(
    *,
    ocr_rows: Sequence[tuple[int, str]],
    category: str,
    name: str,
    description: str,
    final_label: int | None = None,
) -> tuple[int | None, str]:
    """Choose a readable, card-relevant OCR quote; return no photo for OCR noise."""
    if category == SUPPLEMENTS_CATEGORY:
        for photo_number, raw_text in ocr_rows:
            raw_value = str(raw_text)
            negation = extract_matching_text(SUPPLEMENTS_NEGATION_PATTERNS, raw_value)
            if final_label == 0:
                if negation:
                    return photo_number, normalize_text(negation)
                continue
            if negation:
                continue
            marker = extract_matching_text(COMMENT_SUPPLEMENTS_OCR_MARKER_PATTERNS, raw_value)
            if marker:
                return photo_number, normalize_text(marker)
        return None, ""

    card_stems = _extract_ocr_token_stems(f"{name} {description}")
    best: tuple[int, int, str] | None = None
    for photo_number, raw_text in ocr_rows:
        lines = [
            normalize_text(line).strip(" .,:;—-")
            for line in str(raw_text).splitlines()
            if normalize_text(line).strip(" .,:;—-")
        ][:30]
        for position in range(len(lines)):
            for span in (1, 2):
                candidate = " ".join(lines[position : position + span])
                if not candidate:
                    continue
                candidate = candidate.replace("<", " ").replace(">", " ")
                candidate = candidate.replace("«", '"').replace("»", '"')
                candidate = re.sub(r"\s+", " ", candidate).strip(" .,:;—-")
                # Quotes must be exact OCR fragments, not clipped approximations.
                if len(candidate) > 64:
                    continue
                letters = re.findall(r"[A-Za-zА-Яа-яЁё]", candidate)
                if len(candidate) < 5 or len(letters) < 4:
                    continue
                stems = _extract_ocr_token_stems(candidate)
                overlap = len(stems & card_stems)
                signal = _contains_category_ocr_signal(category, candidate)
                if overlap < 2 and not signal:
                    continue
                score = overlap * 10 + int(signal) * 8
                score += min(len(stems), 4)
                # Prefer a compact phrase over a single isolated label.
                if 10 <= len(candidate) <= 52:
                    score += 2
                ranked = (score, -position, candidate)
                if best is None or ranked > best:
                    best = ranked
        if best is not None:
            return photo_number, best[2]
    return None, ""


def _matches_regex(pattern: str, text: str) -> bool:
    return re.search(pattern, text, re.IGNORECASE) is not None


def _describe_product_context(
    *,
    category: str,
    final_label: int,
    name: str,
    description: str,
    has_conflicting_evidence: bool = False,
) -> str:
    """Return an honest card-specific fact when no strict rule was selected."""
    name_text = normalize_text(name)
    text = f"{name_text}. {normalize_text(description)}"

    if category == SUPPLEMENTS_CATEGORY:
        if int(final_label) == 1:
            facts: list[str] = []
            if _matches_regex(
                r"\b(?:витамин\w*|минерал\w*|магни\w*|желез\w*|цинк\w*|"
                r"омега[-\s]?3|коллаген\w*|коэнзим\w*|пробиотик\w*)\b",
                text,
            ):
                facts.append("указаны витамины, минералы или другие активные вещества")
            if _matches_regex(
                r"\b(?:капсул\w*|таблет\w*|саше|стик\w*|порош\w*|пастил\w*|"
                r"жевательн\w+\s+резин\w*)\b",
                text,
            ):
                facts.append("описана форма продукта для дозированного употребления")
            if _matches_regex(
                r"\b(?:принима\w*|при[её]м\w*|употребля\w*|дозиров\w*|"
                r"по\s+\d+\s+(?:капсул\w*|таблет\w*|порци\w*)|курс\w+\s+при[её]ма)\b",
                text,
            ):
                facts.append("приведены дозировка или схема приёма")
            if facts:
                return " и ".join(facts[:2])
            return "название и описание характеризуют товар как добавку для приёма внутрь"

        if _matches_regex(
            r"\b(?:протеин\w*|гейнер\w*|bcaa|креатин\w*|аминокислот\w*|"
            r"предтренировочн\w*)\b",
            name_text,
        ):
            return "товар заявлен как спортивное питание, а не как отдельная категория БАД"
        if _matches_regex(
            r"\b(?:крем\w*|сыворотк\w*|шампун\w*|маск\w*|космети\w*|"
            r"таблетниц\w*|органайзер\w*|контейнер\w*)\b",
            name_text,
        ):
            return "по названию продаётся косметический товар или аксессуар, а не добавка"
        if has_conflicting_evidence:
            return "по основному назначению товар не относится к биологически активным добавкам"
        return "в названии и описании нет прямой маркировки товара как БАД"

    if int(final_label) == 1:
        flammable_facts = (
            (r"\bспич(?:ки|ек|ечн\w*)\b", "продаются спички или спичечный товар"),
            (r"\bзажигалк\w*\b", "продаётся зажигалка"),
            (
                r"\b(?:газ\w*(?:\s+бутан\w*)?\s+для\s+заправк\w*|газов\w+\s+баллон\w*|баллон\w*\s+(?:с\s+)?газ\w*)\b",
                "указан газ для заправки или баллон с горючим газом",
            ),
            (
                r"\b(?:бензин\w*|керосин\w*|топлив\w*|сух\w*\s+горюч\w*|жидк\w+\s+для\s+розжиг\w*)\b",
                "указано горючее или топливо",
            ),
            (
                r"\b(?:дым\w+\s+шашк\w*|бенгальск\w+\s+огн\w*|свеч\w*[-\s]+фонтан\w*)\b",
                "указано воспламеняемое пиротехническое изделие",
            ),
            (
                r"\b(?:розжиг\w*\s+набор|набор\w*\s+коробк\w*|\d+\s+коробк\w*)\b",
                "название указывает на набор коробков для розжига",
            ),
            (
                r"\bв\s+комплект\w*\s+с\s+угл[её]м\b",
                "в названии прямо указано, что уголь входит в комплект",
            ),
        )
        for pattern, fact in flammable_facts:
            if _matches_regex(pattern, name_text):
                return fact
        description_facts = (
            (r"\bспич(?:ки|ек|ечн\w*)\b", "в описании прямо упомянуты спички"),
            (r"\bзажигалк\w*\b", "в описании прямо упомянута зажигалка"),
            (
                r"\b(?:газов\w+\s+баллон\w*|баллон\w*\s+(?:с\s+)?газ\w*)\b",
                "в описании прямо упомянут баллон с газом",
            ),
            (
                r"\b(?:бензин\w*|керосин\w*|топлив\w*|сух\w*\s+горюч\w*)\b",
                "в описании прямо упомянуто горючее или топливо",
            ),
        )
        description_text = normalize_text(description)
        for pattern, fact in description_facts:
            if _matches_regex(pattern, description_text):
                return fact
        return "название и описание содержат признаки горючего содержимого или источника огня"

    if _matches_regex(
        r"(?:\bсух\w*\s+горюч\w*\b|"
        r"\b(?:бензин\w*|топлив\w*|газ\w*)\b[^.!?]{0,60}"
        r"\b(?:зажигал\w*|zippo|зиппо|заправк\w*)\b|"
        r"\bфитил\w*\s+для\s+(?:свеч\w*|ламп\w*|зажигал\w*)\b|"
        r"\bзаправк\w*\s+для\s+(?:зажигал\w*|горел\w*|ламп\w*)\b)",
        name_text,
    ):
        return "товар является заправкой или расходным материалом для другого устройства, а не самостоятельным источником огня"
    if _matches_regex(
        r"\b(?:дров\w*|угол\w*|брик(?:ет|еты|етов)\w*|пеллет\w*)\b",
        name_text,
    ):
        return "указан твёрдый материал для печи или розжига, но не газовое либо жидкое топливо"
    if _matches_regex(r"\b(?:ароматическ\w*|декоративн\w*)\s+свеч\w*\b", name_text):
        return "продаётся обычная ароматическая или декоративная свеча, а не пиротехническая свеча-фонтан"
    if _matches_regex(r"\bхлопушк\w*\s+пневматическ\w*\b", name_text):
        return "продаётся пневматическая хлопушка без заявленного воспламеняемого состава"
    if _matches_regex(
        r"\b(?:опахал\w*|махалк\w*|веер\w*|шампур\w*|реш[её]тк\w*|"
        r"чехол\w*|держател\w*|подставк\w*|адаптер\w*|переходник\w*)\b",
        name_text,
    ):
        return "продаётся принадлежность или аксессуар, а не горючее либо источник огня"
    if _matches_regex(
        r"\b(?:мангал\w*|грил\w*|печ\w*|плит\w*|котел\w*|барбекю)\b",
        name_text,
    ):
        return "продаётся оборудование для приготовления, а не включённое горючее"
    if has_conflicting_evidence:
        return "основной товар не является самостоятельным источником огня или отдельным горючим содержимым"
    return "в названии и описании нет прямого указания на газ, топливо, спички, зажигалку или пиротехнический состав"


def _render_model_based_comment(
    *,
    category: str,
    final_label: int,
    name: str,
    description: str,
    ocr_photo: int | None = None,
    ocr_fact: str = "",
    has_conflicting_evidence: bool = False,
) -> str:
    category_name = "БАД" if category == SUPPLEMENTS_CATEGORY else "ЛВ"
    decision = "относится" if int(final_label) == 1 else "не относится"
    fact = _describe_product_context(
        category=category,
        final_label=final_label,
        name=name,
        description=description,
        has_conflicting_evidence=has_conflicting_evidence,
    )
    photo_note = (
        f" На фото {ocr_photo} прочитан фрагмент «{ocr_fact}»."
        if ocr_photo is not None and ocr_fact
        else ""
    )
    suffix = f": {fact}.{photo_note} Поэтому товар {decision} к категории {category_name}."
    full_name = _truncate_product_name(name, 100_000)
    if photo_note and len(f"Карточка «{full_name}»{suffix}") > 290:
        # The product identity and decision rationale are more valuable than an
        # optional OCR quote when all three do not fit cleanly.
        photo_note = ""
        suffix = f": {fact}. Поэтому товар {decision} к категории {category_name}."
    # Use the whole title whenever the 290-character jury-facing target allows it.
    # Only the title is shortened, always at a word boundary.
    name_budget = max(18, 290 - len("Карточка «»") - len(suffix))
    return f"Карточка «{_truncate_product_name(name, name_budget)}»{suffix}"


def build_comment(
    *,
    product_id: str,
    category: str,
    final_label: int,
    evidence: CommentEvidence | None,
    name: str = "",
    description: str = "",
    conflicting_evidence: CommentEvidence | None = None,
    ocr_photo: int | None = None,
    ocr_fact: str = "",
) -> str:
    if (
        category == SUPPLEMENTS_CATEGORY
        and ocr_fact
        and not matches_any_pattern(
            COMMENT_SUPPLEMENTS_OCR_MARKER_PATTERNS,
            ocr_fact,
        )
    ):
        ocr_photo = None
        ocr_fact = ""
    if category == SUPPLEMENTS_CATEGORY and ocr_fact:
        is_negated = matches_any_pattern(SUPPLEMENTS_NEGATION_PATTERNS, ocr_fact)
        if (int(final_label) == 0) != is_negated:
            ocr_photo = None
            ocr_fact = ""
    if evidence is None:
        comment = _render_model_based_comment(
            category=category,
            final_label=final_label,
            name=name,
            description=description,
            ocr_photo=ocr_photo,
            ocr_fact=ocr_fact,
            has_conflicting_evidence=conflicting_evidence is not None,
        )
        return validate_comment(comment)
    reason = EVIDENCE_DESCRIPTION_BY_REASON[evidence.reason]
    location = _describe_evidence_location(evidence)
    category_name = "БАД" if category == SUPPLEMENTS_CATEGORY else "ЛВ"
    observed = (
        f"{location} найдена формулировка «{evidence.fact}»; она указывает, что {reason}"
        if evidence.fact
        else f"{location} установлено, что {reason}"
    )
    photo_note = (
        f" На фото {ocr_photo} прочитан фрагмент «{ocr_fact}»."
        if ocr_photo is not None and ocr_fact and not evidence.photos
        else ""
    )
    if int(final_label) == 1:
        templates = (
            f"{observed}.{photo_note} Поэтому товар относится к категории {category_name}.",
            f"{observed}.{photo_note} Это подтверждает принадлежность товара к категории {category_name}.",
        )
    else:
        templates = (
            f"{observed}.{photo_note} Поэтому товар не относится к категории {category_name}.",
            f"{observed}.{photo_note} Это подтверждает отсутствие принадлежности к категории {category_name}.",
        )
    chosen = templates[get_deterministic_template_index(product_id, len(templates))]
    if len(re.sub(r"\s+", " ", chosen).strip()) > 290 and photo_note:
        if int(final_label) == 1:
            without_photo = (
                f"{observed}. Поэтому товар относится к категории {category_name}.",
                f"{observed}. Это подтверждает принадлежность товара к категории {category_name}.",
            )
        else:
            without_photo = (
                f"{observed}. Поэтому товар не относится к категории {category_name}.",
                f"{observed}. Это подтверждает отсутствие принадлежности к категории {category_name}.",
            )
        chosen = without_photo[get_deterministic_template_index(product_id, len(without_photo))]
        photo_note = ""
    if len(re.sub(r"\s+", " ", chosen).strip()) > 290:
        relation = "относится" if int(final_label) == 1 else "не относится"
        if evidence.fact:
            prefix = f"{location} найдена формулировка «"
            suffix = f"».{photo_note} Поэтому товар {relation} к категории {category_name}."
            fact_budget = max(24, 290 - len(prefix) - len(suffix))
            chosen = f"{prefix}{_truncate_text(evidence.fact, fact_budget)}{suffix}"
        else:
            chosen = (
                f"{location} установлен признак: {reason}.{photo_note} "
                f"Поэтому товар {relation} к категории {category_name}."
            )
    return validate_comment(chosen)


def build_submission_comment_records(
    *,
    test_products: pd.DataFrame,
    predicted_labels_by_index: Mapping[int, int],
    images_directory: Path,
    ocr_text_by_path: Mapping[str, str],
    submission_config: Mapping[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    """Build comments together with the evidence needed for a human audit."""
    evidence_by_index: dict[int, CommentEvidence | None] = {}
    conflicting_evidence_by_index: dict[int, CommentEvidence | None] = {}
    ocr_excerpt_by_index: dict[int, tuple[int | None, str]] = {}
    evidence_comment_count = 0

    for index, product in test_products.iterrows():
        predicted_label = predicted_labels_by_index[index]
        image_limit = max_images_for_category(submission_config, str(product.category))
        ocr_texts_by_photo = load_raw_ocr_texts(
            images_directory=images_directory,
            product_id=str(product.id),
            limit=image_limit,
            ocr_text_by_path=ocr_text_by_path,
        )
        ocr_excerpt_by_index[index] = select_relevant_ocr_excerpt(
            ocr_rows=ocr_texts_by_photo,
            category=str(product.category),
            name=str(product["name"]),
            description=str(product.description),
            final_label=predicted_label,
        )
        detected_evidence = detect_comment_evidence(
            category=str(product.category),
            name=str(product["name"]),
            description=str(product.description),
            product_id=str(product.id),
            images_directory=images_directory,
            image_limit=image_limit,
            ocr_text_by_path=ocr_text_by_path,
        )
        existing_photos = tuple(
            number
            for number, _ in find_product_image_paths(
                images_directory, str(product.id), image_limit
            )
        )
        consistent_evidence = select_evidence_matching_prediction(
            evidence=detected_evidence,
            category=str(product.category),
            final_label=predicted_label,
            existing_photos=existing_photos,
        )
        evidence_by_index[index] = consistent_evidence
        conflicting_evidence_by_index[index] = (
            detected_evidence
            if detected_evidence is not None
            and expected_label_for_reason(detected_evidence.reason) != predicted_label
            else None
        )
        if consistent_evidence is not None:
            evidence_comment_count += 1

    records_by_index: dict[int, dict[str, Any]] = {}
    model_comment_count = 0
    conflicting_evidence_count = 0
    photo_comment_count = 0
    for index, product in test_products.iterrows():
        evidence = evidence_by_index[index]
        if evidence is None:
            model_comment_count += 1
            if conflicting_evidence_by_index[index] is not None:
                conflicting_evidence_count += 1
        ocr_photo, ocr_fact = ocr_excerpt_by_index[index]
        comment = build_comment(
            product_id=str(product.id),
            category=str(product.category),
            final_label=predicted_labels_by_index[index],
            evidence=evidence,
            name=str(product["name"]),
            description=str(product.description),
            conflicting_evidence=conflicting_evidence_by_index[index],
            ocr_photo=ocr_photo,
            ocr_fact=ocr_fact,
        )
        conflicting_evidence = conflicting_evidence_by_index[index]
        if evidence is not None:
            comment_source = "rule_evidence"
        elif conflicting_evidence is not None:
            comment_source = "conflict_trace"
        else:
            comment_source = "model_decision"
        references_photo = bool(re.search(r"\bфото\s+\d+\b", comment, re.IGNORECASE))
        records_by_index[index] = {
            "comment": comment,
            "comment_source": comment_source,
            "evidence_source": evidence.source if evidence is not None else "",
            "evidence_reason": evidence.reason if evidence is not None else "",
            "evidence_confidence": evidence.confidence if evidence is not None else "",
            "evidence_photos": evidence.photos if evidence is not None else (),
            "evidence_fact": evidence.fact if evidence is not None else "",
            "conflicting_reason": (
                conflicting_evidence.reason if conflicting_evidence is not None else ""
            ),
            "ocr_photo": ocr_photo,
            "ocr_excerpt": ocr_fact,
            "references_photo": references_photo,
        }
        if references_photo:
            photo_comment_count += 1
    statistics = {
        "rule_evidence": evidence_comment_count,
        "model_decision": model_comment_count - conflicting_evidence_count,
        "conflict_trace": conflicting_evidence_count,
        "comments_with_photos": photo_comment_count,
    }
    return records_by_index, statistics


def build_submission_comments(
    *,
    test_products: pd.DataFrame,
    predicted_labels_by_index: Mapping[int, int],
    images_directory: Path,
    ocr_text_by_path: Mapping[str, str],
    submission_config: Mapping[str, Any],
) -> tuple[dict[int, str], dict[str, int]]:
    """Build the competition comments while keeping audit details optional."""
    records, statistics = build_submission_comment_records(
        test_products=test_products,
        predicted_labels_by_index=predicted_labels_by_index,
        images_directory=images_directory,
        ocr_text_by_path=ocr_text_by_path,
        submission_config=submission_config,
    )
    return {index: str(record["comment"]) for index, record in records.items()}, statistics
