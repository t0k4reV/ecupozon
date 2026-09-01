"""Deterministic evidence rules used only to explain Gemma decisions."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from .core import (
    FLAMMABLE_CATEGORY,
    SUPPLEMENTS_CATEGORY,
    CommentEvidence,
    find_product_image_paths,
    normalize_text,
)

SUPPLEMENTS_POSITIVE_REASONS = frozenset({"EXPLICIT_SUPPLEMENT", "SUPPLEMENT_LABEL_ON_PACKAGE"})
SUPPLEMENTS_NEGATIVE_REASONS = frozenset(
    {"EXPLICIT_NOT_SUPPLEMENT", "SPORT_NUTRITION_ONLY", "SUPPLEMENT_ACCESSORY"}
)
FLAMMABLE_POSITIVE_REASONS = frozenset(
    {
        "IGNITION_SOURCE",
        "MATCHES",
        "LIGHTER",
        "GAS_CANISTER",
        "COMBUSTIBLE_FUEL",
        "CANDLE",
        "SMOKE_DEVICE",
        "INCLUDED_LV",
    }
)
FLAMMABLE_NEGATIVE_REASONS = frozenset(
    {
        "NOT_INCLUDED",
        "EMPTY_DEVICE",
        "EXTERNAL_FUEL",
        "BUILTIN_IGNITION",
        "COMBUSTIBLE_COMPONENT",
        "ACCESSORY_ONLY",
    }
)
EVIDENCE_REASONS_BY_CATEGORY = {
    SUPPLEMENTS_CATEGORY: SUPPLEMENTS_POSITIVE_REASONS | SUPPLEMENTS_NEGATIVE_REASONS,
    FLAMMABLE_CATEGORY: FLAMMABLE_POSITIVE_REASONS | FLAMMABLE_NEGATIVE_REASONS,
}
POSITIVE_REASONS = SUPPLEMENTS_POSITIVE_REASONS | FLAMMABLE_POSITIVE_REASONS
NEGATIVE_REASONS = SUPPLEMENTS_NEGATIVE_REASONS | FLAMMABLE_NEGATIVE_REASONS
VALID_EVIDENCE_SOURCES = frozenset({"TEXT", "IMAGE", "BOTH"})
VALID_EVIDENCE_CONFIDENCE_LEVELS = frozenset({"HIGH", "MEDIUM"})

EVIDENCE_DESCRIPTION_BY_REASON = {
    "EXPLICIT_SUPPLEMENT": "товар прямо обозначен как биологически активная добавка",
    "SUPPLEMENT_LABEL_ON_PACKAGE": "присутствует маркировка товара как биологически активной добавки",
    "EXPLICIT_NOT_SUPPLEMENT": "прямо указано, что товар не является биологически активной добавкой",
    "SPORT_NUTRITION_ONLY": "товар описан как спортивное питание без прямой маркировки БАД",
    "SUPPLEMENT_ACCESSORY": "продаётся аксессуар или ёмкость для хранения, а не сама добавка",
    "IGNITION_SOURCE": "обнаружен самостоятельный источник воспламенения",
    "MATCHES": "товар или явно указанная часть комплекта — спички",
    "LIGHTER": "товар или явно указанная часть комплекта — зажигалка",
    "GAS_CANISTER": "товар или явно указанная часть комплекта — баллон с горючим газом",
    "COMBUSTIBLE_FUEL": "товар или явно указанная часть комплекта — горючее либо топливо",
    "CANDLE": "обнаружена пиротехническая свеча или свеча-фонтан",
    "SMOKE_DEVICE": "обнаружено дымовое устройство с воспламеняемым составом",
    "INCLUDED_LV": "прямо указано наличие легковоспламеняющегося предмета в комплекте",
    "NOT_INCLUDED": "прямо указано, что горючее или баллон не входит в комплект",
    "EMPTY_DEVICE": "продаётся устройство без самостоятельного горючего содержимого",
    "EXTERNAL_FUEL": "устройство использует отдельно подключаемое топливо, не входящее в товар",
    "BUILTIN_IGNITION": "источник поджига является встроенным механизмом другого устройства",
    "COMBUSTIBLE_COMPONENT": "горючий материал является только компонентом другого изделия",
    "ACCESSORY_ONLY": "продаётся аксессуар, а не источник воспламенения или горючее содержимое",
}

SUPPLEMENTS_NEGATION_PATTERNS = (
    re.compile(
        r"\bне\s+(?:явля\w*|счита\w*)\s+"
        r"(?:бад\b|биологически\s+активн\w*\s+добавк\w*)"
        r"|\bне\s+относ\w*\s+(?:к\s+)?(?:категори\w*\s+)?"
        r"(?:бад\b|биологически\s+активн\w*\s+добавк\w*)",
        re.IGNORECASE,
    ),
    re.compile(r"\bnot\s+(?:a\s+)?dietary\s+supplement\b", re.IGNORECASE),
)
SUPPLEMENTS_MARKER_PATTERNS = (
    re.compile(r"(?<![а-яёa-z0-9])бад(?![а-яёa-z0-9])", re.IGNORECASE),
    re.compile(r"\bбиологически\s+активн\w*\s+добавк\w*", re.IGNORECASE),
    re.compile(r"\bdietary\s+supplement\b", re.IGNORECASE),
    re.compile(r"\bбиодобавк\w*\b", re.IGNORECASE),
)
OCR_SUPPLEMENTS_MARKER_PATTERNS = SUPPLEMENTS_MARKER_PATTERNS + (
    re.compile(r"\b(?:dietary|oietary|dictary)\s+(?:supplement|supolement)\b", re.IGNORECASE),
    re.compile(r"\bsupplement\s+facts\b", re.IGNORECASE),
)
# Judge-facing OCR quotes use a deliberately narrower standard than rule search:
# every quoted supplement fragment must itself contain an explicit category marker.
COMMENT_SUPPLEMENTS_OCR_MARKER_PATTERNS = (
    re.compile(r"(?<![а-яёa-z0-9])бад(?![а-яёa-z0-9])", re.IGNORECASE),
    re.compile(r"\bбиологически\s+активн\w*\s+добавк\w*", re.IGNORECASE),
    re.compile(
        r"\b(?:dietary|oietary|dictary)\s+(?:supplement|supolement)\b",
        re.IGNORECASE,
    ),
)
SPORT_NUTRITION_PATTERNS = (
    re.compile(r"\bспортивн\w*\s+питан\w*", re.IGNORECASE),
    re.compile(r"\b(?:протеин|гейнер|bcaa|креатин)\w*\b", re.IGNORECASE),
)
SUPPLEMENT_ACCESSORY_PATTERNS = (
    re.compile(
        r"\b(?:таблетниц\w*|органайзер\w*|контейнер\w*|чехол\w*|кейс\w*|"
        r"дозатор\w*)\b[^.!?]{0,70}\b(?:для\s+)?(?:таблет\w*|капсул\w*|бад\b)",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:упаковк\w*|баночк\w*)\s+для\s+(?:бад\b|капсул\w*|таблет\w*)", re.IGNORECASE),
)

FLAMMABLE_NOT_INCLUDED_PATTERNS = (
    re.compile(
        r"\bбез\s+(?:газов\w+\s+)?(?:баллон\w*|спич\w*|топлив\w*|горюч\w*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:баллон\w*|спич\w*|топлив\w*|горюч\w*)[^.!?]{0,60}"
        r"\b(?:не\s+вход\w*|прода\w*\s+отдельно|приобрета\w*\s+отдельно)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:не\s+вход\w*|прода\w*\s+отдельно|приобрета\w*\s+отдельно)"
        r"[^.!?]{0,60}(?:баллон\w*|спич\w*|топлив\w*|горюч\w*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bв\s+комплект\w*\s+(?:не\s+вход\w*|нет)\s+"
        r"(?:газов\w+\s+)?(?:баллон\w*|спич\w*|зажигалк\w*|топлив\w*|горюч\w*)",
        re.IGNORECASE,
    ),
)
FLAMMABLE_INCLUDED_PATTERNS = (
    re.compile(
        r"\bв\s+комплект\w*\s+(?:вход\w*|ид[её]т|есть|поставля\w*)\s+"
        r"(?!(?:адаптер|переходник|шланг|редуктор)\w*)[^.!?]{0,20}"
        r"(?:газов\w+\s+баллон\w*|спич\w*|зажигалк\w*|"
        r"сух\w*\s+горюч\w*|топливо\b|угол\w*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:газов\w+\s+баллон\w*|спич\w*|зажигалк\w*|сух\w*\s+горюч\w*|"
        r"топливо\b|угол\w*)[^.!?]{0,15}\bвход\w*\s+в\s+комплект",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bкомплектаци\w*\s*[:—-][^.!?]{0,30}"
        r"(?:газов\w+\s+баллон\w*|спич\w*|зажигалк\w*|"
        r"сух\w*\s+горюч\w*|топливо\b|угол\w*)",
        re.IGNORECASE,
    ),
)
FLAMMABLE_GAS_DEVICE_PATTERN = re.compile(
    r"\b(?:газов\w+\s+)?(?:плит\w*|плитк\w*|грил\w*|горелк\w*|"
    r"насадк\w*[-\s]+горелк\w*)",
    re.IGNORECASE,
)
FLAMMABLE_EXTERNAL_FUEL_PATTERNS = (
    re.compile(r"\bработ\w*[^.!?]{0,90}\b(?:от|на)\s+(?:\w+\s+){0,3}баллон\w*", re.IGNORECASE),
    re.compile(r"\b(?:подключ\w*|устанавлив\w*|креп\w*)[^.!?]{0,90}\bбаллон\w*", re.IGNORECASE),
    re.compile(
        r"\b(?:насадк\w*[-\s]+горелк\w*|горелк\w*)[^.!?]{0,80}\bна\s+(?:бутанов\w+\s+|газов\w+\s+)?баллон\w*",
        re.IGNORECASE,
    ),
)
FLAMMABLE_EMPTY_DEVICE_PATTERNS = (
    re.compile(
        r"\b(?:пуст\w*|незаправлен\w*)\s+(?:горелк\w*|зажигалк\w*|резервуар\w*)", re.IGNORECASE
    ),
    re.compile(r"\b(?:плит\w*|грил\w*|мангал\w*)\b[^.!?]{0,80}\bбез\s+топлив\w*", re.IGNORECASE),
)
FLAMMABLE_BUILTIN_IGNITION_PATTERN = re.compile(
    r"\b(?:пьезо\s*поджиг\w*|пьезоподжиг\w*|электро\s*поджиг\w*|электроподжиг\w*)\b",
    re.IGNORECASE,
)
FLAMMABLE_DEVICE_PATTERN = re.compile(
    r"\b(?:плит\w*|плитк\w*|грил\w*|горелк\w*|мангал\w*|печ\w*|устройств\w*)\b",
    re.IGNORECASE,
)
FLAMMABLE_COMBUSTIBLE_COMPONENT_PATTERNS = (
    re.compile(
        r"\b(?:горюч\w*|воспламеня\w*)\s+(?:компонент\w*|част\w*|элемент\w*)\b", re.IGNORECASE
    ),
    re.compile(r"\bфитил\w*\s+(?:для|к)\s+(?:свеч\w*|ламп\w*)\b", re.IGNORECASE),
)
FLAMMABLE_ACCESSORY_ONLY_PATTERNS = (
    re.compile(
        r"\b(?:чехол\w*|футляр\w*|держател\w*|подставк\w*|органайзер\w*|"
        r"коробк\w*|коробок|наклейк\w*)\s+(?:для|под)\s+"
        r"(?:спич\w*|зажигалк\w*|свеч\w*|(?:газов\w+\s+)?баллон\w*)",
        re.IGNORECASE,
    ),
    re.compile(r"\bнабор\s+для\s+(?:спич\w*|зажигалк\w*)", re.IGNORECASE),
    re.compile(
        r"\b(?:адаптер\w*|переходник\w*|шланг\w*|редуктор\w*|клапан\w*|"
        r"насадк\w*)\s+(?:для|на|к)\s+(?:газов\w+\s+)?баллон\w*",
        re.IGNORECASE,
    ),
    re.compile(r"\bпуст\w*\s+(?:коробк\w*|коробок)\s+(?:для|из-под)\s+спич\w*", re.IGNORECASE),
)
FLAMMABLE_SELF_CONTAINED_BURNER_PATTERN = re.compile(
    r"\b(?:мини[-\s]?горелк\w*|зажигалк\w*[-\s]+горелк\w*|"
    r"горелк\w*[^.!?]{0,50}(?:перезаправ\w*|заправляем\w*|"
    r"встроенн\w*\s+резервуар\w*))\b",
    re.IGNORECASE,
)
FLAMMABLE_NAME_REASON_PATTERNS = (
    ("MATCHES", (re.compile(r"(?:^|\bнабор\w*\s+)(?:спички|спичек)\b", re.IGNORECASE),)),
    ("LIGHTER", (re.compile(r"^\s*зажигалк\w*\b", re.IGNORECASE),)),
    (
        "GAS_CANISTER",
        (
            re.compile(r"^\s*(?:газ\w*\s+для\s+заправк\w*|газов\w+\s+баллон\w*)\b", re.IGNORECASE),
            re.compile(
                r"^\s*баллон(?:а|ы|ов|ом|е)?\b[^.!?]{0,35}\b(?:бутан|пропан|газ|mapp|мапп)\w*\b",
                re.IGNORECASE,
            ),
            re.compile(r"\+\s*\d+\s+баллон(?:а|ы|ов)?\s+с\s+газ\w*", re.IGNORECASE),
        ),
    ),
    (
        "COMBUSTIBLE_FUEL",
        (
            re.compile(
                r"^\s*(?:жидк\w+\s+для\s+розжиг\w*|бензин\b|керосин\b|сух\w*\s+горюч\w*)",
                re.IGNORECASE,
            ),
            re.compile(
                r"^\s*(?:топливн\w+\s+брикет\w*|угол\w+\s+(?:для\s+грил\w*|древесн\w*)|растопк\w*)",
                re.IGNORECASE,
            ),
        ),
    ),
    (
        "CANDLE",
        (
            re.compile(r"^\s*свеч\w*[-\s]+фонтан\w*\b", re.IGNORECASE),
            re.compile(r"^\s*фонтан\w*\s+(?:для|в)\s+торт\w*\b", re.IGNORECASE),
            re.compile(r"^\s*бенгальск\w*\s+(?:свеч\w*|огн\w*)\b", re.IGNORECASE),
        ),
    ),
    (
        "SMOKE_DEVICE",
        (
            re.compile(
                r"^\s*(?:цветн\w*\s+дым|дым\w*\s+(?:шашк\w*|факел\w*|устройств\w*))\b",
                re.IGNORECASE,
            ),
        ),
    ),
)


def matches_any_pattern(patterns: Sequence[re.Pattern[str]], text: str) -> bool:
    return any(pattern.search(text) is not None for pattern in patterns)


def extract_matching_text(patterns: Sequence[re.Pattern[str]], text: str) -> str:
    """Return the exact normalized fragment that triggered a deterministic rule."""
    for pattern in patterns:
        match = pattern.search(text)
        if match is None:
            continue
        value = normalize_text(match.group(0)).replace("<", " ").replace(">", " ")
        value = value.replace("«", '"').replace("»", '"').strip(" .,:;—-")
        if len(value) > 82:
            value = value[:83].rsplit(" ", 1)[0].rstrip(" .,:;—-") + "…"
        return value
    return ""


def _extract_matching_ocr_text(
    rows: Sequence[tuple[int, str]],
    patterns: Sequence[re.Pattern[str]],
) -> str:
    for _, text in rows:
        value = extract_matching_text(patterns, text)
        if value:
            return value
    return ""


def _load_normalized_ocr_texts(
    *,
    images_directory: Path,
    product_id: str,
    limit: int,
    ocr_text_by_path: Mapping[str, str],
) -> list[tuple[int, str]]:
    values: list[tuple[int, str]] = []
    for number, path in find_product_image_paths(images_directory, product_id, limit):
        text = normalize_text(ocr_text_by_path.get(str(path.resolve()), ""))
        if text:
            values.append((number, text))
    return values


def load_raw_ocr_texts(
    *,
    images_directory: Path,
    product_id: str,
    limit: int,
    ocr_text_by_path: Mapping[str, str],
) -> list[tuple[int, str]]:
    values: list[tuple[int, str]] = []
    for number, path in find_product_image_paths(images_directory, product_id, limit):
        text = str(ocr_text_by_path.get(str(path.resolve()), "") or "").strip()
        if normalize_text(text):
            values.append((number, text))
    return values


def detect_comment_evidence(
    *,
    category: str,
    name: str,
    description: str,
    product_id: str,
    images_directory: Path,
    image_limit: int,
    ocr_text_by_path: Mapping[str, str],
) -> CommentEvidence | None:
    name_text = normalize_text(name)
    description_text = normalize_text(description)
    text = f"{name_text}. {description_text}"
    ocr_rows = _load_normalized_ocr_texts(
        images_directory=images_directory,
        product_id=product_id,
        limit=image_limit,
        ocr_text_by_path=ocr_text_by_path,
    )

    if category == SUPPLEMENTS_CATEGORY:
        if matches_any_pattern(SUPPLEMENT_ACCESSORY_PATTERNS, name_text):
            return CommentEvidence(
                "TEXT",
                "SUPPLEMENT_ACCESSORY",
                (),
                "HIGH",
                extract_matching_text(SUPPLEMENT_ACCESSORY_PATTERNS, name_text),
            )
        if matches_any_pattern(SUPPLEMENTS_NEGATION_PATTERNS, text):
            return CommentEvidence(
                "TEXT",
                "EXPLICIT_NOT_SUPPLEMENT",
                (),
                "HIGH",
                extract_matching_text(SUPPLEMENTS_NEGATION_PATTERNS, text),
            )
        negated_photos = tuple(
            number
            for number, ocr in ocr_rows
            if matches_any_pattern(SUPPLEMENTS_NEGATION_PATTERNS, ocr)
        )
        if negated_photos:
            return CommentEvidence(
                "IMAGE",
                "EXPLICIT_NOT_SUPPLEMENT",
                negated_photos,
                "MEDIUM",
                _extract_matching_ocr_text(ocr_rows, SUPPLEMENTS_NEGATION_PATTERNS),
            )
        supplement_photos = tuple(
            number
            for number, ocr in ocr_rows
            if matches_any_pattern(COMMENT_SUPPLEMENTS_OCR_MARKER_PATTERNS, ocr)
        )
        has_supplement_marker = matches_any_pattern(SUPPLEMENTS_MARKER_PATTERNS, text)
        sport_description_pattern = re.compile(
            r"\bспортивн\w*\s+питан\w*",
            re.IGNORECASE,
        )
        has_sport_marker = matches_any_pattern(SPORT_NUTRITION_PATTERNS, name_text) or bool(
            sport_description_pattern.search(description_text)
        )
        # Карточки спортивного питания часто одновременно содержат служебное
        # упоминание БАД. Это конфликт признаков, а не надёжное основание для
        # человекочитаемого комментария.
        if has_sport_marker and (has_supplement_marker or supplement_photos):
            return None
        if has_supplement_marker:
            supplement_fact = extract_matching_text(SUPPLEMENTS_MARKER_PATTERNS, text)
            if supplement_photos:
                return CommentEvidence(
                    "BOTH", "EXPLICIT_SUPPLEMENT", supplement_photos, "HIGH", supplement_fact
                )
            return CommentEvidence("TEXT", "EXPLICIT_SUPPLEMENT", (), "HIGH", supplement_fact)
        if supplement_photos:
            return CommentEvidence(
                "IMAGE",
                "SUPPLEMENT_LABEL_ON_PACKAGE",
                supplement_photos,
                "MEDIUM",
                _extract_matching_ocr_text(ocr_rows, COMMENT_SUPPLEMENTS_OCR_MARKER_PATTERNS),
            )
        if has_sport_marker:
            sport_fact = extract_matching_text(SPORT_NUTRITION_PATTERNS, name_text)
            if not sport_fact:
                sport_fact = extract_matching_text((sport_description_pattern,), description_text)
            return CommentEvidence("TEXT", "SPORT_NUTRITION_ONLY", (), "MEDIUM", sport_fact)
        return None

    if matches_any_pattern(FLAMMABLE_NOT_INCLUDED_PATTERNS, text):
        return CommentEvidence(
            "TEXT",
            "NOT_INCLUDED",
            (),
            "HIGH",
            extract_matching_text(FLAMMABLE_NOT_INCLUDED_PATTERNS, text),
        )
    if matches_any_pattern(FLAMMABLE_ACCESSORY_ONLY_PATTERNS, name_text):
        return CommentEvidence(
            "TEXT",
            "ACCESSORY_ONLY",
            (),
            "HIGH",
            extract_matching_text(FLAMMABLE_ACCESSORY_ONLY_PATTERNS, name_text),
        )
    if matches_any_pattern(FLAMMABLE_INCLUDED_PATTERNS, text):
        return CommentEvidence(
            "TEXT",
            "INCLUDED_LV",
            (),
            "HIGH",
            extract_matching_text(FLAMMABLE_INCLUDED_PATTERNS, text),
        )
    if FLAMMABLE_GAS_DEVICE_PATTERN.search(text) is not None and matches_any_pattern(
        FLAMMABLE_EXTERNAL_FUEL_PATTERNS, text
    ):
        return CommentEvidence(
            "TEXT",
            "EXTERNAL_FUEL",
            (),
            "HIGH",
            extract_matching_text(FLAMMABLE_EXTERNAL_FUEL_PATTERNS, text),
        )
    if (
        FLAMMABLE_DEVICE_PATTERN.search(text) is not None
        and FLAMMABLE_BUILTIN_IGNITION_PATTERN.search(text) is not None
        and FLAMMABLE_SELF_CONTAINED_BURNER_PATTERN.search(name_text) is None
    ):
        return CommentEvidence(
            "TEXT",
            "BUILTIN_IGNITION",
            (),
            "HIGH",
            extract_matching_text((FLAMMABLE_BUILTIN_IGNITION_PATTERN,), text),
        )
    if matches_any_pattern(FLAMMABLE_COMBUSTIBLE_COMPONENT_PATTERNS, text):
        return CommentEvidence(
            "TEXT",
            "COMBUSTIBLE_COMPONENT",
            (),
            "MEDIUM",
            extract_matching_text(FLAMMABLE_COMBUSTIBLE_COMPONENT_PATTERNS, text),
        )
    if matches_any_pattern(FLAMMABLE_ACCESSORY_ONLY_PATTERNS, text):
        return CommentEvidence(
            "TEXT",
            "ACCESSORY_ONLY",
            (),
            "HIGH",
            extract_matching_text(FLAMMABLE_ACCESSORY_ONLY_PATTERNS, text),
        )
    if matches_any_pattern(FLAMMABLE_EMPTY_DEVICE_PATTERNS, text):
        return CommentEvidence(
            "TEXT",
            "EMPTY_DEVICE",
            (),
            "HIGH",
            extract_matching_text(FLAMMABLE_EMPTY_DEVICE_PATTERNS, text),
        )

    if FLAMMABLE_SELF_CONTAINED_BURNER_PATTERN.search(name_text) is not None:
        burner_patterns = (FLAMMABLE_SELF_CONTAINED_BURNER_PATTERN,)
        photos = tuple(
            number for number, ocr in ocr_rows if matches_any_pattern(burner_patterns, ocr)
        )
        if photos:
            return CommentEvidence(
                "BOTH",
                "IGNITION_SOURCE",
                photos,
                "HIGH",
                extract_matching_text((FLAMMABLE_SELF_CONTAINED_BURNER_PATTERN,), name_text),
            )
        return CommentEvidence(
            "TEXT",
            "IGNITION_SOURCE",
            (),
            "HIGH",
            extract_matching_text((FLAMMABLE_SELF_CONTAINED_BURNER_PATTERN,), name_text),
        )

    # Для положительных ЛВ-признаков название определяет продаваемый объект.
    # Описание и OCR могут подтвердить его и указать фотографию, но простого
    # упоминания совместимого баллона/спичек в инструкции недостаточно.
    for reason, patterns in FLAMMABLE_NAME_REASON_PATTERNS:
        if matches_any_pattern(patterns, name_text):
            photos = tuple(number for number, ocr in ocr_rows if matches_any_pattern(patterns, ocr))
            if photos:
                return CommentEvidence(
                    "BOTH", reason, photos, "HIGH", extract_matching_text(patterns, name_text)
                )
            return CommentEvidence(
                "TEXT", reason, (), "HIGH", extract_matching_text(patterns, name_text)
            )

    return None


def expected_label_for_reason(reason: str) -> int | None:
    if reason in POSITIVE_REASONS:
        return 1
    if reason in NEGATIVE_REASONS:
        return 0
    return None


def select_evidence_matching_prediction(
    *,
    evidence: CommentEvidence | None,
    category: str,
    final_label: int,
    existing_photos: Sequence[int],
) -> CommentEvidence | None:
    if evidence is None:
        return None
    if evidence.reason not in EVIDENCE_REASONS_BY_CATEGORY.get(category, frozenset()):
        return None
    if expected_label_for_reason(evidence.reason) != int(final_label):
        return None
    if (
        evidence.source not in VALID_EVIDENCE_SOURCES
        or evidence.confidence not in VALID_EVIDENCE_CONFIDENCE_LEVELS
    ):
        return None
    photos = tuple(dict.fromkeys(int(number) for number in evidence.photos))
    if evidence.source in {"IMAGE", "BOTH"}:
        if not photos or not set(photos).issubset(set(existing_photos)):
            return None
    elif photos:
        return None
    return CommentEvidence(
        evidence.source,
        evidence.reason,
        photos,
        evidence.confidence,
        normalize_text(evidence.fact)[:90],
    )
