from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.common.evaluation.comments_audit import compare_comments_with_golden
from scripts.common.submission.runtime.submission_runtime.comments import (
    build_submission_comment_records,
    validate_comment,
)
from scripts.common.submission.runtime.submission_runtime.ocr import (
    load_precomputed_ocr_texts,
)
from scripts.common.submission.runtime.submission_runtime.rules import (
    detect_comment_evidence,
)

SUPPLEMENTS_CATEGORY = "БАД"
FLAMMABLE_CATEGORY = "Легковоспламеняющиеся"
COMMENT_CONFIG = {
    "max_images": {SUPPLEMENTS_CATEGORY: 1, FLAMMABLE_CATEGORY: 3},
}


class CommentsV4Tests(unittest.TestCase):
    def test_name_column_does_not_resolve_to_series_index(self) -> None:
        products = pd.DataFrame(
            [
                {
                    "id": "product-1",
                    "name": "Историческое название товара",
                    "description": "",
                    "category": SUPPLEMENTS_CATEGORY,
                }
            ],
            index=[42],
        )
        records, _ = build_submission_comment_records(
            test_products=products,
            predicted_labels_by_index={42: 0},
            images_directory=Path("/missing/images"),
            ocr_text_by_path={},
            submission_config=COMMENT_CONFIG,
        )
        self.assertIn("Историческое название товара", records[42]["comment"])
        self.assertNotIn("Карточка «42»", records[42]["comment"])

    def test_ocr_cache_preserves_lines_and_recovers_failed_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            images_directory = root / "images"
            product_directory = images_directory / "1"
            product_directory.mkdir(parents=True)
            image_path = product_directory / "0.jpg"
            image_path.write_bytes(b"not-decoded-by-this-test")
            cache_path = root / "ocr.jsonl"
            records = [
                {
                    "id": "1",
                    "image_path": "images/1/0.jpg",
                    "ocr": "",
                    "error": "RuntimeError: worker failed",
                },
                {
                    "id": "1",
                    "image_path": "images/1/0.jpg",
                    "ocr": "БАД\nDietary Supplement",
                    "error": "",
                },
            ]
            cache_path.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                encoding="utf-8",
            )
            products = pd.DataFrame([{"id": "1", "category": SUPPLEMENTS_CATEGORY}])

            cache, metadata = load_precomputed_ocr_texts(
                test_products=products,
                images_directory=images_directory,
                submission_config=COMMENT_CONFIG,
                cache_path=cache_path,
            )

            self.assertEqual(cache[str(image_path.resolve())], "БАД\nDietary Supplement")
            self.assertEqual(metadata["duplicate_records"], 1)
            self.assertEqual(metadata["recovered_errors"], 1)

    def test_historical_supplement_and_flammable_rules(self) -> None:
        supplement = detect_comment_evidence(
            category=SUPPLEMENTS_CATEGORY,
            name="Витаминный комплекс",
            description="Биологически активная добавка. Не является лекарством.",
            product_id="1",
            images_directory=Path("/missing/images"),
            image_limit=1,
            ocr_text_by_path={},
        )
        flammable = detect_comment_evidence(
            category=FLAMMABLE_CATEGORY,
            name="Газовая горелка с пьезоподжигом",
            description="В комплекте нет газового баллона.",
            product_id="2",
            images_directory=Path("/missing/images"),
            image_limit=3,
            ocr_text_by_path={},
        )
        self.assertIsNotNone(supplement)
        self.assertEqual(supplement.reason, "EXPLICIT_SUPPLEMENT")
        self.assertIsNotNone(flammable)
        self.assertEqual(flammable.reason, "NOT_INCLUDED")

    def test_golden_comparison_passes_when_same_verdict_text_is_exact(self) -> None:
        current = pd.DataFrame(
            [
                {
                    "id": "1",
                    "category": SUPPLEMENTS_CATEGORY,
                    "name": "Товар",
                    "prediction": 1,
                    "probability": 0.9,
                    "threshold": 0.7,
                    "comment": "Комментарий длиной больше пятидесяти символов для проверки результата.",
                    "comment_valid": True,
                    "result": "<комментарий>Комментарий<вердикт>не бан",
                },
                {
                    "id": "2",
                    "category": FLAMMABLE_CATEGORY,
                    "name": "Другой товар",
                    "prediction": 0,
                    "probability": 0.1,
                    "threshold": 0.5,
                    "comment": "Новый комментарий для карточки с изменившимся решением модели.",
                    "comment_valid": True,
                    "result": "<комментарий>Новый комментарий<вердикт>бан",
                },
            ]
        )
        golden = pd.DataFrame(
            [
                {
                    "id": "1",
                    "category": SUPPLEMENTS_CATEGORY,
                    "gemma_v2_prediction": 1,
                    "gemma_v2_probability_1": 0.8,
                    "threshold": 0.7,
                    "comment": current.loc[0, "comment"],
                    "result": current.loc[0, "result"],
                },
                {
                    "id": "2",
                    "category": FLAMMABLE_CATEGORY,
                    "gemma_v2_prediction": 1,
                    "gemma_v2_probability_1": 0.9,
                    "threshold": 0.005,
                    "comment": "Исторический комментарий для другого решения модели.",
                    "result": "<комментарий>Исторический комментарий<вердикт>не бан",
                },
            ]
        )

        report, mismatches = compare_comments_with_golden(current, golden)

        self.assertTrue(report["comments_reproduction_passed"])
        self.assertEqual(report["exact_when_same_verdict_rate"], 1.0)
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches.iloc[0]["mismatch_kind"], "verdict_changed")

    def test_comment_validation_contract(self) -> None:
        comment = validate_comment(
            "Карточка содержит проверяемое основание. Поэтому товар относится к категории БАД."
        )
        self.assertGreaterEqual(len(comment), 50)
        self.assertLessEqual(len(comment), 290)


if __name__ == "__main__":
    unittest.main()
