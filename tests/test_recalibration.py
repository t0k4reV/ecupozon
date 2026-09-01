from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.common.training.lora_training import MODEL_ID
from scripts.common.training.recalibrate_thresholds import (
    MANIFEST_PATHS,
    PROFILE_PATHS,
    REPORT_PATH,
    TARGETS,
    apply_recalibration_plan,
    build_recalibration_plan,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class RecalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.results: dict[str, dict] = {}
        for target in TARGETS:
            self.results[target.name] = self._create_target(target)
        self._create_calibration_sources()
        repository_root = Path(__file__).resolve().parents[1]
        for profile_path in PROFILE_PATHS.values():
            destination = self.root / profile_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((repository_root / profile_path).read_bytes())
        for solution, manifest_path in MANIFEST_PATHS.items():
            categories = {
                target.category: copy.deepcopy(self.results[target.name]["result"])
                for target in TARGETS
                if solution in target.solutions
            }
            write_json(
                self.root / manifest_path,
                {
                    "schema": "e_cup.gemma_lora_training",
                    "schema_version": 1,
                    "solution": solution,
                    "model_id": MODEL_ID,
                    "base_model_revision": "a" * 40,
                    "data": {"manifest_sha256": "b" * 64},
                    "seed": 2026,
                    "parameters": {"epochs": 2},
                    "categories": categories,
                    "reproduction": {"stale": True},
                },
            )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _create_target(self, target) -> dict:
        result_path = self.root / target.result_path
        output_directory = result_path.parent
        final_directory = output_directory / "final"
        final_directory.mkdir(parents=True)
        adapter_config = final_directory / "adapter_config.json"
        adapter_weights = final_directory / "adapter_model.safetensors"
        adapter_config.write_text('{"peft_type":"LORA"}\n', encoding="utf-8")
        adapter_weights.write_bytes(f"weights-{target.name}".encode())

        labels = [0, 0, 1, 1]
        if target.name == "v1_ocr_flammable":
            selected_probabilities = [0.0001, 0.0002, 0.001, 0.99]
        elif target.name == "v2_img3_flammable":
            selected_probabilities = [0.2, 0.3, 0.85, 0.95]
        else:
            selected_probabilities = [0.1, 0.2, 0.8, 0.9]
        predictions = pd.DataFrame(
            {
                "id": [f"{target.name}-{index}" for index in range(4)],
                "label": labels,
                "p_img1": [0.1, 0.2, 0.8, 0.9],
                "probability": selected_probabilities,
                "prediction": [0, 0, 1, 1],
                "correct": [True] * 4,
            }
        )
        if target.probability_column == "p_max":
            predictions["p_max"] = selected_probabilities
        elif target.name == "v1_ocr_flammable":
            predictions["p_img1"] = selected_probabilities
        predictions_path = output_directory / "validation_predictions.csv"
        predictions.to_csv(predictions_path, index=False)
        selection_path = output_directory / "selection.json"
        old_selection = {"selected_mode": target.selected_mode, "threshold_policy": "fixed"}
        write_json(selection_path, old_selection)

        inference = {
            "max_images": 3 if target.name == "v2_img3_flammable" else 1,
            "aggregation": "max" if target.name == "v2_img3_flammable" else "single",
            "threshold": 0.5,
        }
        result = {
            "schema": "e_cup.gemma_lora_category_training",
            "schema_version": 1,
            "model_id": MODEL_ID,
            "base_model_revision": "a" * 40,
            "data": {"manifest_sha256": "b" * 64},
            "seed": 2026,
            "parameters": {"epochs": 2},
            "result": {
                "category": target.category,
                "experiment": target.experiment,
                "adapter": str(final_directory.relative_to(self.root)),
                "files": {
                    adapter_config.name: sha256(adapter_config),
                    adapter_weights.name: sha256(adapter_weights),
                },
                "split": {"validation": 4},
                "inference": inference,
                "training_image_profile": {"images_per_example": 1},
                "validation": old_selection,
                "validation_artifacts": {
                    selection_path.name: sha256(selection_path),
                    predictions_path.name: sha256(predictions_path),
                },
            },
        }
        write_json(result_path, result)
        return result

    def _create_calibration_sources(self) -> None:
        for solution in MANIFEST_PATHS:
            frames = []
            source_path = None
            for target in TARGETS:
                if solution not in target.solutions:
                    continue
                predictions_path = (
                    self.root / target.result_path
                ).parent / "validation_predictions.csv"
                frame = pd.read_csv(predictions_path, dtype={"id": "string"})
                frame.insert(1, "category", target.category)
                frames.append(frame)
                if target.experiment == solution:
                    source_path = self.root / target.calibration_predictions_path
            if source_path is None:
                raise AssertionError(f"Missing calibration source for {solution}")
            source_path.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(frames, ignore_index=True).to_csv(source_path, index=False)

    def test_dry_run_apply_and_repeat_are_safe(self) -> None:
        tracked_files = [self.root / target.result_path for target in TARGETS]
        tracked_files.extend(self.root / path for path in MANIFEST_PATHS.values())
        hashes_before = {path: sha256(path) for path in tracked_files}

        prepared, report, original_hashes = build_recalibration_plan(self.root)

        self.assertEqual(hashes_before, {path: sha256(path) for path in tracked_files})
        self.assertEqual(report["targets"]["supplements"]["selected_threshold"], 0.8)
        self.assertEqual(
            report["targets"]["v1_ocr_flammable"]["selected_threshold"],
            0.001,
        )
        self.assertEqual(
            report["targets"]["v2_img3_flammable"]["selected_threshold"],
            0.85,
        )

        updated = apply_recalibration_plan(self.root, prepared, report, original_hashes)

        self.assertTrue(updated)
        self.assertTrue((self.root / REPORT_PATH).is_file())
        for solution, manifest_path in MANIFEST_PATHS.items():
            manifest = json.loads((self.root / manifest_path).read_text(encoding="utf-8"))
            expected = {
                "БАД": 0.8,
                "Легковоспламеняющиеся": (0.001 if solution == "v1_ocr" else 0.85),
            }
            self.assertEqual(
                {
                    category: result["inference"]["threshold"]
                    for category, result in manifest["categories"].items()
                },
                expected,
            )
            profile = json.loads((self.root / PROFILE_PATHS[solution]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["reproduction"], profile["training"]["reproduction"])

        prepared_again, report_again, hashes_again = build_recalibration_plan(self.root)
        updated_again = apply_recalibration_plan(
            self.root,
            prepared_again,
            report_again,
            hashes_again,
        )
        self.assertEqual(updated_again, [])

    def test_stale_validation_hash_is_rejected(self) -> None:
        target = TARGETS[0]
        selection_path = (self.root / target.result_path).parent / "selection.json"
        selection_path.write_text("{}\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "missing or stale"):
            build_recalibration_plan(self.root)

    def test_final_model_predictions_are_the_calibration_source(self) -> None:
        target = TARGETS[0]
        source_path = self.root / target.calibration_predictions_path
        source = pd.read_csv(source_path, dtype={"id": "string"})
        category_rows = source["category"].eq(target.category)
        source.loc[category_rows, "p_img1"] = [0.1, 0.2, 0.85, 0.95]
        source.to_csv(source_path, index=False)
        source_hash = sha256(source_path)

        _, report, _ = build_recalibration_plan(self.root)

        self.assertEqual(report["targets"][target.name]["selected_threshold"], 0.85)
        self.assertEqual(sha256(source_path), source_hash)


if __name__ == "__main__":
    unittest.main()
