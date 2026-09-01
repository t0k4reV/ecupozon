"""Prepare and verify the project-local Gemma snapshot, downloading only if absent."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

MODEL_ID = "google/gemma-4-E4B-it"
MODEL_REVISION = "main"
REQUIRED_MODEL_FILES = {
    "config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
}
MODEL_WEIGHTS_FILE = "model.safetensors"
MODEL_WEIGHTS_INDEX_FILE = "model.safetensors.index.json"


def get_default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=get_default_project_root())
    return parser.parse_args()


def calculate_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_snapshot(model_directory: Path) -> list[str]:
    missing_files = sorted(
        filename for filename in REQUIRED_MODEL_FILES if not (model_directory / filename).is_file()
    )
    if missing_files:
        raise RuntimeError(f"Gemma snapshot is incomplete: {missing_files}")

    weights_path = model_directory / MODEL_WEIGHTS_FILE
    if weights_path.is_file():
        return [MODEL_WEIGHTS_FILE]

    index_path = model_directory / MODEL_WEIGHTS_INDEX_FILE
    if not index_path.is_file():
        raise RuntimeError(
            f"Gemma snapshot has neither {MODEL_WEIGHTS_FILE} nor {MODEL_WEIGHTS_INDEX_FILE}"
        )
    weights_index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = weights_index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"Invalid model weight index: {index_path}")
    weight_shards = sorted(set(weight_map.values()))
    missing_shards = [
        filename for filename in weight_shards if not (model_directory / filename).is_file()
    ]
    if missing_shards:
        raise RuntimeError(f"Gemma snapshot is missing weight shards: {missing_shards}")
    return [MODEL_WEIGHTS_INDEX_FILE, *weight_shards]


def get_lfs_sha256(sibling: Any) -> str | None:
    lfs = getattr(sibling, "lfs", None)
    value = lfs.get("sha256") if isinstance(lfs, Mapping) else getattr(lfs, "sha256", None)
    return value if isinstance(value, str) and len(value) == 64 else None


def write_manifest(model_directory: Path, payload: dict[str, Any]) -> None:
    manifest_path = model_directory / "download_manifest.json"
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.unlink(missing_ok=True)
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(manifest_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    args = parse_arguments()
    from huggingface_hub import HfApi, snapshot_download

    project_root = args.project_root.expanduser().resolve()
    model_directory = project_root / "models" / "google" / "gemma-4-E4B-it"
    hf_token = os.getenv("HF_TOKEN") or None
    model_info = HfApi(token=hf_token).model_info(
        MODEL_ID,
        revision=MODEL_REVISION,
        files_metadata=True,
    )
    resolved_revision = model_info.sha
    if not resolved_revision:
        raise RuntimeError(f"Hugging Face did not return a commit for {MODEL_ID}")

    snapshot_source = "existing_local_snapshot"
    if model_directory.exists() and any(model_directory.iterdir()):
        weight_files = validate_model_snapshot(model_directory)
    else:
        model_directory.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=MODEL_ID,
            revision=resolved_revision,
            local_dir=model_directory,
            token=hf_token,
            max_workers=4,
        )
        weight_files = validate_model_snapshot(model_directory)
        snapshot_source = "huggingface_download"

    remote_files = {sibling.rfilename: sibling for sibling in model_info.siblings or ()}
    snapshot_files = sorted(
        path
        for path in model_directory.iterdir()
        if path.is_file() and path.name != "download_manifest.json"
    )
    file_hashes = {path.name: calculate_sha256(path) for path in snapshot_files}
    for filename in weight_files:
        if filename.endswith(".safetensors"):
            sibling = remote_files.get(filename)
            remote_sha256 = get_lfs_sha256(sibling) if sibling is not None else None
            if remote_sha256 is None:
                raise RuntimeError(f"Hugging Face returned no LFS SHA-256 for {filename}")
            if file_hashes[filename] != remote_sha256:
                raise RuntimeError(
                    f"Local {filename} does not match {MODEL_ID}@{resolved_revision}; "
                    "refusing to download over the existing snapshot"
                )

    write_manifest(
        model_directory,
        {
            "model_id": MODEL_ID,
            "requested_revision": MODEL_REVISION,
            "resolved_revision": resolved_revision,
            "source": snapshot_source,
            "weight_files": weight_files,
            "files": file_hashes,
        },
    )
    print(
        f"Gemma {resolved_revision} verified at {model_directory} ({snapshot_source})",
        flush=True,
    )


if __name__ == "__main__":
    main()
