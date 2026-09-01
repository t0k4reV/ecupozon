"""Download and verify the two offline EasyOCR weights used by submission."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

DOWNLOAD_TIMEOUT_SECONDS = 120
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
EASYOCR_MODELS = {
    "craft_mlt_25k.pth": {
        "url": "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip",
        "sha256": "4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17",
    },
    "cyrillic_g2.pth": {
        "url": "https://github.com/JaidedAI/EasyOCR/releases/download/v1.6.1/cyrillic_g2.zip",
        "sha256": "48d0f3b58f28aa64651ab1032cc2d498c4de25135829668e87c14e7a07529f29",
    },
}


def get_default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=get_default_project_root())
    return parser.parse_args()


def calculate_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_safe_archive_member(member_name: str) -> bool:
    member_path = PurePosixPath(member_name)
    return bool(member_name) and not member_path.is_absolute() and ".." not in member_path.parts


def download_ocr_model(
    filename: str,
    model_metadata: dict[str, str],
    destination_path: Path,
) -> None:
    if (
        destination_path.is_file()
        and calculate_sha256(destination_path) == model_metadata["sha256"]
    ):
        print(f"Verified existing {destination_path}")
        return

    with tempfile.TemporaryDirectory(prefix="ecup-easyocr-") as temporary_name:
        temporary_directory = Path(temporary_name)
        archive_path = temporary_directory / f"{filename}.zip"
        with (
            urlopen(model_metadata["url"], timeout=DOWNLOAD_TIMEOUT_SECONDS) as response,
            archive_path.open("wb") as output_stream,
        ):
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_ARCHIVE_BYTES:
                raise ValueError(f"EasyOCR archive is unexpectedly large: {content_length} bytes")
            downloaded = 0
            while chunk := response.read(1024 * 1024):
                downloaded += len(chunk)
                if downloaded > MAX_ARCHIVE_BYTES:
                    raise ValueError("EasyOCR archive exceeded the download size limit")
                output_stream.write(chunk)

        with zipfile.ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"Downloaded archive is corrupt: {model_metadata['url']}")
            matching_members = [
                member
                for member in archive.infolist()
                if is_safe_archive_member(member.filename)
                and PurePosixPath(member.filename).name == filename
            ]
            if len(matching_members) != 1:
                raise ValueError(f"Expected one {filename} in {model_metadata['url']}")
            extracted_path = temporary_directory / filename
            with (
                archive.open(matching_members[0]) as source_stream,
                extracted_path.open("wb") as output_stream,
            ):
                shutil.copyfileobj(source_stream, output_stream, length=1024 * 1024)

        actual_sha256 = calculate_sha256(extracted_path)
        if actual_sha256 != model_metadata["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {filename}: {actual_sha256}")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_destination = destination_path.with_suffix(destination_path.suffix + ".tmp")
        if temporary_destination.exists():
            raise FileExistsError(f"Stale temporary download: {temporary_destination}")
        try:
            shutil.copy2(extracted_path, temporary_destination)
            temporary_destination.replace(destination_path)
        except Exception:
            temporary_destination.unlink(missing_ok=True)
            raise
    print(f"Downloaded and verified {destination_path}")


def main() -> None:
    args = parse_arguments()
    models_directory = args.project_root.expanduser().resolve() / "artifacts" / "easyocr"
    for filename, model_metadata in EASYOCR_MODELS.items():
        download_ocr_model(
            filename,
            model_metadata,
            models_directory / filename,
        )


if __name__ == "__main__":
    main()
