"""
src/extract_dataset.py
======================
Automated dataset downloader and unpacker for EdgeWeather.
Downloads Arrow dataset archives from Hugging Face and extracts images into
the canonical directory structure: data/raw/<class_name>/<sample_id>.jpg.
"""

from __future__ import annotations

import argparse
import io
import urllib.request
from pathlib import Path

import pyarrow.ipc as ipc
from PIL import Image
from tqdm import tqdm

LABEL_NAMES = [
    "dew",
    "fog",       # mapped from fogsmog in source archive
    "frost",
    "glaze",
    "hail",
    "lightning",
    "rain",
    "rainbow",
    "rime",
    "sandstorm",
    "snow",
]

HF_ARROW_URLS = {
    "valid": "https://huggingface.co/datasets/dazzle-nu/autotrain-data-weather-classification/resolve/main/processed/valid/dataset.arrow",
    "train": "https://huggingface.co/datasets/dazzle-nu/autotrain-data-weather-classification/resolve/main/processed/train/dataset.arrow",
}


class DownloadProgressBar(tqdm):
    """Callback hook for urllib to render a dynamic download progress bar."""
    def update_to(self, b: int = 1, bsize: int = 1, tsize: int | None = None) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path) -> Path:
    """Download a remote URL to local disk with progress monitoring."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Archive already exists locally: {output_path} ({output_path.stat().st_size / (1024*1024):.1f} MB)")
        return output_path

    print(f"Downloading from {url}...")
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    print(f"Saved: {output_path} ({output_path.stat().st_size / (1024*1024):.1f} MB)")
    return output_path


def extract_arrow_file(arrow_path: Path, output_root: Path, prefix: str = "valid") -> int:
    """Extract embedded JPEG byte structures from Arrow IPC stream into class subdirectories."""
    print(f"Reading Arrow IPC stream: {arrow_path}...")
    reader = ipc.open_stream(str(arrow_path))
    table = reader.read_all()

    num_rows = table.num_rows
    print(f"Total rows in {prefix} archive: {num_rows:,}")

    images_col = table["image"]
    targets_col = table["target"]

    saved_count = 0
    for i in tqdm(range(num_rows), desc=f"Extracting {prefix}", unit="img"):
        target_idx = targets_col[i].as_py()
        label = LABEL_NAMES[target_idx] if target_idx < len(LABEL_NAMES) else f"class_{target_idx}"

        img_struct = images_col[i].as_py()
        img_bytes = img_struct.get("bytes")
        if not img_bytes:
            continue

        class_dir = output_root / label
        class_dir.mkdir(parents=True, exist_ok=True)

        out_path = class_dir / f"{prefix}_{i:05d}.jpg"
        try:
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            pil_img.save(out_path, format="JPEG", quality=95)
            saved_count += 1
        except Exception:
            continue

    print(f"Extracted {saved_count:,} valid images to {output_root}")
    return saved_count


def run_pipeline(downloads_dir: Path, raw_dir: Path, skip_download: bool = False) -> int:
    """End-to-end dataset acquisition and unpack pipeline."""
    downloads_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    valid_arrow = downloads_dir / "valid.arrow"
    train_arrow = downloads_dir / "train.arrow"

    if not skip_download:
        if not valid_arrow.exists():
            download_file(HF_ARROW_URLS["valid"], valid_arrow)
        if not train_arrow.exists():
            download_file(HF_ARROW_URLS["train"], train_arrow)

    total = 0
    if valid_arrow.exists():
        total += extract_arrow_file(valid_arrow, raw_dir, prefix="val")
    else:
        print(f"Warning: {valid_arrow} not found. Skipping validation archive extraction.")

    if train_arrow.exists():
        total += extract_arrow_file(train_arrow, raw_dir, prefix="trn")
    else:
        print(f"Warning: {train_arrow} not found. Skipping training archive extraction.")

    print(f"\nDataset extraction complete! Total images organized in {raw_dir}: {total:,}")
    for class_folder in sorted(raw_dir.iterdir()):
        if class_folder.is_dir():
            count = len(list(class_folder.glob("*.jpg")))
            print(f"  - {class_folder.name:12s}: {count:,} images")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EdgeWeather: Download and extract 11-class atmospheric weather dataset."
    )
    parser.add_argument(
        "--downloads_dir",
        type=Path,
        default=Path("data/downloads"),
        help="Directory to cache downloaded Arrow archives.",
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("data/raw"),
        help="Destination directory for organized class folders.",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip downloading if Arrow archives are already present in downloads_dir.",
    )
    args = parser.parse_args()

    run_pipeline(
        downloads_dir=args.downloads_dir,
        raw_dir=args.raw_dir,
        skip_download=args.skip_download,
    )


if __name__ == "__main__":
    main()
