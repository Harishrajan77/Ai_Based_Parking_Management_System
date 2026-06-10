import argparse
import random
import shutil
from pathlib import Path


VIDEO_DIR = Path(__file__).resolve().parents[1]
SOURCE_IMAGES = VIDEO_DIR / "images"
SOURCE_LABELS = VIDEO_DIR / "labels"
OUTPUT_DIR = VIDEO_DIR / "parking_occupancy_yolo"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ["empty", "occupied"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a YOLOv8 parking occupancy dataset.")
    parser.add_argument("--images", type=Path, default=SOURCE_IMAGES, help="Source image folder.")
    parser.add_argument("--labels", type=Path, default=SOURCE_LABELS, help="Source YOLO label folder.")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Prepared YOLO dataset output folder.")
    parser.add_argument("--train", type=float, default=0.80, help="Train split ratio.")
    parser.add_argument("--valid", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument("--overwrite", action="store_true", help="Recreate the output folder.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def collect_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    image_files = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    pairs = []
    missing_labels = []

    for image_path in image_files:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))
        else:
            missing_labels.append(image_path.name)

    if missing_labels:
        preview = ", ".join(missing_labels[:5])
        raise FileNotFoundError(f"Missing labels for {len(missing_labels)} images. First missing: {preview}")

    if not pairs:
        raise RuntimeError(f"No image/label pairs found in {images_dir} and {labels_dir}.")

    return pairs


def make_split_dirs(output_dir: Path) -> None:
    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)


def split_pairs(pairs: list[tuple[Path, Path]], train_ratio: float, valid_ratio: float, seed: int):
    if not 0 < train_ratio < 1:
        raise ValueError("--train must be between 0 and 1.")
    if not 0 < valid_ratio < 1:
        raise ValueError("--valid must be between 0 and 1.")
    if train_ratio + valid_ratio >= 1:
        raise ValueError("--train + --valid must be less than 1 so a test split remains.")

    pairs = pairs[:]
    random.Random(seed).shuffle(pairs)
    train_end = int(len(pairs) * train_ratio)
    valid_end = train_end + int(len(pairs) * valid_ratio)
    return {
        "train": pairs[:train_end],
        "val": pairs[train_end:valid_end],
        "test": pairs[valid_end:],
    }


def copy_split(split_name: str, pairs: list[tuple[Path, Path]], output_dir: Path) -> None:
    image_dir = output_dir / split_name / "images"
    label_dir = output_dir / split_name / "labels"

    for image_path, label_path in pairs:
        shutil.copy2(image_path, image_dir / image_path.name)
        shutil.copy2(label_path, label_dir / label_path.name)


def write_data_yaml(output_dir: Path) -> Path:
    yaml_path = output_dir / "data.yaml"
    names = ", ".join(f"'{name}'" for name in CLASS_NAMES)
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_dir.as_posix()}",
                "train: train/images",
                "val: val/images",
                "test: test/images",
                "",
                f"names: [{names}]",
                f"nc: {len(CLASS_NAMES)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def prepare_dataset(images_dir: Path, labels_dir: Path, output_dir: Path, train_ratio: float, valid_ratio: float, seed: int, overwrite: bool) -> Path:
    images_dir = resolve(images_dir)
    labels_dir = resolve(labels_dir)
    output_dir = resolve(output_dir)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels folder not found: {labels_dir}")

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    make_split_dirs(output_dir)

    pairs = collect_pairs(images_dir, labels_dir)
    splits = split_pairs(pairs, train_ratio, valid_ratio, seed)
    for split_name, split_pairs_list in splits.items():
        copy_split(split_name, split_pairs_list, output_dir)

    yaml_path = write_data_yaml(output_dir)

    print(f"Prepared dataset: {output_dir}")
    print(f"Train: {len(splits['train'])}")
    print(f"Val: {len(splits['val'])}")
    print(f"Test: {len(splits['test'])}")
    print(f"Data YAML: {yaml_path}")
    return yaml_path


def main() -> None:
    args = parse_args()
    prepare_dataset(args.images, args.labels, args.output, args.train, args.valid, args.seed, args.overwrite)


if __name__ == "__main__":
    main()
