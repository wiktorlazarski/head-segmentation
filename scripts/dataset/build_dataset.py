import argparse
import logging
import random
import shutil
import typing as t
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess dataset.")

    # fmt: off
    parser.add_argument("--preprocessed_dset", "-d", type=Path, required=True, help="Preprocessed CelebA dataset directory.")
    parser.add_argument("--output_dset_root", "-o", type=Path, required=True, help="Output dataset directory.")
    # fmt: on

    return parser.parse_args()


def load_images(dset_path: Path) -> t.List[str]:
    return list((dset_path / "images").glob("*.jpg"))


def split_dataset(
    dset_images: list,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> t.Tuple[list, list, list]:
    random.shuffle(dset_images)

    dset_size = len(dset_images)

    train_last_index = int(dset_size * train_frac)
    train_dset = dset_images[:train_last_index]

    val_last_index = train_last_index + int(dset_size * val_frac)
    val_dset = dset_images[train_last_index:val_last_index]

    test_dset = dset_images[val_last_index:]

    return train_dset, val_dset, test_dset


def create_dataset_structure(output_dir: Path) -> None:
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"

    for dataset_dir in [train_dir, val_dir, test_dir]:
        images_dir = dataset_dir / "images"
        segmaps_dir = dataset_dir / "segmaps"

        images_dir.mkdir(parents=True, exist_ok=False)
        segmaps_dir.mkdir(parents=True, exist_ok=False)


def copy_images_to_dataset_dir(
    train_dset: list,
    val_dset: list,
    test_dset: list,
    output_dset_root: Path,
) -> None:
    for image_file in train_dset:
        copy_sample(image_file, output_dset_root / "train")

    for image_file in val_dset:
        copy_sample(image_file, output_dset_root / "val")

    for image_file in test_dset:
        copy_sample(image_file, output_dset_root / "test")


def copy_sample(image_file: Path, dset_path: Path) -> None:
    shutil.copy(image_file, dset_path / "images" / image_file.name)

    segmap_path = image_file.parent.parent / "segmaps" / f"{image_file.stem}.png"
    shutil.copy(segmap_path, dset_path / "segmaps" / segmap_path.name)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )

    logging.info("Building dataset STARTED")
    args = parse_args()

    image_files = load_images(args.preprocessed_dset)
    train_dset, val_dset, test_dset = split_dataset(image_files)

    create_dataset_structure(args.output_dset_root)

    copy_images_to_dataset_dir(
        train_dset=train_dset,
        val_dset=val_dset,
        test_dset=test_dset,
        output_dset_root=args.output_dset_root,
    )
    shutil.copy(
        src=args.preprocessed_dset / "metadata.csv",
        dst=args.output_dset_root / "metadata.csv",
    )

    logging.info("Building dataset FINISHED")


if __name__ == "__main__":
    main()
