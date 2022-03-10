import pathlib as p
import typing as t

import cv2
import torch
import torchvision

import head_segmentation.image_processing as ip
import scripts.training.augmentations as aug


class CelebAHeadSegmentationDataset(torchvision.datasets.VisionDataset):
    def __init__(
        self,
        dataset_root: str,
        preprocess_pipeline: t.Optional[ip.PreprocessingPipeline] = None,
        augmentation_pipeline: t.Optional[aug.AugmentationPipeline] = None,
    ):
        super().__init__(root=dataset_root)

        self.image_files = self._load_image_files()
        self.preprocess_pipeline = preprocess_pipeline
        self.aug_pipeline = augmentation_pipeline

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
        image_file = self.image_files[index]

        image, segmap = self._load_sample(image_file)

        if self.aug_pipeline is not None:
            image, segmap = self.aug_pipeline(image=image, segmap=segmap)

        if self.preprocess_pipeline is not None:
            image = self.preprocess_pipeline.preprocess_image(image=image)
            segmap = self.preprocess_pipeline.preprocess_segmap(segmap=segmap)

        return image, segmap

    def _load_image_files(self) -> t.List[str]:
        images_dir = p.Path(self.root) / "images"

        image_files = sorted(list(images_dir.glob("*.jpg")))

        return image_files

    def _load_sample(
        self, image_filename: p.Path
    ) -> t.Tuple[torch.Tensor, torch.Tensor]:
        image = cv2.imread(str(image_filename), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        segmap_file = (
            image_filename.parent.parent / "segmaps" / f"{image_filename.stem}.png"
        )
        segmap = cv2.imread(str(segmap_file), cv2.IMREAD_GRAYSCALE)

        return image, segmap
