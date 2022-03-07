import random
import typing as t

import albumentations as A
import numpy as np
import torch


class AugmentationPipeline:
    __RESIZE_AUGMENTATIONS = {
        "crop0.1": A.CropAndPad(percent=-0.1, keep_size=False, always_apply=True),
        "crop0.05": A.CropAndPad(percent=-0.05, keep_size=False, always_apply=True),
        "crop0.02": A.CropAndPad(percent=-0.02, keep_size=False, always_apply=True),
        "identity": A.CropAndPad(percent=0.0, keep_size=False, always_apply=True),
        "pad0.02": A.CropAndPad(percent=0.02, keep_size=False, always_apply=True),
        "pad0.05": A.CropAndPad(percent=0.05, keep_size=False, always_apply=True),
        "pad0.1": A.CropAndPad(percent=0.1, keep_size=False, always_apply=True),
    }

    __AUGMENTATIONS = {
        "brightness": A.RandomBrightnessContrast(brightness_limit=0.4, p=0.8),
        "affine": A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-60, 60),
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            shear=0.0,
            p=0.5,
        ),
        "hflip": A.HorizontalFlip(p=0.5),
        "gaussian_blur": A.GaussianBlur(blur_limit=(21, 31), p=0.5),
        "motion_blur": A.MotionBlur(blur_limit=(21, 31), p=0.5),
        "rain_noise": A.RandomRain(p=0.05),
        "fog_noise": A.RandomFog(p=0.05),
    }

    def __init__(
        self,
        all_augmentations: bool = True,
        size_augmentation_keys: t.Optional[t.List[str]] = None,
        content_augmentation_keys: t.Optional[t.List[str]] = None,
    ):
        content_augmentation_ops = []
        self.resize_augmentation_ops = size_augmentation_keys

        if all_augmentations:
            self.resize_augmentation_ops = list(
                AugmentationPipeline.__RESIZE_AUGMENTATIONS.values()
            )
            content_augmentation_ops = list(
                AugmentationPipeline.__AUGMENTATIONS.values()
            )

        if not all_augmentations and content_augmentation_keys is not None:
            content_augmentation_ops = [
                AugmentationPipeline.__AUGMENTATIONS[aug]
                for aug in content_augmentation_keys
            ]

        self.aug_pipeline = A.Compose(content_augmentation_ops)

    def __call__(
        self, image: np.ndarray, segmap: np.ndarray
    ) -> t.Tuple[torch.Tensor, torch.Tensor]:
        if self.resize_augmentation_ops is not None:
            resize_aug = random.choice(self.resize_augmentation_ops)
            resized_image = resize_aug(image=image, mask=segmap)

            aug_result = self.aug_pipeline(
                image=resized_image["image"], mask=resized_image["mask"]
            )
        else:
            aug_result = self.aug_pipeline(image=image, mask=segmap)

        return aug_result["image"], aug_result["mask"]
