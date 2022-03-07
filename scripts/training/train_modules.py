import os
import typing as t

import pytorch_lightning as pl
import torch

import head_segmentation.image_processing as ip
import head_segmentation.model as mdl
import scripts.training.augmentations as aug
import scripts.training.data_loading as dl


class HumanHeadSegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        dataset_root: str,
        nn_image_input_resolution: int,
        batch_size: int,
        num_workers: int,
        all_augmentations: bool,
        size_augmentation_keys: t.Optional[t.List[str]] = None,
        content_augmentation_keys: t.Optional[t.List[str]] = None,
    ):
        self.dataset_root = dataset_root
        self.nn_image_input_resolution = nn_image_input_resolution
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.all_augmentations = all_augmentations
        self.size_augmentation_keys = size_augmentation_keys
        self.content_augmentation_keys = content_augmentation_keys

    def setup(self, stage: t.Optional[str] = None) -> None:
        preprocessing_pipeline = ip.PreprocessingPipeline(
            nn_image_input_resolution=self.nn_image_input_resolution,
        )
        augmentation_pipeline = aug.AugmentationPipeline(
            all_augmentations=self.all_augmentations,
            size_augmentation_keys=self.size_augmentation_keys,
            content_augmentation_keys=self.content_augmentation_keys,
        )

        self.train_dataset = dl.CelebAHeadSegmentationDataset(
            dataset_root=os.path.join(self.dataset_root, "train"),
            preprocess_pipeline=preprocessing_pipeline,
            augmentation_pipeline=augmentation_pipeline,
        )

        self.validation_dataset = dl.CelebAHeadSegmentationDataset(
            dataset_root=os.path.join(self.dataset_root, "val"),
            preprocess_pipeline=preprocessing_pipeline,
        )

        self.test_dataset = dl.CelebAHeadSegmentationDataset(
            dataset_root=os.path.join(self.dataset_root, "test"),
            preprocess_pipeline=preprocessing_pipeline,
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )


class HumanHeadSegmentationModelModule(pl.LightningModule):
    def __init__(
        self,
        *,
        encoder_name: str,
        encoder_depth: int,
        pretrained: bool,
        nn_image_input_resolution: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.neural_net = mdl.HeadSegmentationModel(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            pretrained=pretrained,
            nn_image_input_resolution=nn_image_input_resolution,
        )

    def training_step(
        self, batch: t.Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> pl.utilities.types.STEP_OUTPUT:
        pass

    def validation_step(
        self, batch: t.Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> pl.utilities.types.STEP_OUTPUT:
        pass

    def test_step(
        self, batch: t.Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> pl.utilities.types.STEP_OUTPUT:
        pass

    def training_epoch_end(self, outputs: pl.utilities.types.EPOCH_OUTPUT) -> None:
        pass

    def validation_epoch_end(self, outputs: pl.utilities.types.EPOCH_OUTPUT) -> None:
        pass

    def test_epoch_end(self, outputs: pl.utilities.types.EPOCH_OUTPUT) -> None:
        pass
