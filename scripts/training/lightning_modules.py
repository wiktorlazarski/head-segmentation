import os
import typing as t

import pytorch_lightning as pl
import torch
from torchmetrics import ConfusionMatrix

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
        super().__init__()

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
        lr: float,
        encoder_name: str,
        encoder_depth: int,
        pretrained: bool,
        nn_image_input_resolution: int,
        background_weight: float = 1.0,
        head_weight: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = lr
        self.criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor([background_weight, head_weight])
        )

        self.train_cm_metric = ConfusionMatrix(num_classes=2)
        self.val_cm_metric = ConfusionMatrix(num_classes=2)
        self.test_cm_metric = ConfusionMatrix(num_classes=2)

        self.neural_net = mdl.HeadSegmentationModel(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            pretrained=pretrained,
            nn_image_input_resolution=nn_image_input_resolution,
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.neural_net.parameters(), lr=self.learning_rate)

    def training_step(
        self, batch: t.Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> pl.utilities.types.STEP_OUTPUT:
        step_results = self._step(batch, cm_metric=self.train_cm_metric)

        self.log("train_step_loss", step_results["loss"].item(), on_step=True)

        return step_results

    def validation_step(
        self, batch: t.Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> pl.utilities.types.STEP_OUTPUT:
        return self._step(batch, cm_metric=self.val_cm_metric)

    def test_step(
        self, batch: t.Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> pl.utilities.types.STEP_OUTPUT:
        return self._step(batch, cm_metric=self.test_cm_metric)

    def training_epoch_end(self, outputs: pl.utilities.types.EPOCH_OUTPUT) -> None:
        self._summarize_epoch(
            log_prefix="train", outputs=outputs, cm_metric=self.train_cm_metric
        )

    def validation_epoch_end(self, outputs: pl.utilities.types.EPOCH_OUTPUT) -> None:
        self._summarize_epoch(
            log_prefix="val", outputs=outputs, cm_metric=self.val_cm_metric
        )

    def test_epoch_end(self, outputs: pl.utilities.types.EPOCH_OUTPUT) -> None:
        self._summarize_epoch(
            log_prefix="test", outputs=outputs, cm_metric=self.test_cm_metric
        )

    def _step(
        self, batch: t.Tuple[torch.Tensor, torch.Tensor], cm_metric: ConfusionMatrix
    ) -> pl.utilities.types.STEP_OUTPUT:
        image, true_segmap = batch

        pred_segmap = self.neural_net(image)

        loss = self.criterion(pred_segmap, true_segmap)

        cm_metric(pred_segmap, true_segmap)

        return {"loss": loss}

    def _summarize_epoch(
        self,
        log_prefix: str,
        outputs: pl.utilities.types.EPOCH_OUTPUT,
        cm_metric: ConfusionMatrix,
    ) -> None:
        mean_loss = torch.tensor([out["loss"] for out in outputs]).mean()
        self.log(f"{log_prefix}_loss", mean_loss, on_epoch=True)

        cm = cm_metric.compute()
        cm = cm.detach()

        ious = cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) - cm.diag() + 1e-15)
        background_iou, head_iou = ious[0], ious[1]
        mIoU = ious.mean()

        self.log(f"{log_prefix}_background_IoU", background_iou, on_epoch=True)
        self.log(f"{log_prefix}_head_IoU", head_iou, on_epoch=True)
        self.log(f"{log_prefix}_mIoU", mIoU, on_epoch=True)

        cm_metric.reset()
