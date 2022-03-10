import datetime
import logging
import os
import warnings

import hydra
import omegaconf
import pytorch_lightning as pl

import scripts.training.lightning_modules as lm


@hydra.main(
    config_path=os.path.join(os.getcwd(), "configs"), config_name="training_experiment"
)
def main(configs: omegaconf.DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s train.py: %(message)s")
    logging.info("Processed started.")

    logging.info("Creating dataset module.")
    # Training data and model modules
    dataset_module = lm.HumanHeadSegmentationDataModule(
        dataset_root=configs.dataset_module.dataset_root,
        nn_image_input_resolution=configs.dataset_module.nn_image_input_resolution,
        batch_size=configs.dataset_module.batch_size,
        num_workers=configs.dataset_module.num_workers,
        all_augmentations=configs.dataset_module.all_augmentations,
        size_augmentation_keys=configs.dataset_module.size_augmentation_keys,
        content_augmentation_keys=configs.dataset_module.content_augmentation_keys,
    )

    logging.info("Creating neural network module.")
    nn_module = lm.HumanHeadSegmentationModelModule(
        lr=configs.nn_module.lr,
        encoder_name=configs.nn_module.encoder_name,
        encoder_depth=configs.nn_module.encoder_depth,
        pretrained=configs.nn_module.use_pretrained,
        nn_image_input_resolution=configs.dataset_module.nn_image_input_resolution,
    )

    # Callbacks
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor=configs.training.early_stop.monitor,
        patience=configs.training.early_stop.patience,
        mode=configs.training.early_stop.mode,
    )

    model_ckpt_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        monitor=configs.training.early_stop.monitor,
        mode=configs.training.early_stop.mode,
        # fmt: off
        filename="{configs.training.wandb_name}" + "-{epoch}-" + "{configs.training.early_stop.monitor:.2f}",
        # fmt: on
        save_top_k=3,
        dirpath=".",
        save_last=True,
    )

    # W&B Logger
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    wandb_logger = pl.loggers.WandbLogger(
        project=configs.training.wandb_project,
        name=f"{configs.training.wandb_name}-{timestamp}",
    )

    # Training env configs
    nn_trainer = pl.Trainer(
        logger=wandb_logger,
        log_every_n_steps=1,
        callbacks=[early_stop_callback, model_ckpt_callback],
        max_epochs=configs.training.max_epochs,
        weights_save_path="models",
        gpus=1 if configs.training.with_gpu else 0,
    )

    # Train loop
    logging.info("Starting training loop.")
    nn_trainer.fit(nn_module, dataset_module)

    # Test loop
    logging.info("Starting testing loop.")
    nn_trainer.test()

    logging.info("Processed finished.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    main()
