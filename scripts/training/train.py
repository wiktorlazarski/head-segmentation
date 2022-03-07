import logging
import os

import hydra
import omegaconf
import pytorch_lightning as pl

import scripts.training.train_modules as tm


@hydra.main(
    config_path=os.path.join(os.getcwd(), "configs"), config_name="training_experiment"
)
def main(configs: omegaconf.DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s train.py: %(message)s")
    logging.info("Processed started.")

    logging.info("Creating dataset module.")
    dataset_module = tm.HumanHeadSegmentationDataModule(
        dataset_root=configs.dataset_module.dataset_root,
        nn_image_input_resolution=configs.dataset_module.nn_image_input_resolution,
        batch_size=configs.dataset_module.batch_size,
        num_workers=configs.dataset_module.num_workers,
        all_augmentations=configs.dataset_module.all_augmentations,
        size_augmentation_keys=configs.dataset_module.size_augmentation_keys,
        content_augmentation_keys=configs.dataset_module.content_augmentation_keys,
    )

    logging.info("Creating neural network module.")
    nn_module = tm.HumanHeadSegmentationModelModule(
        encoder_name=configs.nn_module.encoder_name,
        encoder_depth=configs.nn_module.encoder_depth,
        pretrained=configs.nn_module.use_pretrained,
        nn_image_input_resolution=configs.dataset_module.nn_image_input_resolution,
    )

    nn_trainer = pl.Trainer()
    logging.info("Starting training loop.")
    nn_trainer.fit(nn_module, dataset_module)

    logging.info("Starting testing loop.")
    nn_trainer.test()

    logging.info("Processed finished.")


if __name__ == "__main__":
    main()
