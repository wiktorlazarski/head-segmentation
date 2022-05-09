import argparse

import torch
import torchmetrics
from loguru import logger

import head_segmentation.segmentation_pipeline as seg_pipeline
import scripts.training.data_loading as dl


def parse_args() -> None:
    # fmt: off
    parser = argparse.ArgumentParser("Evaluates segmentation maps predictions on full resolution")

    parser.add_argument("-dp", "--dataset_path", required=True, type=str, help="Path to a test dataset.")
    parser.add_argument("-mp", "--model_path", required=True, type=str, help="Model's checkpoint to evaluate.")
    parser.add_argument("-nn_in", "--nn_image_input_resolution", required=True, type=int, help="Neural Network input image resolution.")

    # fmt: on
    return parser.parse_args()


def evaluate() -> None:
    logger.info("🚀 Evaluation process started.")

    args = parse_args()

    logger.info("📚 Creating dataset module.")
    eval_dataset = dl.CelebAHeadSegmentationDataset(dataset_root=args.dataset_path)

    logger.info("🕸 Loading neural network module.")
    segmentation_pipeline = seg_pipeline.HumanHeadSegmentationPipeline(
        model_path=args.model_path,
        image_input_resolution=args.nn_image_input_resolution,
    )

    cm_metric = torchmetrics.ConfusionMatrix(num_classes=2)

    logger.info("🔁 Evaluation loop.")
    for image, true_segmap in eval_dataset:
        predicted_segmap = segmentation_pipeline.predict(image)

        cm_metric(torch.tensor(predicted_segmap), torch.tensor(true_segmap))

    cm = cm_metric.compute()
    cm = cm.detach()

    ious = cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) - cm.diag() + 1e-15)
    background_iou, head_iou = ious[0], ious[1]
    mIoU = ious.mean()

    logger.info(f"📈 Evaluation summary for a model {args.model_path}:")
    logger.info(f"\t 🖼 Background class IoU: {background_iou:.6f}")
    logger.info(f"\t 👦 Head class IoU: {head_iou:.6f}")
    logger.info(f"\t 🤷 mIoU: {mIoU:.6f}")

    logger.info("🏁 Evaluation process finished.")


if __name__ == "__main__":
    evaluate()
