import os

import cv2
import gdown
import numpy as np
import torch
from loguru import logger

import head_segmentation.constants as C
import head_segmentation.image_processing as ip
import head_segmentation.model as mdl


class HumanHeadSegmentationPipeline:
    def __init__(
        self,
        model_path: str = C.HEAD_SEGMENTATION_MODEL_PATH,
        model_url: str = C.HEAD_SEGMENTATION_MODEL_URL,
        device: torch.device = torch.device('cpu')
    ):
        if not os.path.exists(model_path):
            model_path = C.HEAD_SEGMENTATION_MODEL_PATH

            logger.warning(
                f"Model {model_path} doesn't exist. Downloading the model to {model_path}."
            )

            gdown.download(model_url, model_path, quiet=False)

        self.device = device
        ckpt = torch.load(model_path, map_location=torch.device("cpu"))
        hparams = ckpt["hyper_parameters"]

        self._preprocessing_pipeline = ip.PreprocessingPipeline(
            nn_image_input_resolution=hparams["nn_image_input_resolution"]
        )
        self._model = mdl.HeadSegmentationModel.load_from_checkpoint(
            ckpt_path=model_path
        )
        self._model.to(self.device)
        self._model.eval()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.predict(image)

    def predict(self, image: np.ndarray) -> np.ndarray:
        preprocessed_image = self._preprocess_image(image)
        preprocessed_image = preprocessed_image.to(self.device)
        mdl_out = self._model(preprocessed_image)
        mdl_out = mdl_out.cpu()
        pred_segmap = self._postprocess_model_output(mdl_out, original_image=image)
        return pred_segmap

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        preprocessed_image = self._preprocessing_pipeline.preprocess_image(image)
        preprocessed_image = preprocessed_image.unsqueeze(0)
        return preprocessed_image

    def _postprocess_model_output(
        self, out: torch.Tensor, original_image: np.ndarray
    ) -> np.ndarray:
        out = out.squeeze()
        out = out.argmax(dim=0)
        out = out.numpy().astype(np.uint8)
        h, w = original_image.shape[:2]
        postprocessed = cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)

        return postprocessed
