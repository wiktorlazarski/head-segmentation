import head_segmentation.image_processing as ip
import head_segmentation.model as mdl
import head_segmentation.constants as C
import cv2
import numpy as np
import torch


class HumanHeadSegmentationPipeline:
    def __init__(
        self,
        model_path: str = C.HEAD_SEGMENTATION_MODEL_PATH,
        image_input_resolution: int = 512,
    ):

        self._preprocessing_pipeline = ip.PreprocessingPipeline(
            nn_image_input_resolution=image_input_resolution
        )
        self._model = mdl.HeadSegmentationModel.load_from_checkpoint(
            ckpt_path=model_path
        )
        self._model.eval()

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

    def predict(self, image: np.ndarray) -> np.ndarray:
        preprocessed_image = self._preprocess_image(image)
        mdl_out = self._model(preprocessed_image)
        pred_segmap = self._postprocess_model_output(mdl_out, original_image=image)
        return pred_segmap
