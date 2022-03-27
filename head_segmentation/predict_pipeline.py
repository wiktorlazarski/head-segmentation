from .image_processing import PreprocessingPipeline
from .model import HeadSegmentationModel
from .constants import HEAD_SEGMENTATION_MODEL_PATH
import cv2
import numpy as np


class PredictPipeline:
    def __init__(self, model_path=HEAD_SEGMENTATION_MODEL_PATH):

        self.preprocessing_pipeline = PreprocessingPipeline(
            nn_image_input_resolution=512
        )
        self.model = HeadSegmentationModel.load_from_checkpoint(ckpt_path=model_path)
        self.model.eval()

    def preprocess_image(self, image):
        preprocessed_image = self.preprocessing_pipeline.preprocess_image(image)
        preprocessed_image = preprocessed_image.unsqueeze(0)
        return preprocessed_image

    def postprocess_image(self, out, original_image):
        out = out.squeeze()
        out = out.argmax(dim=0)
        out = out.numpy().astype(np.uint8)
        h, w = original_image.shape[:2]
        postprocessed = cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)

        return postprocessed

    def predict(self, image):
        preprocessed_image = self.preprocess_image(image)
        mdl_out = self.model(preprocessed_image)
        pred_segmap = self.postprocess_image(mdl_out, original_image=image)
        return pred_segmap
