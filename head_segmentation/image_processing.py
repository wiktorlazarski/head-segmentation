import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class PreprocessingPipeline:

    IMAGENET_MEANS = [0.485, 0.456, 0.406]
    IMAGENET_STDS = [0.229, 0.224, 0.225]

    def __init__(self, nn_input_image_size: int):
        self.nn_input_image_size = nn_input_image_size
        self.image_preprocessing_pipeline = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(nn_input_image_size),
                transforms.Normalize(
                    mean=PreprocessingPipeline.IMAGENET_MEANS,
                    std=PreprocessingPipeline.IMAGENET_STDS,
                ),
            ]
        )

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        return self.preprocess_image(image)

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        return self.image_preprocessing_pipeline(Image.fromarray(image))

    def preprocess_segmap(self, segmap: np.ndarray) -> torch.Tensor:
        output_dim = (self.nn_input_image_size, self.nn_input_image_size)

        preprocessed_segmap = cv2.resize(segmap, output_dim)

        return torch.Tensor(preprocessed_segmap)
