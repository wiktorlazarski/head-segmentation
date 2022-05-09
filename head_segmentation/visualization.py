import typing as t

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class VisualizationModule:
    def __init__(
        self,
        figsize: t.Tuple[int, int] = (12, 10),
        font_size: int = 16,
    ):
        self.figsize = figsize
        self.font_size = font_size

        self.cmap = "gray"

    def visualize_image(self, image: np.ndarray) -> matplotlib.figure.Figure:
        f = plt.figure(figsize=self.figsize)
        plt.imshow(image, cmap=self.cmap)
        return f

    def visualize_prediction(
        self, image: np.ndarray, pred_segmap: np.ndarray, save_images=False
    ) -> t.Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        segmented_region = image * cv2.cvtColor(pred_segmap, cv2.COLOR_GRAY2RGB)

        figsize = (3 * self.figsize[0], 2 * self.figsize[1])

        f, ax = plt.subplots(1, 3, figsize=figsize)
        f.tight_layout()

        ax[0].imshow(image)
        ax[0].set_title("Input image", fontsize=self.font_size)

        ax[1].imshow(pred_segmap, cmap=self.cmap)
        ax[1].set_title("Predicted segmentation map", fontsize=self.font_size)

        ax[2].imshow(segmented_region)
        ax[2].set_title("Cropped image region", fontsize=self.font_size)

        if save_images:
            plt.savefig("out.png")

        return f, ax
