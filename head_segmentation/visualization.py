import matplotlib.pyplot as plt
import numpy as np
import typing as t


class VisualizationModule:
    def __init__(
        self,
        figsize: t.Tuple[int, int] = (7, 5),
        font_size: int = 16,
        save_images=False,
    ):
        self.figsize = figsize
        self.font_size = font_size
        self.save_images = save_images

        self.cmap = "gray"

    def visualize_image(self, image: np.ndarray):
        f = plt.figure(figsize=self.figsize)
        plt.imshow(image, cmap=self.cmap)
        return f

    def visualize_prediction(self, image: np.ndarray, pred_segmap: np.ndarray):
        segmented_region = np.zeros(image.shape, dtype=np.uint8)
        segmented_region[pred_segmap == 1] = image[pred_segmap == 1]

        figsize = (3 * self.figsize[0], 2 * self.figsize[1])

        f, ax = plt.subplots(1, 3, figsize=figsize)
        f.tight_layout()

        ax[0].imshow(image)
        ax[0].set_title("Input image", fontsize=self.font_size)

        ax[1].imshow(pred_segmap, cmap=self.cmap)
        ax[1].set_title("Predicted segmentation map", fontsize=self.font_size)

        ax[2].imshow(segmented_region)
        ax[2].set_title("Segmented region image", fontsize=self.font_size)

        if self.save_images:
            plt.savefig("out.png")

        return f, ax
