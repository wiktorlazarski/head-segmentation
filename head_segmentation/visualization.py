import matplotlib.pyplot as plt
import numpy as np


class VisualizationModule:
    def __init__(self, figsize: (int, int) = (16, 12), font_size: int = 48):
        self.figsize = figsize
        self.font_size = font_size

        self.cmap = "gray"
        self.facecolor = "white"

    def visualize_image(self, image):
        plt.imshow(image, cmap=self.cmap)

    def visualize_prediction(self, image, pred_segmap):
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

        plt.savefig("out.png")
        plt.show()
