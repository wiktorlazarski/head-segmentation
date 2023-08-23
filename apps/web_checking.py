import numpy as np
import streamlit as st
from PIL import Image

import head_segmentation.segmentation_pipeline as seg_pipeline
import head_segmentation.visualization as vis


@st.cache_resource()
def load_model() -> seg_pipeline.HumanHeadSegmentationPipeline:
    return seg_pipeline.HumanHeadSegmentationPipeline()


@st.cache_resource()
def create_vis_module() -> vis.VisualizationModule:
    return vis.VisualizationModule()


def main() -> None:
    st.write("# 👦 Human Head Segmentation")

    pipeline = load_model()
    visualizer = create_vis_module()

    image_path = st.file_uploader("Upload your image")

    if image_path is not None:
        image = np.asarray(Image.open(image_path))
        if image.shape[-1] > 3:
            image = image[..., :3]

        segmap = pipeline.predict(image)

        figure, _ = visualizer.visualize_prediction(image, segmap)

        st.write("### Segmentation result")
        st.pyplot(figure)


if __name__ == "__main__":
    main()
