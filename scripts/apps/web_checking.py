import numpy as np
import streamlit as st
from PIL import Image

import head_segmentation.predict_pipeline as pred_pipe
import head_segmentation.visualization as vis


@st.cache(allow_output_mutation=True)
def load_model() -> pred_pipe.HumanHeadSegmentationPipeline:
    return pred_pipe.HumanHeadSegmentationPipeline()


@st.cache(allow_output_mutation=True)
def create_vis_module() -> vis.VisualizationModule:
    return vis.VisualizationModule()


def main() -> None:
    st.write("# 👦 Human Head Segmentation")

    seg_pipeline = load_model()
    visualizer = create_vis_module()

    image_path = st.file_uploader("Upload your image")

    if image_path is not None:
        image = np.asarray(Image.open(image_path))
        if image.shape[-1] > 3:
            image = image[..., :3]

        segmap = seg_pipeline.predict(image)

        figure, _ = visualizer.visualize_prediction(image, segmap)

        st.write("### Segmentation result")
        st.pyplot(figure)


if __name__ == "__main__":
    main()
