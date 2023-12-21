______________________________________________________________________
<div align="center">

# 👦 Human Head Semantic Segmentation

<p align="center">
  <a href="https://github.com/wiktorlazarski">🧑‍🎓 Wiktor</a>
  <a href="https://github.com/Szuumii">🧑‍🎓 Kuba</a>
</p>

______________________________________________________________________

[![ci-testing](https://github.com/wiktorlazarski/head-segmentation/actions/workflows/ci-testing.yml/badge.svg?branch=main&event=push)](https://github.com/wiktorlazarski/head-segmentation/actions/workflows/ci-testing.yml)
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QScgxBRXWbGbQ3DmYIJ2Ja3sD_mJ_Efx?usp=sharing)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

## 💎 Installation with `pip`

Installation is as simple as running:

```bash
pip install git+https://github.com/wiktorlazarski/head-segmentation.git
```

## 🔨 How to use

### 🤔 Inference
```python
import head_segmentation.segmentation_pipeline as seg_pipeline

segmentation_pipeline = seg_pipeline.HumanHeadSegmentationPipeline()

segmentation_map = segmentation_pipeline.predict(image)
```

### 🎨 Visualizing

```python
import matplotlib.pyplot as plt

import head_segmentation.visualization as vis

visualizer = vis.VisualizationModule()

figure, _ = visualizer.visualize_prediction(image, segmentation_map)
plt.show()
```

## ⚙️ Setup for development

```bash
# Clone repo
git clone https://github.com/wiktorlazarski/head-segmentation.git

# Go to repo directory
cd head-segmentation

# (Optional) Create virtual environment
python -m venv venv
source ./venv/bin/activate

# Install project in editable mode
pip install -e .[dev]

# (Optional but recommended) Install pre-commit hooks to preserve code format consistency
pre-commit install
```

## 🐍 Setup for development with Anaconda or Miniconda

```bash
# Clone repo
git clone https://github.com/wiktorlazarski/head-segmentation.git

# Go to repo directory
cd head-segmentation

# Create and activate conda environment
conda env create -f ./conda_env.yml
conda activate head_segmentation

# (Optional but recommended) Install pre-commit hooks to preserve code format consistency
pre-commit install
```

## 🔬 Quantitative results

**Keep in mind** that we trained our model with CelebA dataset, which means that our model may not necessarily perform well on your data, since they may come from a different distribution than CelebA.

The table below presents results, computed on the full scale test set images, of three best models we trained. Model naming convention is as followed: `<backbone>_<nn_input_image_resultion>`.


|      Model     | mobilenetv2_256 | mobilenetv2_512 | resnet34_512 |
|:--------------:|:---------------:|:---------------:|:------------:|
|    head IoU    |     0.967606    |     0.967337    | **0.968457** |
| background IoU |     0.942936    |     0.942160    | **0.944469** |
|      mIoU      |     0.955271    |     0.954749    | **0.956463** |


## 🧐 Qualitative results

![alt text](https://github.com/wiktorlazarski/head-segmentation/blob/main/doc/images/wiktor.png)
![alt text](https://github.com/wiktorlazarski/head-segmentation/blob/main/doc/images/kuba.png)
![alt text](https://github.com/wiktorlazarski/head-segmentation/blob/main/doc/images/wiktor_with_glasses.png)
![alt text](https://github.com/wiktorlazarski/head-segmentation/blob/main/doc/images/kuba_with_helmet.png)

If you want to check predictions on some of your images, please feel free to use our Streamlit application.

```bash
cd head-segmentation

streamlit run ./scripts/apps/web_checking.py
```

## ⏰ Inference time

If you are strict with time, you can use gpu to acclerate inference. Visualization also consume some time, you can just save the final result as below.

```python
import torch
from PIL import Image
import head_segmentation.segmentation_pipeline as seg_pipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

segmentation_pipeline = seg_pipeline.HumanHeadSegmentationPipeline(device=device)

segmentation_map = segmentation_pipeline.predict(image)

segmented_region = image * cv2.cvtColor(segmentation_map, cv2.COLOR_GRAY2RGB)

pil_image = Image.fromarray(segmented_region)
pil_image.save(save_path)
```

The table below presents inference time which is tested on Tesla T4 (just for reference). The first image will take more time.

|                |       save figure     | just save final result|
|:--------------:|:---------------------:|:---------------------:|
|       cpu      |       around 2.1s     |       around 0.8s     |
|       gpu      |       around 1.4s     |       around 0.15s    |

<div align="center">

### 🤗 Enjoy the model!

</div>
