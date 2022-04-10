______________________________________________________________________
<div align="center">

# 👦 Human Head Semantic Segmentation

<p align="center">
  <a href="https://github.com/wiktorlazarski">🧑‍🎓 Wiktor</a>
  <a href="https://github.com/Szuumii">🧑‍🎓 Kuba</a>
</p>

______________________________________________________________________

[![ci-testing](https://github.com/wiktorlazarski/head-segmentation/actions/workflows/ci-testing.yml/badge.svg?branch=master&event=push)](https://github.com/wiktorlazarski/head-segmentation/actions/workflows/ci-testing.yml)
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Wj090FmpbK2IO2-qa4tBuzHKrxjlQItM?usp=sharing)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/wiktorlazarski/head-segmentation/blob/master/LICENSE)

</div>

## 💎 Installation with `pip`

Installation is as simple as running:

```bash
pip install git+https://github.com/wiktorlazarski/head-segmentation.git
```

## 🔨 How to use

### 🤔 Inference
```python
import head_segmentation.predict_pipeline as pred_pipeline

segmentation_pipeline = pred_pipeline.HumanHeadSegmentationPipeline()

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

![alt text](https://github.com/wiktorlazarski/head-segmentation/blob/master/doc/images/wiktor.png)
![alt text](https://github.com/wiktorlazarski/head-segmentation/blob/master/doc/images/kuba.png)
![alt text](https://github.com/wiktorlazarski/head-segmentation/blob/master/doc/images/wiktor_with_glasses.png)
![alt text](https://github.com/wiktorlazarski/head-segmentation/blob/master/doc/images/kuba_with_helmet.png)

If you want to check predictions on some of your images, please feel free to use our Streamlit application.

```bash
cd head-segmentation

streamlit run ./scripts/apps/web_checking.py
```

<div align="center">

### 🤗 Enjoy the model!

</div>

