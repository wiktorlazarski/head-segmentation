import os
import typing as t

###### DATASET ######

LABEL2INDEX = {"background": 0, "head": 1}
INDEX2LABEL = {0: "background", 1: "head"}

###### MODEL ######

HEAD_SEGMENTATION_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model",
    "best_model_from_experiments.ckpt",
)
