import os

os.environ["SM_FRAMEWORK"] = "tf.keras"

import segmentation_models as sm
import tensorflow as tf

model = tf.keras.models.load_model(
    "mlruns/6/daa278a47c854bd2ba0096bea592f4ec/artifacts/fpn/data/model", compile=False
)
