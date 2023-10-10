import tensorflow as tf
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import os

os.environ["SM_FRAMEWORK"] = "tf.keras"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import segmentation_models as sm


@hydra.main(version_base=None, config_path="../../config", config_name="serving_model")
def main(cfg):
    # # convert to TF
    model = tf.keras.models.load_model(cfg.model_ref, compile=False)
    model.compile(
        "Adam",
        loss=sm.losses.DiceLoss,
        metrics=[sm.metrics.FScore, sm.metrics.IOUScore],
    )
    concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    concrete_func.inputs[0].set_shape([None, 128, 256, 3])
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
    ]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()
    with open("deployment/backend/model.tflite", "wb") as handle:
        handle.write(tflite_model)


# execute fct
if __name__ == "__main__":
    main()
