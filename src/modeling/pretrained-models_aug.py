#!/usr/bin/env python3

import sys

from typing import Any
from hydra.core.utils import JobReturn
import numpy as np
import tensorflow as tf
import logging
import mlflow
import matplotlib.pyplot as plt
from labellines import labelLines
import hydra
from omegaconf import DictConfig, OmegaConf

import gc

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
import pandas as pd
import time
import os
from pathlib import Path
from hydra.experimental.callback import Callback

sys.path.append("src/modeling/")

os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
from keras_unet.models import custom_unet, vanilla_unet
import glob
from data_generator import DataGenerator

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# TODO : LIMIT MEMORY USAGE
# First, Get a list of GPU devices
gpus = tf.config.list_physical_devices("GPU")
# Restrict to only the first GPU.
tf.config.set_visible_devices(gpus[:1], device_type="GPU")
# Create a LogicalDevice with the appropriate memory limit
log_dev_conf = tf.config.LogicalDeviceConfiguration(memory_limit=9 * 1024)  # 9 GB
# # try dynamix ressources allocation
# tf.config.experimental.set_memory_growth(gpus[0], True)
# Apply the logical device configuration to the first GPU
tf.config.set_logical_device_configuration(gpus[0], [log_dev_conf])

from datetime import datetime
from packaging import version


class mycallback(Callback):
    def on_job_end(self, config: DictConfig, **kwargs: Any):
        tf.keras.backend.clear_session()
        gc.collect()
        print("job end")


@hydra.main(version_base=None, config_path="../../config", config_name="pretrained_aug")
def makerun(cfg: DictConfig):
    tf.keras.backend.clear_session()
    # pass user_config as sys.arg to merge config files
    if cfg.user_config is not None:
        user_config = OmegaConf.load(cfg.user_config)
        cfg = OmegaConf.merge(cfg, user_config)
    mlflow.set_tracking_uri("sqlite:///log.db")
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # DATASETS
    # *Transform 3 types of filter (aug_geo..) inst of aug_list
    aug_list = (
        str(cfg.generator.auglist.geo).split("_")
        + str(cfg.generator.auglist.col).split("_")
        + str(cfg.generator.auglist.ker).split("_")
    )
    aug_list = [x for x in aug_list if x != "None"]
    if len(aug_list) == 0:
        aug_list = None
    x_train = [str(f) for f in Path(cfg.data.trainX).rglob("*.png")]
    y_train = [str(f) for f in Path(cfg.data.trainY).rglob("*labelIds.png")]
    x_valid = [str(f) for f in Path(cfg.data.validX).rglob("*.png")]
    y_valid = [str(f) for f in Path(cfg.data.validY).rglob("*labelIds.png")]
    x_train.sort()
    y_train.sort()
    x_valid.sort()
    y_valid.sort()
    mosaic = False
    oversampling = False
    attention_mask = False
    attention_mask_size = False
    if cfg.generator.mosaic == "True":
        mosaic = True
    if cfg.generator.oversampling == "True":
        oversampling = True
    if cfg.generator.attention_mask == "True":
        attention_mask = True
    if cfg.generator.attention_mask_size == "True":
        attention_mask_size = True
    #! Reshape inputs for pspnet
    if cfg.model.model_type == "pspnet":
        ratio_wh = cfg.data.input_width / cfg.data.input_height
        cfg.data.input_height = int(48 * round(cfg.data.input_height / 48))
        cfg.data.input_width = int(ratio_wh * cfg.data.input_height)
    #! Reset batch_size if using efficientnetb7
    if cfg.model.backbone == "efficientnetb7":
        cfg.generator.batch_size = int(cfg.generator.batch_size / 2)

    training_generator = DataGenerator(
        img_list=x_train,
        mask_list=y_train,
        batch_size=cfg.generator.batch_size,
        shuffle=True,
        aug_list=aug_list,
        img_height=cfg.data.input_height,
        img_width=cfg.data.input_width,
        mosaic=mosaic,
        oversampling=oversampling,
        seed=cfg.generator.seed,
        clim=cfg.generator.auglist.clim,
        blim=cfg.generator.auglist.blim,
        oversampling_n=cfg.generator.oversampling_n,
        oversampling_max=cfg.generator.oversampling_max,
        attention_mask=attention_mask,
        attention_mask_size=attention_mask_size,
    )
    validation_generator = DataGenerator(
        img_list=x_valid,
        mask_list=y_valid,
        batch_size=cfg.generator.batch_size,
        shuffle=True,
        img_height=cfg.data.input_height,
        img_width=cfg.data.input_width,
    )

    # ⁡⁣⁣⁢𝗠𝗢𝗗𝗘𝗟𝗦⁡
    #! get model architecture
    if cfg.model.model_type == "unet":
        # *unet
        model = sm.Unet(
            cfg.model.backbone,
            input_shape=(cfg.data.input_height, cfg.data.input_width, 3),
            classes=8,
            activation="softmax",
        )
    elif cfg.model.model_type == "pspnet":
        # *pspnet
        model = sm.PSPNet(
            cfg.model.backbone,
            input_shape=(cfg.data.input_height, cfg.data.input_width, 3),
            classes=8,
            activation="softmax",
        )

    elif cfg.model.model_type == "fpn":
        # *fpn
        model = sm.FPN(
            cfg.model.backbone,
            input_shape=(cfg.data.input_height, cfg.data.input_width, 3),
            classes=8,
            activation="softmax",
        )

    #! custom loss fct
    loss = getattr(sm.losses, cfg.model.loss_function)

    #!COMPILE
    model.compile(
        loss=loss(),
        optimizer=cfg.model.optimizer,
        metrics=[sm.metrics.IOUScore(), sm.metrics.FScore()],
        run_eagerly=True,
    )

    #!START RUN
    with mlflow.start_run() as run:
        params = {
            "n_epoch": cfg.model.no_epochs,
            "batch_size": cfg.generator.batch_size,
            "optimizer": cfg.model.optimizer,
            "loss_function": cfg.model.loss_function,
            "model_type": cfg.model.model_type,
            "backbone": cfg.model.backbone,
            "augmentation": cfg.generator.auglist,
            "oversampling": cfg.generator.oversampling,
            "mosaic": cfg.generator.mosaic,
            "seed": cfg.generator.seed,
            "height": cfg.data.input_height,
            "width": cfg.data.input_width,
            "oversampling_n": cfg.generator.oversampling_n,
            "oversampling_max": cfg.generator.oversampling_max,
            "attention_mask_size": cfg.generator.attention_mask_size,
        }

        #! Tracking params mlflow
        mlflow.log_params(params)
        start = time.time()

        #!Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3
        )

        #! FIT MODEL
        history = model.fit(
            x=training_generator,
            validation_data=validation_generator,
            use_multiprocessing=True,
            workers=cfg.generator.workers,
            epochs=cfg.model.no_epochs,
            verbose=cfg.model.verbosity,
            callbacks=[early_stopping],
        )
        time_spent = time.time() - start

        #! EVALUATE MODEL : loss and cvscores
        # allow to test for overfitting
        scores = model.evaluate(
            validation_generator,
            use_multiprocessing=True,
            workers=cfg.generator.workers,
            verbose=0,
        )

        # dict scores
        scores = {
            "loss": scores[0],
            "IOUScore": scores[1],
            "FScore": scores[2],
            "Training_time": time_spent,
        }

        mlflow.log_metrics(scores)
        mlflow.tensorflow.log_model(
            model,
            registered_model_name=cfg.model.model_type,
            artifact_path=cfg.model.model_type,
        )
    return scores["IOUScore"], scores["FScore"]


# execute fct
if __name__ == "__main__":
    makerun()
