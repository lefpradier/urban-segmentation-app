import numpy as np
import tensorflow as tf
import logging
import mlflow
import matplotlib.pyplot as plt
from labellines import labelLines
import hydra
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
import pandas as pd
import time
import os
import sys
from pathlib import Path

sys.path.append("src/modeling/")

os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
from keras_unet.models import custom_unet, vanilla_unet
import glob
from data_generator import DataGenerator

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"


@hydra.main(
    version_base=None, config_path="../../config", config_name="pretrained_unet"
)
def makerun(cfg: DictConfig):
    # pass user_config as sys.arg to merge config files
    if cfg.user_config is not None:
        user_config = OmegaConf.load(cfg.user_config)
        cfg = OmegaConf.merge(cfg, user_config)
    mlflow.set_tracking_uri("sqlite:///log.db")
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # DATASETS
    if cfg.generator.auglist is not None:
        aug_list = cfg.generator.auglist.split("_")
    else:
        aug_list = None
    x_train = [str(f) for f in Path(cfg.data.trainX).rglob("*.png")]
    y_train = [str(f) for f in Path(cfg.data.trainY).rglob("*labelIds.png")]
    x_valid = [str(f) for f in Path(cfg.data.validX).rglob("*.png")]
    y_valid = [str(f) for f in Path(cfg.data.validY).rglob("*labelIds.png")]
    x_train.sort()
    y_train.sort()
    x_valid.sort()
    y_valid.sort()
    training_generator = DataGenerator(
        img_list=x_train,
        mask_list=y_train,
        batch_size=cfg.generator.batch_size,
        shuffle=True,
        aug_list=aug_list,
        img_height=cfg.data.input_height,
        img_width=cfg.data.input_width,
    )
    validation_generator = DataGenerator(
        img_list=x_valid,
        mask_list=y_valid,
        batch_size=cfg.generator.batch_size,
        shuffle=True,
        img_height=cfg.data.input_height,
        img_width=cfg.data.input_width,
    )

    #! get model architecture
    model = sm.PSPNet(
        cfg.model.backbone,
        input_shape=(cfg.data.input_height, cfg.data.input_width, 3),
        classes=8,
        activation="softmax",
    )

    #! custom loss fct
    loss = getattr(sm.losses, cfg.model.loss_function)

    #! model architecture : compile
    # ?class weight based on imbalance, cf test_init
    #!Define class weights betwee 0-1
    # w = [
    #     1 / 0.106**2,
    #     1 / 0.389**2,
    #     1 / 0.218**2,
    #     1 / 0.018**2,
    #     1 / 0.145**2,
    #     1 / 0.036**2,
    #     1 / 0.012**2,
    #     1 / 0.075**2,
    # ]

    #!regularization of all layers
    # model = set_regularization(model, kernel_regularizer=tf.keras.regularizers.l2(0.01))

    #!COMPILE
    model.compile(
        # loss=focal_tversky_loss,
        loss=loss(),  # class_weights=[x / np.sum(w) for x in w]),
        optimizer=cfg.model.optimizer,  # tf.keras.optimizers.Adam(weight_decay=0.004)
        metrics=[sm.metrics.IOUScore(), sm.metrics.FScore()],
        run_eagerly=True,
    )
    print(model.summary())

    #!START RUN
    with mlflow.start_run(run_name=cfg.mlflow.run_name) as run:
        params = {
            "n_epoch": cfg.model.no_epochs,
            "batch_size": cfg.generator.batch_size,
            "optimizer": cfg.model.optimizer,
            "loss_function": cfg.model.loss_function,
            "model_type": cfg.model.model_type,
            "backbone": cfg.model.backbone,
        }

        #! suivie params mlflow
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
        print(history)

        # ! PLOT TRAINING AND VALIDATION SCORES
        plt.style.use("custom_dark")
        fig, ax = plt.subplots(1)
        ax.plot(history.history["f1-score"], label="FScore(Train)")
        ax.plot(history.history["val_iou_score"], label="val_IOUScore(Validation)")
        ax.plot(history.history["iou_score"], label="IOUScore(Train)")
        ax.plot(history.history["val_f1-score"], label="val_FScore(Validation)")
        ax.plot(history.history["loss"], label="Loss(Train)")
        ax.plot(history.history["val_loss"], label="val_loss(Validation)")
        ax.set_title("Scores and loss over epochs")
        ax.set_ylabel("Value")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Epoch")
        labelLines(ax.get_lines())
        fig.savefig("plots/scores_epoch_%s.png" % cfg.mlflow.run_name)
        mlflow.log_artifact("plots/scores_epoch_%s.png" % cfg.mlflow.run_name)

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
            model, registered_model_name=cfg.model.name, artifact_path="PSPNET"
        )


# execute fct
if __name__ == "__main__":
    makerun()
