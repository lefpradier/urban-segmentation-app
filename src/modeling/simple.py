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
import segmentation_models as sm
from keras_unet.models import vanilla_unet
import glob
from data_generator import DataGenerator

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"


@hydra.main(version_base=None, config_path="../../config", config_name="simple")
def makerun(cfg: DictConfig):
    # pass user_config as sys.arg to merge config files
    if cfg.user_config is not None:
        user_config = OmegaConf.load(cfg.user_config)
        cfg = OmegaConf.merge(cfg, user_config)
    mlflow.set_tracking_uri("sqlite:///log.db")
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # DATASETS
    training_generator = DataGenerator(
        img_list=[f for f in glob.iglob(cfg.data.trainX + "**/*.png", recursive=True)],
        mask_list=[
            f for f in glob.iglob(cfg.data.trainY + "**/*labelIds.png", recursive=True)
        ],
        batch_size=cfg.model.batch_size,
        shuffle=True,
        aug_list=None,
        img_height=cfg.data.input_height,
        img_width=cfg.data.input_width,
    )
    validation_generator = DataGenerator(
        img_list=[f for f in glob.iglob(cfg.data.validX + "**/*.png", recursive=True)],
        mask_list=[
            f for f in glob.iglob(cfg.data.validY + "**/*labelIds.png", recursive=True)
        ],
        batch_size=cfg.model.batch_size,
        shuffle=True,
        aug_list=None,
        img_height=cfg.data.input_height,
        img_width=cfg.data.input_width,
    )

    #! get model architecture
    model = vanilla_unet(input_shape=(cfg.data.input_height, cfg.data.input_width, 3))

    #! custom loss fct
    loss = getattr(sm.losses, cfg.model.loss_function)

    #! model architecture : compile
    model.compile(
        loss=loss(),
        optimizer=cfg.model.optimizer,
        metrics=[sm.metrics.IOUScore(), sm.metrics.FScore()],
    )
    print(model.summary())

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

        #! FIT MODEL
        history = model.fit_generator(
            generator=training_generator,
            validation_data=validation_generator,
            use_multiprocessing=True,
            workers=cfg.generator.workers,
            epochs=cfg.model.no_epochs,
            verbose=cfg.model.verbosity,
        )
        time_spent = time.time() - start

        # ! PLOT TRAINING AND VALIDATION SCORES
        plt.style.use("custom_dark")
        fig, ax = plt.subplots(1)
        ax.plot(history.history["FScore"], label="F_Score(Train)")
        ax.plot(history.history["val_IOUScore"], label="IOU_Score(Validation)")
        ax.plot(history.history["IOUScore"], label="IOU_Score(Train)")
        ax.plot(history.history["val_FScore"], label="F_Score(Validation)")
        ax.plot(history.history["loss"], label="Loss(Train)")
        ax.plot(history.history["val_loss"], label="Loss(Validation)")
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
            model, registered_model_name=cfg.model.name, artifact_path="SIMPLE"
        )


# execute fct
if __name__ == "__main__":
    makerun()
