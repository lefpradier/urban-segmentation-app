import os
import sys
import subprocess


def create_dir():
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/augmentation", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("config", exist_ok=True)
    os.makedirs("pretrained", exist_ok=True)
    os.makedirs("src", exist_ok=True)
    os.makedirs("src/augmentation", exist_ok=True)
    os.makedirs("src/data_structure", exist_ok=True)
    os.makedirs("src/modeling", exist_ok=True)
    os.makedirs("deployment", exist_ok=True)


# execute fct
if __name__ == "__main__":
    create_dir()
