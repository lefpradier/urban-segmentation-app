import os
import cv2
import numpy as np
from flask import Flask
from matplotlib.figure import Figure


#!1.FCT RENVOIE LIST IMG DANS UN DOSSIER
def listfiles(path, suffix=".png"):
    files = [
        f
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f)) and f.endswith(suffix)
    ]
    return files


#!2.FCT CALL PREDICT ROUTE AND LOAD IMG
def load_resize_img(image_idx, path, width=256, height=128, mask=False):
    image = cv2.imread(os.path.join(path, image_idx))
    image = cv2.resize(image, (width, height))
    if mask:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.squeeze(image)
        image = convert_ground_truth(image)
    else:
        image = image / 255.0
    return image


#!3.PLOTIMG AVEC MATPLOTLIB
# imshow sur img et de la sauvegarder poour la apsser au buffer puis de l'afficher dans la route predict
def plot_image(image, mask, prediction, name):
    # Generate the figure **without using pyplot**.
    fig = Figure(figsize=(12, 3))  # , facecolor="#111111")
    fig.suptitle(name, color="white", weight="bold")
    axs = fig.subplots(1, 3)
    axs[0].set_title("Selected image", color="white", y=-0.2)
    axs[0].imshow(image[:, :, ::-1])
    axs[1].set_title("Ground truth", color="white", y=-0.2)
    axs[1].imshow(mask)
    axs[2].set_title("Predicted segmentation", color="white", y=-0.2)
    axs[2].imshow(prediction)
    for i in range(3):
        axs[i].axes.xaxis.set_visible(False)
        axs[i].axes.yaxis.set_visible(False)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # Save it to a temporary buffer.
    fig.savefig("static/plot.png", format="png", transparent=True)


#!4. CONVERT MODEL RESPONSE TO MASK IMG
def convert_response(x):
    x = np.squeeze(x)
    template = np.zeros((128, 256))
    for layer in range(8):
        template = np.where(x[:, :, layer] > 0.5, layer, template)
    return template


#!5. REDUCE NUMBER OF CATEGORIES IN MASK
def convert_ground_truth(x):
    cats = [
        [0, 1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10],
        [11, 12, 13, 14, 15, 16],
        [17, 18, 19, 20],
        [21, 22],
        [23],
        [24, 25],
        [26, 27, 28, 29, 30, 31, 32, 33, -1],
    ]
    for layer in range(8):
        x = np.where(np.isin(x, cats[layer]), layer, x)
    return x
