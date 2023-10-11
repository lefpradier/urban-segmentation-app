# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
import json
import requests
from utils import listfiles, load_resize_img, plot_image, convert_response
import numpy as np
import random
from werkzeug.exceptions import NotFound, HTTPException

# from functions import extract_keywords

####### UTILS FCT
files = listfiles("static/images/")
files = [f.split("_leftImg")[0] for f in files]
files = ["Choose a picture"] + files
print(files)
app = Flask(__name__)


#!PING
@app.route("/")
def hello():
    return "Hello World!"


#!VISUALISATION : WEB APP


@app.route("/dashboard/")
def index():
    return render_template(
        "index.html", files=files, showimage=False, showmessage=False
    )


@app.route("/predict", methods=["GET", "POST"])
# load and rezise img
def predict():
    #! 1. Import and resize image
    input_data = list(request.form.values())
    image_idx = input_data[0]
    if image_idx == files[0]:
        image_idx = random.choice([f for f in files if f != files[0]])
    image = load_resize_img(image_idx + "_leftImg8bit.png", "static/images/")
    mask = load_resize_img(
        image_idx + "_gtFine_labelIds.png", "static/masks/", mask=True
    )
    #! 2. Convert image to JSON and send to the API
    data = {"image": json.dumps(image.tolist())}
    try:
        response = requests.post(
            "https://urban-segmentation-api.azurewebsites.net/segment", json=data
        )
        #! 3. Convert prediction back to numpy array, and convert to image
        if response.ok:
            prediction = np.asarray(json.loads(response.json()))
            prediction = convert_response(prediction)
            plot_image(image, mask, prediction, image_idx)
            showimage = True
            showmessage = False
            prediction_text = ""
        else:
            prediction_text = "Error 404, please retry prediction."
            showimage = False
            showmessage = True
    except requests.exceptions.Timeout:
        showimage = False
        showmessage = True
        prediction_text = "Error. Please try again."
        # Maybe set up for a retry, or continue in a retry loop
    except requests.exceptions.TooManyRedirects:
        showimage = False
        showmessage = True
        prediction_text = "Error. Please try again."
        # Tell the user their URL was bad and try a different one
    except requests.exceptions.RequestException as e:
        showimage = False
        showmessage = True
        prediction_text = "Error. Please try again."

    return render_template(
        "index.html",
        prediction_text=prediction_text,
        files=files,
        prediction_image="static/plot.png",
        showimage=showimage,
        showmessage=showmessage,
    )


@app.errorhandler(NotFound)
def page_not_found(e: HTTPException):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port="5000")
