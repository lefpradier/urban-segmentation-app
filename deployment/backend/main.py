import sys

# sys.path.insert(1, "backend")
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import uuid
import uvicorn
from fastapi import FastAPI
import numpy as np
import asyncio
import json
from pydantic import BaseModel

# from fastapi.exceptions import RequestValidationError
# from fastapi.responses import PlainTextResponse
import tensorflow as tf

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Test model on random input data.
input_shape = input_details[0]["shape"]


app = FastAPI()


class Item(BaseModel):
    image: str


# PING
@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}


# predict
@app.post("/segment")
async def predict_stm(item: Item):
    image = item.image
    input_data = np.asarray(json.loads(image))
    input_data = np.float32(input_data)  # convert to float32
    input_data = np.expand_dims(
        np.asarray(input_data), 0
    )  # increase the number of dimensions to simulate a batch of size 1
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    pred = interpreter.get_tensor(output_details[0]["index"])
    return json.dumps(pred.tolist())


# @app.exception_handler(RequestValidationError)
# async def validation_exception_handler(request, ex):
#     return PlainTextResponse(str(ex), status_code=400)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
