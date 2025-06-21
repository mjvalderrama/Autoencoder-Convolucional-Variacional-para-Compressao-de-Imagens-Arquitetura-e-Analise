import numpy as np
from tensorflow import keras
from PIL import Image

(_, _), (x_test, _) = keras.datasets.mnist.load_data()
x_test = x_test.astype("uint8")

for i in range(2):
    img = Image.fromarray(x_test[i])
    img.save(f"mnist_teste_{i}.png")