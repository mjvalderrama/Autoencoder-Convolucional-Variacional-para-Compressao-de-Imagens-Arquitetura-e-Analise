import numpy as np
from tensorflow import keras
from PIL import Image

# Carregar decoder treinado
decoder = keras.models.load_model("vae_decoder.h5", compile=False)

# Carregar vetor comprimido
z = np.load("compressed_digit_3.npy")

# Decodificar (reconstruir)
reconstructed = decoder.predict(z)
reconstructed_img = (reconstructed[0, :, :, 0] * 255).astype("uint8")
img = Image.fromarray(reconstructed_img)
img.save("reconstructed_digit_3.png")