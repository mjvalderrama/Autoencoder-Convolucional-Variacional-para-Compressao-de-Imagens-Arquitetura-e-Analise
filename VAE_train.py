import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Carregar dataset MNIST
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.
x_train = np.expand_dims(x_train, -1)
x_test = x_test.astype("float32") / 255.
x_test = np.expand_dims(x_test, -1)


# Variaveis
latent_dim = 16
batch_size = 128
epochs = 50


# Encoder
encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Flatten()(encoder_inputs)
x = layers.Dense(512, activation="relu")(x)      # 1ª camada
x = layers.Dense(256, activation="relu")(x)      # 2ª camada
x = layers.Dense(128, activation="relu")(x)      # 3ª camada
x = layers.Dense(64, activation="relu")(x)       # 4ª camada
x = layers.Dense(32, activation="relu")(x)       # 5ª camada
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

@keras.utils.register_keras_serializable()
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

# Decoder
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(32, activation="relu")(latent_inputs)           # 1ª camada
x = layers.Dense(64, activation="relu")(x)                       # 2ª camada
x = layers.Dense(128, activation="relu")(x)                      # 3ª camada
x = layers.Dense(256, activation="relu")(x)                      # 4ª camada
x = layers.Dense(512, activation="relu")(x)                      # 5ª camada
x = layers.Dense(28 * 28, activation="sigmoid")(x)
decoder_outputs = layers.Reshape((28, 28, 1))(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

# VAE Model
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": total_loss, "reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss}

def dummy_loss(y_true, y_pred):
    return 0.0

vae = VAE(encoder, decoder)
vae.compile(optimizer="adam", loss=dummy_loss)
vae.fit(x_train, epochs = epochs, batch_size = batch_size, validation_data=(x_test, x_test))

# Salvar modelos
encoder.save("vae_encoder.h5")
decoder.save("vae_decoder.h5")