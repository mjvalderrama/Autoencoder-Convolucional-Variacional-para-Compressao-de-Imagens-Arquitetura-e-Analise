import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Opcional: suprime avisos de performance da Intel oneDNN.

import numpy as np
from tensorflow import keras
from PIL import Image
import tensorflow as tf

@keras.utils.register_keras_serializable()
def sampling(args):
    """Função de reparametrização (sampling) para a camada latente do VAE."""
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

if __name__ == "__main__":
    
    try:
        # Constrói caminhos absolutos para os arquivos, baseados na localização deste script.
        # Isso torna o script executável de qualquer diretório.
        diretorio_do_script = os.path.dirname(os.path.abspath(__file__))
        caminho_modelo = os.path.join(diretorio_do_script, "CVAE_encoder_train.h5")
        caminho_imagem = os.path.join(diretorio_do_script, "digit_3.png")

        # Verifica a existência dos arquivos necessários.
        if not os.path.exists(caminho_modelo):
            raise FileNotFoundError(f"ERRO: Arquivo do modelo não encontrado: {caminho_modelo}")
        if not os.path.exists(caminho_imagem):
            raise FileNotFoundError(f"ERRO: Arquivo de imagem não encontrado: {caminho_imagem}")

        # Carrega o encoder, informando ao Keras sobre a função personalizada 'sampling'.
        # Isso é necessário para modelos com camadas Lambda ou funções customizadas.
        encoder = keras.models.load_model(
            caminho_modelo,
            custom_objects={'sampling': sampling},
            compile=False
        )

        # Carrega e pré-processa a imagem de teste.
        img = Image.open(caminho_imagem).convert("L").resize((28, 28))
        img_arr = np.array(img).astype("float32") / 255.
        img_arr = np.expand_dims(img_arr, axis=(0, -1))

        # Codifica a imagem.
        # IMPORTANTE: A linha abaixo assume que o modelo 'encoder' foi salvo com 3 saídas.
        # Se o modelo original tiver um número diferente de saídas, esta linha causará um erro.
        z_mean, z_log_var, z = encoder.predict(img_arr)
        
        # Salva o vetor latente 'z' (a representação comprimida da imagem).
        caminho_arquivo_comprimido = os.path.join(diretorio_do_script, "digit_3_compressed.npy")
        np.save(caminho_arquivo_comprimido, z)
        
        print(f"✅ SUCESSO: Imagem codificada e salva como 'digit_3_compressed.npy'")
        print(f"   - Forma do vetor latente salvo: {z.shape}")

    except Exception as e:
        print(f"\n❌ Ocorreu um erro durante a execução: {e}")