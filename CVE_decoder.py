import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Opcional: suprime avisos de performance da Intel oneDNN.

import numpy as np
from tensorflow import keras
from PIL import Image

if __name__ == "__main__":

    try:
        # Constrói caminhos absolutos para os arquivos, baseados na localização deste script.
        # Isso garante que o script funcione independentemente de onde for executado.
        diretorio_do_script = os.path.dirname(os.path.abspath(__file__))
        
        # Define os nomes dos arquivos de entrada e saída
        nome_decoder = "CVAE_decoder_train.h5"
        nome_vetor_comprimido = "digit_3_compressed.npy"
        nome_imagem_reconstruida = "digit_3_reconstructed.png" # Alterado para um nome mais claro

        # Cria os caminhos completos
        caminho_decoder = os.path.join(diretorio_do_script, nome_decoder)
        caminho_vetor = os.path.join(diretorio_do_script, nome_vetor_comprimido)
        caminho_saida = os.path.join(diretorio_do_script, nome_imagem_reconstruida)

        # --- CARREGAMENTO DOS ARQUIVOS DE ENTRADA ---

        # Verifica e carrega o modelo do decoder
        print(f"INFO: Carregando decoder de '{caminho_decoder}'...")
        if not os.path.exists(caminho_decoder):
            raise FileNotFoundError(f"ERRO: Arquivo do decoder não encontrado: {caminho_decoder}")
        decoder = keras.models.load_model(caminho_decoder, compile=False)
        print("✅ SUCESSO: Decoder carregado.")

        # Verifica e carrega o vetor comprimido
        print(f"INFO: Carregando vetor comprimido de '{caminho_vetor}'...")
        if not os.path.exists(caminho_vetor):
            raise FileNotFoundError(f"ERRO: Arquivo do vetor comprimido não encontrado: {caminho_vetor}")
        z = np.load(caminho_vetor)
        print("✅ SUCESSO: Vetor comprimido carregado.")

        # --- DECODIFICAÇÃO E SALVAMENTO DA IMAGEM ---

        print("INFO: Decodificando o vetor para reconstruir a imagem...")
        # Usa o decoder para prever (reconstruir) a imagem a partir do vetor latente 'z'
        reconstructed_array = decoder.predict(z)
        
        # Converte o array numpy de volta para uma imagem visível
        # [0, :, :, 0] seleciona a primeira (e única) imagem do lote
        # e o primeiro (e único) canal de cor.
        reconstructed_img_data = (reconstructed_array[0, :, :, 0] * 255).astype("uint8")
        
        # Cria e salva a imagem reconstruída
        img_reconstruida = Image.fromarray(reconstructed_img_data)
        img_reconstruida.save(caminho_saida)

        print(f"\n✅ SUCESSO FINAL: Imagem reconstruída e salva como '{nome_imagem_reconstruida}'!")

    except Exception as e:
        print(f"\n❌ Ocorreu um erro durante a execução: {e}")