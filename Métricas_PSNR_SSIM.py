import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import os
import argparse # Biblioteca para argumentos de linha de comando (opcional, mas boa prática)

def calcular_metricas(caminho_img_original, caminho_img_reconstruida):
    """
    Calcula as métricas PSNR e SSIM entre uma imagem original e uma reconstruída.

    Args:
        caminho_img_original (str): O caminho completo para o arquivo da imagem original.
        caminho_img_reconstruida (str): O caminho completo para o arquivo da imagem reconstruída.

    Returns:
        dict: Um dicionário contendo os valores de 'PSNR' e 'SSIM', ou None se ocorrer um erro.
    """
    try:
        # Verifica se os arquivos existem antes de tentar carregá-los
        if not os.path.exists(caminho_img_original):
            print(f"Erro: Arquivo da imagem original não encontrado em '{caminho_img_original}'")
            return None
        if not os.path.exists(caminho_img_reconstruida):
            print(f"Erro: Arquivo da imagem reconstruída não encontrado em '{caminho_img_reconstruida}'")
            return None

        # Carrega as imagens usando Pillow e converte para escala de cinza ('L')
        # para garantir a consistência com o processamento do MNIST.
        img_original_pil = Image.open(caminho_img_original).convert('L')
        img_reconst_pil = Image.open(caminho_img_reconstruida).convert('L')

        # Converte as imagens para arrays numpy
        img_original = np.array(img_original_pil)
        img_reconst = np.array(img_reconst_pil)

        # Garante que os arrays tenham as mesmas dimensões
        if img_original.shape != img_reconst.shape:
            print(f"Erro: As imagens devem ter as mesmas dimensões. "
                  f"Original: {img_original.shape}, Reconstruída: {img_reconst.shape}")
            return None

        # Calcula o PSNR
        # data_range é o valor máximo do pixel (255 para imagens de 8 bits)
        valor_psnr = psnr(img_original, img_reconst, data_range=255)

        # Calcula o SSIM
        valor_ssim = ssim(img_original, img_reconst, data_range=255)

        return {"PSNR": valor_psnr, "SSIM": valor_ssim}

    except Exception as e:
        print(f"Ocorreu um erro inesperado durante o cálculo das métricas: {e}")
        return None


if __name__ == "__main__":
    
    # 1. Obter o diretório onde o script atual está localizado.
    diretorio_do_script = os.path.dirname(os.path.abspath(__file__))

    # 2. Definir os nomes dos arquivos que queremos encontrar.
    nome_imagem_original = "digit_3.png"
    nome_imagem_reconstruida = "digit_3_reconstructed.png"

    # 3. Criar o caminho completo e absoluto para cada imagem.
    caminho_original_completo = os.path.join(diretorio_do_script, nome_imagem_original)
    caminho_reconst_completo = os.path.join(diretorio_do_script, nome_imagem_reconstruida)
    
    # -------------------------------------------------------------

    # 4. Chamar a função de cálculo usando os caminhos completos.
    metricas = calcular_metricas(caminho_original_completo, caminho_reconst_completo)

    # 5. Exibir os resultados se o cálculo foi bem-sucedido.
    if metricas:
        print("--- Métricas de Qualidade da Imagem ---")
        print(f"  Imagem Original:      {nome_imagem_original}")
        print(f"  Imagem Reconstruída:  {nome_imagem_reconstruida}")
        print("---------------------------------------")
        print(f"  PSNR: {metricas['PSNR']:.2f} dB")
        print(f"  SSIM: {metricas['SSIM']:.4f}")
        print("---------------------------------------")
        print("\nLembretes:")
        print("  - PSNR (Peak Signal-to-Noise Ratio): Quanto maior, melhor (menor o erro). Valores acima de 30 dB são considerados bons.")
        print("  - SSIM (Structural Similarity Index): Quanto mais próximo de 1, melhor (mais similaridade estrutural).")