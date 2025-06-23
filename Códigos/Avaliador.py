import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Função para calcular PSNR
def calculate_psnr(img1, img2):
    return cv2.PSNR(img1, img2)

# Função para calcular SSIM
def calculate_ssim(img1, img2):
    # Converte para escala de cinza, se necessário
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ssim_value, _ = ssim(img1, img2, full=True)
    return ssim_value

# Caminhos das imagens
path_bmp = 'digit_7.bmp'
path_png1 = 'digit_7.png'  # Referência
path_jpg = 'digit_7.jpg'
path_png2 = 'digit_7_reconstructed.png'

# Carregar as imagens
img_bmp = cv2.imread(path_bmp)
img_png1 = cv2.imread(path_png1)
img_jpg = cv2.imread(path_jpg)
img_png2 = cv2.imread(path_png2)


# Lista de comparação
imagens = {
    'BMP': img_bmp,
    'JPG': img_jpg,
    'PNG2': img_png2
}

# Calcular e exibir PSNR e SSIM para cada imagem em relação à PNG1
for nome, img in imagens.items():
    psnr_value = calculate_psnr(img_png1, img)
    ssim_value = calculate_ssim(img_png1, img)
    print(f'Comparando PNG1 com {nome}:')
    print(f'  PSNR: {psnr_value:.2f} dB')
    print(f'  SSIM: {ssim_value:.4f}')
    print('--------------------------------')
