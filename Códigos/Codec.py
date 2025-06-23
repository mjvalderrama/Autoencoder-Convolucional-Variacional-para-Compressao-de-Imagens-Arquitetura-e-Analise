import cv2

# Caminho da imagem de entrada
input_image_path = 'digit_3.png'  # Pode ser qualquer formato (jpg, png, etc)

# Caminhos de saída
output_bmp_path = 'digit_3.bmp'
output_jpg_path = 'digit_3.jpg'

# Carrega a imagem
image = cv2.imread(input_image_path)

# Verifica se a imagem foi carregada corretamente
if image is None:
    print("Erro ao carregar a imagem. Verifique o caminho.")
else:
    # Salva em BMP
    cv2.imwrite(output_bmp_path, image)
    print(f"Imagem salva como {output_bmp_path}")

    # Salva em JPG
    cv2.imwrite(output_jpg_path, image)
    print(f"Imagem salva como {output_jpg_path}")
