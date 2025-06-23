import cv2
import numpy as np
import os

# Ler imagem
input_path = 'Dog.png'
image = cv2.imread(input_path)

if image is None:
    print('Erro ao carregar a imagem.')
    exit()

# Redimensionar para 256x256
image_resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

# Kernel Sharpen (realce de características)
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]], dtype=np.float32)

# Tamanho da borda (mais exagerado para visualização)
border_size = 40  # Aumente este valor para deixar ainda maior

# Criar diferentes versões com diferentes paddings
padded_constant = cv2.copyMakeBorder(image_resized, border_size, border_size, border_size, border_size,
                                     borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])  # Zero padding (preto)

padded_reflect = cv2.copyMakeBorder(image_resized, border_size, border_size, border_size, border_size,
                                    borderType=cv2.BORDER_REFLECT)

padded_replicate = cv2.copyMakeBorder(image_resized, border_size, border_size, border_size, border_size,
                                      borderType=cv2.BORDER_REPLICATE)

# Aplicar o filtro em cada imagem com borda
result_constant = cv2.filter2D(padded_constant, -1, kernel)
result_reflect = cv2.filter2D(padded_reflect, -1, kernel)
result_replicate = cv2.filter2D(padded_replicate, -1, kernel)

result_constant = cv2.resize(result_constant, (256, 256), interpolation=cv2.INTER_AREA)
result_reflect = cv2.resize(result_reflect, (256, 256), interpolation=cv2.INTER_AREA)
result_replicate = cv2.resize(result_replicate, (256, 256), interpolation=cv2.INTER_AREA)


# Mostrar os resultados
cv2.imshow('Constant Padding (Zero)', result_constant)
cv2.imshow('Reflect Padding', result_reflect)
cv2.imshow('Replicate Padding', result_replicate)

cv2.imwrite('result_constant.png', result_constant)
cv2.imwrite('result_reflect.png', result_reflect)
cv2.imwrite('result_replicate.png', result_replicate)



cv2.waitKey(0)
cv2.destroyAllWindows()
