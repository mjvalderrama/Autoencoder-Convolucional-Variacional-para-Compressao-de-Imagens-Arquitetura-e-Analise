import cv2
import numpy as np

# Ler a imagem
input_path = 'Dog.png'
image = cv2.imread(input_path)

if image is None:
    print('Erro ao carregar a imagem.')
    exit()

# Redimensionar para 512x512
image_resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)

# Converter para escala de cinza
gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

# Aplicar Sharpen para destacar características antes do pooling
kernel_sharpen = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)
gray_sharpened = cv2.filter2D(gray, -1, kernel_sharpen)

# Função MaxPooling 2x2 com stride 2
def max_pooling(img):
    pooled = np.zeros((img.shape[0] // 2, img.shape[1] // 2), dtype=img.dtype)
    for y in range(0, img.shape[0], 2):
        for x in range(0, img.shape[1], 2):
            region = img[y:y+2, x:x+2]
            pooled[y//2, x//2] = np.max(region)
    return pooled

# Função AvgPooling 2x2 com stride 2
def avg_pooling(img):
    pooled = np.zeros((img.shape[0] // 2, img.shape[1] // 2), dtype=np.uint8)
    for y in range(0, img.shape[0], 2):
        for x in range(0, img.shape[1], 2):
            region = img[y:y+2, x:x+2]
            pooled[y//2, x//2] = np.mean(region)
    return pooled

# Aplicar pooling
max_pooled = max_pooling(gray_sharpened)  # Saída 256x256
avg_pooled = avg_pooling(gray_sharpened)  # Saída 256x256

# Mostrar as imagens
cv2.imshow('Max Pooling (256x256)', max_pooled)
cv2.imshow('Avg Pooling (256x256)', avg_pooled)

#cv2.imwrite('a.png', max_pooled)
#cv2.imwrite('b.png', avg_pooled)

cv2.waitKey(0)
cv2.destroyAllWindows()

