import cv2
import numpy as np

# Carregar imagem
image = cv2.imread('Dog.png')

image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)



# Kernel de blur
blur_kernel = np.array(    [[1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1]], np.float32) / 25
blurred = cv2.filter2D(image, -1, blur_kernel)

# Kernel de sharpness
sharp_kernel = np.array(   [[0, -1,  0],
                            [-1, 5, -1],
                            [0, -1,  0]   ])
sharpened = cv2.filter2D(image, -1, sharp_kernel)

# Kernel de laplace
laplace_kernel = np.array( [[ 0,  -1,  0],
                            [-1,   4, -1],
                            [ 0, - 1,  0]   ])
laplace = cv2.filter2D(image, -1, laplace_kernel)

# Kernel Horizontal
hor_kernel = np.array(     [[-1, -2,  -1],
                            [0, 0, 0],
                            [1, 2,  1]   ])
hor = cv2.filter2D(image, -1, hor_kernel)

# Kernel Vertical
ver_kernel = np.array(     [[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]   ])
ver = cv2.filter2D(image, -1, ver_kernel)





#cv2.imwrite('Original.png', image)
#cv2.imwrite('Blurred.png', blurred)
#cv2.imwrite('Sharpened.png', sharpened)
#cv2.imwrite('Laplace.png', laplace)
#cv2.imwrite('Horizontal.png', hor)
cv2.imwrite('Vertical.png', ver)


# Mostrar resultados
cv2.imshow('Original', image)
cv2.imshow('Blurred', blurred)
cv2.imshow('Sharpened', sharpened)
cv2.imshow('Laplaciano', laplace)
cv2.imshow('Horizintal', hor)
cv2.imshow('Vertical', ver)




cv2.waitKey(0)
cv2.destroyAllWindows()