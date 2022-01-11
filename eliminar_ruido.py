import cv2
import numpy as np
import math
img='im5.jpg'
image = cv2.imread(img)
def gasuss_noise(image, mean=0, var=0.001):
    '''
        añadir ruido gaussiano
        image:imagen original
        mean : mean of gaussiana matriz
        var : varianza,que es mayor que tenga mas ruido
    '''
    image = np.array(image/255, dtype=float)#Normalizar la imagen original
    noise = np.random.normal(mean, var ** 0.5, image.shape)#Crear una matriz gaussiana
    noise_img = image + noise#Conseguir la imagen ruidosa
    if noise_img.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    noise_img = np.clip(noise_img, low_clip, 1.0)#restringir el value entre low-clip y 1.0
    noise_img = np.uint8(noise_img*255)#deponer la normalizacion
    #cv.imshow("gasuss", out)
    #noise = noise*255
    return noise_img

def psnr(img1, img2):
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

noise_img = gasuss_noise(image, mean=0, var=0.003)
cv2.imwrite('noise.jpg',noise_img)
before_psnr=psnr(image,noise_img)
#Usando y comparando medio filtro y medium filtro para eliminar ruido
#elegir mejor kernel_size de filtro
size_list=[3,5,7]
blur_psnr=0
medianBlur_psnr=0
blur_size=0
medianBlur_size=0
for kernel_size in size_list:
    img_blur=cv2.blur(noise_img,(kernel_size,kernel_size))
    if psnr(image,img_blur)>blur_psnr:
        blur_psnr=psnr(image,img_blur)
        blur_size=kernel_size
for kernel_size in size_list:
    img_median_blur=cv2.medianBlur(noise_img,kernel_size)
    if psnr(image,img_median_blur)>medianBlur_psnr:
        medianBlur_psnr=psnr(image,img_median_blur)
        medianBlur_size=kernel_size
#Sleccionar lo cual psnr es mas alto
if medianBlur_psnr>blur_psnr:#Usar medianBlur
    denoise_img=cv2.medianBlur(noise_img,medianBlur_size)
    cv2.imwrite('denoise.jpg',denoise_img)
    after_psnr=medianBlur_psnr
    print('Usar medianBlur filtro eliminar ruido,y su kernel size es {},su psnr es {}'.format(medianBlur_size,medianBlur_psnr))
else:
    denoise_img=cv2.blur(noise_img,(blur_size,blur_size))
    cv2.imwrite('denoise.jpg', denoise_img)
    after_psnr = blur_psnr
    print('Usar medianBlur filtro eliminar ruido,y su kernel size es {},su psnr es {}'.format(medianBlur_size,blur_psnr))

#Compara el value antes de denoise para verificar el efecto de denoise
if after_psnr>before_psnr:
    print('denoise con exito,psnr ha aumentado')
else:
    print('psnr no esta mejorado,no denoise con exito')