#!/usr/bin/env python3
#
# Código para igualar el contraste y brillo de la foto
# Grupo IP Pilar Sánchez-Testillano
# Yolanda Pérez
# FJR/OCT2023

import cv2
import numpy as np
from PIL import Image, ImageTk
import os

filepath_1 = input("Primera imagen de referencia -ruta de archivo:\n")
filename_1 = os.path.basename(filepath_1)
filepath_2 = input("Segunda imagen de referencia -ruta de archivo:\n")
filename_2 = os.path.basename(filepath_2)
filepath_3 = input("Tercera imagen de referencia -ruta de archivo:\n")
filename_3 = os.path.basename(filepath_3)
ruta_guardar = input("~~Carpeta de guardado : ")


# i = Image.open(filepath_1)
i = Image.open('C:\\Users\\Usuario\\Desktop\\Fotos para contar\\divididas\\ttc\\1\\rep2\\prueba.jpg')
# i2 = Image.open(filepath_2)
i2 = Image.open("C:\\Users\\Usuario\\Desktop\\Fotos para contar\\divididas\\ttc\\1\\rep1\\1TTC 1.2_Region 1 Merged_Crop001.jpgoutput_0_0.jpg")
# i3 = Image.open(filepath_3)
i3 = Image.open("C:\\Users\\Usuario\\Desktop\\Fotos para contar\\divididas\\ttc\\1\\rep3\\11.1_Region 1 Merged_Crop001.jpgoutput_0_0.jpg")
img = np.array(i)
img2 = np.array(i2)
img3 = np.array(i3)
height, width = i.size
height2, width2 = i2.size
height3, width3 = i3.size
img = img / 255
img2 = img2 / 255
img3 = img3 / 255
summation = 0
summation2 = 0
summation3 = 0
count = 0
count2 = 0
count3 = 0

for i in range(height):
    for j in range(width):
        summation = summation + img[i, j]
        count += 1
for x in range(height2):
    for y in range(width2):
        summation2 = summation2 + img2[x, y]
        count2 += 1

for x in range(height3):
    for y in range(width3):
        summation3 = summation3 + img3[x, y]
        count3 += 1

summation = (summation[0] + summation[1] + summation[2])/3
summation2 = (summation2[0] + summation2[1] + summation2[2])/3
summation3 = (summation3[0] + summation3[1] + summation3[2])/3

average = summation / count
average2 = summation2 / count2
average3 = summation3 / count3
new_avg = (average+average2+average3)/3

r = new_avg / average
r2 = new_avg / average2
r3 = new_avg / average3

brightness = r
brightness2 = r2
brightness3 = r3

image_bright = np.ones(i.size, dtype=np.float) * brightness
image_bright2 = np.ones(i2.size, dtype=np.float) * brightness2
image_bright3 = np.ones(i3.size, dtype=np.float) * brightness3

image = img * image_bright
image2 = img2 * image_bright2
image3 = img3 * image_bright3

cv2.imshow('img', image)
cv2.imshow('img2', image2)
cv2.imshow('img3', image3)
cv2.waitKey(0)
cv2.destroyAllWindows()