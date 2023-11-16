#!/usr/bin/env python3
#
# Reconocimiento de imágenes.
# Division de imagenes jpg en 4 imagenes y guardado en carpeta
# Grupo IP Pilar Sánchez-Testillano
# Yolanda Pérez
# FJR/OCT2023
# 
# Este codigo fue programado para:
# python 3.6.5
# pillow 5.4.1
# opencv 3.4.2.17
#
# Parámetros particulares
# dp=1.3, param1=50, minRadius=10, maxRadius=30


import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from tkinter import messagebox
import os
Image.MAX_IMAGE_PIXELS = None

# Defining main function 
def dividir(): 
    tile_size = 1024

    file_path = input("Enter filepath from picture to divide in 12:")
    ruta_guardar = input("Enter output dir:")
    filename = os.path.basename(file_path)

    if file_path: #
        # Leer la imagen
        image = Image.open(file_path)

        # Get the dimensions of the image
        width, height = image.size

        # Calculate the width and height of each divided region
        region_width = width // 2
        region_height = height // 6

        cont_foto=0
        # Loop through and save each region as a JPG image
        for i in range(2):
            for j in range(6):
                cont_foto += 1
                left = i * region_width
                upper = j * region_height
                right = left + region_width
                lower = upper + region_height

                # Crop and save the region as a PNG
                region = image.crop((left, upper, right, lower))
                region.save(ruta_guardar+f"\\{filename}output_{cont_foto}.jpg")
    return
  
  
# Using the special variable  
# __name__ 
if __name__=="__main__": 
    dividir() 