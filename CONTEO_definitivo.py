#!/usr/bin/env python3
#
# Reconocimiento de imágenes.
# Detección de círculos y recuento en grupos según color.
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
# dp=1.3, param1=50, param2=15, minRadius=5, maxRadius=12 (calibrado con recono3)

# solo funciona con jpg


# image processing, numeric calc
from math import nextafter
import cv2
import numpy as np
from PIL import Image, ImageTk
# gui
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
# directory checks, access
import os
# data wrangling
import pandas as pd
# plot final detected and classified colors 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# procesado y clasificacion de colores medio
from scipy.cluster.vq import whiten
from sklearn.cluster import KMeans

from skimage.draw import disk

# guardar y cargar modelos
import pickle

# remove size limitation
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))

# mejoras
# incorporar varios kmeans (1 por replica) - agregacion
# por directorio
# por fichero
# incorporar informacion de directorio 



def calculate_average_color(roi):
    # Calcula el color promedio de la región ROI
    average_color = np.mean(roi, axis=(0, 1)).astype(int)
    return tuple(average_color)



def entrenar_kmeans():
    
    # Crea carpeta de resultados general
    # ruta_guardar = input("Escribe el nombre del directorio \n-----------------------")
    ruta_guardar = "C:\\Users\\Usuario\\Desktop\\Fotos para contar\\output_segundo_modelo"

    try:
        os.mkdir(ruta_guardar) #
    except:
        1

    # pregunta por directorio y saca imagenes
    dir_path = filedialog.askdirectory(title="Directorio con fotos para entrenar KMEANS")


    # coge solo jpg o tif
    onlyfiles = [f for f in os.listdir(dir_path) if os.path.isfile("\\".join([dir_path, f])) and ".jpg" in f or ".tif" in f and "Merged" not in f ]

    
    tag_circle = 0
    num_circles = 0

     # df_color_circles : guarda infromacion sobre los valores rgb de todos los pixeles separados por cada circulo, se preprocesa para hacer clustering no supervisado
    # df_info_circles : guarda informacion sobre las coordenadas de los circulos y en qué imagen aparecen, se usará para anotar las imágenes
       
    df_color_circles = pd.DataFrame(columns=["r", "g", "b", "Ctag"] )
    df_info_circles = pd.DataFrame(columns=["Ctag", "x", "y", "r", "filename"])

    #####################################################################################################################
    # PRIMERA PARTE : ANALISIS DE  IMAGEN
    # PRIMERA ITERACION POR FICHEROS - DETECTA CIRCULOS, GUARDA INFORMACIÓN ACERCA DE METRICAS
    # MÁS ADELANTE SE INCORPORARÁ INFORMACIÓN ACERCA DEL - COLOR DOMINANTE DE LOS 3 CLUSTERS, Nº DE CIRCULOS EN CADA CLUSTER

    # MEDIA RGB IMAGEN - VALOR APROXIMADO DEL BACKGROUND
    # RADIO DE CIRCULOS SIN THRESHOLD INFERIOR
    # RADIO DE CIRCULOS CRIBADOS
    # NUMERO DE CIRCULOS CRIBADO

    for filename in onlyfiles:

        radios_circulo_np = []
        radios_circulo_p = []
        

        if filename:
            

            # Leer la imagen
            image = cv2.imread(dir_path+"\\"+filename, 1) #

            print(f"Performing processing on file : {filename}\n")

            try:
                # Convertir a escala de grises
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #
            
                
                # ECUALIZACION
                #ecualizada = cv2.equalizeHist(gray) #
                
                # Aplicar un suavizado para reducir el ruido
                #5, 5
                blurred = cv2.GaussianBlur(gray, (9, 9), 0) #
                
                # Detección de bordes utilizando el detector de Canny
                #umin 50
                edges = cv2.Canny(blurred, 60, 100)
                
                # Detección de círculos utilizando HoughCircles
                #dp=1.3, minDist=60, param1=55, param2=20, minRadius=30, maxRadius=60
                circles = cv2.HoughCircles(
                    edges, cv2.HOUGH_GRADIENT,  dp=1.4, minDist=40, param1=55, param2=30, minRadius=30, maxRadius=40
                )
                
                
                # iterando por cada circulo detectado
                if circles is not None:
                    circles = np.uint16(np.around(circles))

                    for circle in circles[0, :]:
                        x, y, r = circle
                        # guardar informacion
                        radios_circulo_np.append(r)

                        if r < 40:  # Solo destacar círculos menores de 40 píxeles
                            
                            tag_circle += 1
                            # guardar informacion
                            x, y, r = circle

                            radios_circulo_p.append(r)
                            # nº de circulos detectados
                            num_circles += 1

                            # Extraer el color promedio de la región del círculo
                            try:
                                roi = image[y - r:y + r, x - r:x + r]
                                # only circle:
                                #
                                # mask = np.zeros(image.shape, dtype=np.uint8)
                                # mask = cv2.circle(image, (x,y), r, (255,255,255), -1)
                                # roi = cv2.bitwise_and(image, mask)

                            except:
                                1
                            
                            b, g, red    = roi[:, :, 0].flatten(), roi[:, :, 1].flatten(), roi[:, :, 2].flatten() # For RGB image

                            # anexar a df 
                            df_info_circles = pd.concat([df_info_circles, pd.DataFrame([[tag_circle, x, y, r, filename]], columns =["Ctag", "x", "y", "r", "filename"])], ignore_index = True, axis = 0 )
                            dict_tmp = {"r" : red, "g": g, "b" : b, "Ctag" : [tag_circle]*len(red) }
                            df_color_circles = pd.concat([df_color_circles, pd.DataFrame(data=dict_tmp)], ignore_index = True, axis = 0  )
                
                Med_radios_circulo_np = sum(radios_circulo_np) / len (radios_circulo_np)
                Med_radios_circulo_p = sum(radios_circulo_p) / len (radios_circulo_p)

                output_estadisticas_foto  = [filename, Med_radios_circulo_np, Med_radios_circulo_p, RGB_fondo, num_circles ]
                df_info_foto = pd.concat([df_info_foto, pd.DataFrame([output_estadisticas_foto], columns=columnas_excel)])
                num_circles = 0
            except:
                pass
    
    ###################################################################################
    # SEGUNDA PARTE - ESTANDARIZACION DE VALORES RGB PARA LOS CIRCULOS Y CLUSTERIZACIÓN
    
    # 1) Estandarizacion con https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.whiten.html
    # Asume misma desviacion por canal de colores en los circulos - decisión logica si proceden crops de misma foto

    print("Scaling color channels in cell rois...\n")
    
    df_processed = whiten(np.array(df_color_circles[['r', "g", "b"]].values.tolist()))
    df_color_circles['r_scaled'] = df_processed[:, 0]
    df_color_circles['g_scaled'] = df_processed[:, 1]
    df_color_circles['b_scaled'] = df_processed[:, 2]

    # sacar promedio rgb roi para que todas las observaciones tengan la misma dimension
    df_to_cluster = pd.DataFrame(columns=["r_scaled", "g_scaled", "b_scaled", "Ctag"])
    
    print("Getting mean rgb values per circle...\n")

    df_to_cluster = df_color_circles.groupby(["Ctag"], sort=False, as_index=False).agg({"r_scaled":"mean","g_scaled":"mean", "b_scaled":"mean"})

    ##########################################################################
    # TERCERA PARTE : ENTRENAR y guardar modelo de  KMEANS
    # KMEANS

    print("Performing kmeans...\n")

    km = KMeans( n_clusters=3 ) # 3 clasificaciones
    km.fit(df_to_cluster[["r_scaled", "g_scaled", "b_scaled"]])
    df_to_cluster['label'] = km.predict(df_to_cluster[["r_scaled", "g_scaled", "b_scaled"]])
    
    
    # save model
    t_name = input("Escribe el nombre de la tinción")
    pickle.dump(km, open(f"{ruta_guardar}\\kmeans_{t_name}.pkl", 'wb')) #Saving the model
    
    # Muestra un mensaje de confirmación
    tk.messagebox.showinfo("Modelo de KMEANS guardado", f"La imagen y estadisticas de la imagen se ha guardado en el directorio {ruta_guardar}")
    
    
    # print("Getting cluster centers...\n")
    
    # analisis 

    # Plotear Referencia de color de cada clase
    # Volver a multiplicar por std para sacar valor RGB
    # colors = []
    # r_std, g_std, b_std = df_to_cluster[["r_scaled", "g_scaled", "b_scaled"]].std() 
    
    # for cluster_center in km.cluster_centers_.tolist():
    #     scaled_r, scaled_g, scaled_b = cluster_center
    #     colors.append((
    #     round(scaled_r / r_std * 255),
    #     round(scaled_g / g_std * 255),
    #     round(scaled_b / b_std * 255)
    #     ))
    # print("Cluster centers \n------\n", str("-".join[colors[0]]), str("-".join[colors[1]]), str("-".join[colors[2]]))

    del df_to_cluster
    del df_color_circles
    return
    
    
    
    
    
def predecir_cl():

    # 3 dataframes
    columnas_excel = ["experiment","replica", "filename", "nº circulos"]
        
    # df_info_foto : guarda informacion de cada imagen, contaje de circulos y sus radios detectados con hough transform
    # df_color_circles : guarda infromacion sobre los valores rgb de todos los pixeles separados por cada circulo, se preprocesa para hacer clustering no supervisado
    # df_info_circles : guarda informacion sobre las coordenadas de los circulos y en qué imagen aparecen, se usará para anotar las imágenes
    df_info_foto = pd.DataFrame(columns = columnas_excel )         
    df_color_circles = pd.DataFrame(columns=["r", "g", "b", "Ctag"] )
    df_info_circles = pd.DataFrame(columns=["Ctag", "x", "y", "r", "filename"])

    # Crea carpeta de resultados general
    # ruta_guardar = input("Escribe el nombre del directorio para guardar las fotos \n-----------------------")
    ruta_guardar = "C:\\Users\\Usuario\\Desktop\\Fotos para contar\\output_segundo_modelo"
    try:
        os.mkdir(ruta_guardar) #
    except:
            1

    # load kmeans
    modelfile = filedialog.askopenfilename(filetypes =[('Pickle Files', '*.pkl')])
    print(modelfile)
    if modelfile is not None:
        # load the model from disk
        loaded_model = pickle.load(open(modelfile, 'rb'))



    # pregunta por directorio y saca imagenes
    # dir_path = filedialog.askdirectory(title="Arbol de directorios con fotos para contaje")
    
    #dir_path = "C:\\Users\\Usuario\\Desktop\\Fotos para contar\\divididas\\ttc_12\\analisis_ttc"
    dir_path = "C:\\Users\\Usuario\\Desktop\\Fotos para contar\\divididas\\evans_12\\analisis_evans"
    df_todas_fotos = pd.DataFrame(columns=["experiment", "replica", "filename", "nº circulos", "nº circulos c1", "nº circulos c2", "nº circulos c3"])
    f_i = 0 # file index


    for subdirpath, subdirs, files in os.walk(dir_path):               

        tag_circle = 0
        num_circles = 0
        
        if len(files) != 0:

            
            experiment = os.path.basename(os.path.dirname(subdirpath))
            repname = os.path.basename(subdirpath)
            

            try:
                os.mkdir(ruta_guardar+f"\\{experiment}\\{repname}") #
            except:
                1

            
                #####################################################################################################################
                # PRIMERA PARTE : ANALISIS DE  IMAGEN
                # PRIMERA ITERACION POR FICHEROS - DETECTA CIRCULOS, GUARDA INFORMACIÓN ACERCA DE METRICAS
                # MÁS ADELANTE SE INCORPORARÁ INFORMACIÓN ACERCA DEL - COLOR DOMINANTE DE LOS 3 CLUSTERS, Nº DE CIRCULOS EN CADA CLUSTER

                # MEDIA RGB IMAGEN - VALOR APROXIMADO DEL BACKGROUND
                # RADIO DE CIRCULOS SIN THRESHOLD INFERIOR
                # RADIO DE CIRCULOS CRIBADOS
                # NUMERO DE CIRCULOS CRIBADO


            for filename in files:
                print(f"{dir_path}\\{experiment}\\{repname}\\{filename}")
                radios_circulo_np = []
                radios_circulo_p = []
                f_i += 1
                

                if filename:
                    

                    # Leer la imagen
                    image = cv2.imread(f"{dir_path}\\{experiment}\\{repname}\\{filename}", 1) #
                    

                    # Convertir a escala de grises
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #
                
                    
                    # ECUALIZACION
                    #ecualizada = cv2.equalizeHist(gray) #
                    
                    # Aplicar un suavizado para reducir el ruido
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0) #
                    
                    # Detección de bordes utilizando el detector de Canny
                    edges = cv2.Canny(blurred, 50, 100)
                    
                    # Detección de círculos utilizando HoughCircles
                    #dp=1.3, minDist=60, param1=55, param2=20, minRadius=30, maxRadius=60
                    circles = cv2.HoughCircles(
                        edges, cv2.HOUGH_GRADIENT,  dp=1.4, minDist=40, param1=55, param2=30, minRadius=30, maxRadius=40
                    )
                    
                    
                    print("Procesando imagen : "+filename+"\n")
                
                    # iterando por cada circulo detectado
                    if circles is not None:
                        circles = np.uint16(np.around(circles))

                        for circle in circles[0, :]:
                            x, y, r = circle
                            # guardar informacion
                            radios_circulo_np.append(r)

                            if r < 40:  # Solo destacar círculos menores de 40 píxeles
                                
                                tag_circle += 1
                                ctag_i = "-".join([str(tag_circle), str(f_i), repname, experiment])
                                # guardar informacion
                                x, y, r = circle

                                radios_circulo_p.append(r)
                                # nº de circulos detectados
                                num_circles += 1

                                # Extraer el color promedio de la región del círculo
                                try:
                                    roi = image[y - r:y + r, x - r:x + r]
                                    # only circle:
                                    #
                                    # mask = np.zeros(image.shape, dtype=np.uint8)
                                    # mask = cv2.circle(image, (x,y), r, (255,255,255), -1)
                                    # roi = cv2.bitwise_and(image, mask)

                                except:
                                    1
                                #color = calculate_average_color(roi)

                                b, g, red    = roi[:, :, 0].flatten(), roi[:, :, 1].flatten(), roi[:, :, 2].flatten() # For RGB image

                                # anexar a df 
                                df_info_circles = pd.concat([df_info_circles, pd.DataFrame([[ctag_i, x, y, r, filename]], columns =["Ctag", "x", "y", "r", "filename"])], ignore_index = True, axis = 0 )
                                dict_tmp = {"r" : red, "g": g, "b" : b, "Ctag" : [ctag_i]*len(red) }
                                df_color_circles = pd.concat([df_color_circles, pd.DataFrame(data=dict_tmp)], ignore_index = True, axis = 0  )
                    
                    
                    #Med_radios_circulo_p = sum(radios_circulo_p) / len (radios_circulo_p)

                    output_estadisticas_foto  = [experiment, repname, filename, num_circles ]
                    df_info_foto = pd.concat([df_info_foto, pd.DataFrame([output_estadisticas_foto], columns=columnas_excel)])
                    num_circles = 0

            
                    ###################################################################################
                    # SEGUNDA PARTE - ESTANDARIZACION DE VALORES RGB PARA LOS CIRCULOS Y CLUSTERIZACIÓN
                    
                    # 1) Estandarizacion con https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.whiten.html
                    # Asume misma desviacion por canal de colores en los circulos - decisión logica si proceden crops de misma foto

                    print("Scaling color channels in cell rois...\n")
                    
                    df_processed = whiten(np.array(df_color_circles[['r', "g", "b"]].values.tolist()))
                    print(df_processed)
                    df_color_circles['r_scaled'] = df_processed[:, 0]
                    df_color_circles['g_scaled'] = df_processed[:, 1]
                    df_color_circles['b_scaled'] = df_processed[:, 2]

                    
                    # sacar promedio rgb roi para que todas las observaciones tengan la misma dimension
                    df_to_cluster = pd.DataFrame(columns=["r_scaled", "g_scaled", "b_scaled", "Ctag"])
                    
                    print("Getting mean rgb values per circle...\n")

                    df_to_cluster = df_color_circles.groupby(["Ctag"], sort=False, as_index=False).agg({"r_scaled":"mean","g_scaled":"mean", "b_scaled":"mean"})
                    df_color_circles = pd.DataFrame(columns=["r", "g", "b", "Ctag"] )
                    
                    # # KMEANS PREDICT WITH LOADED MODEL

                    print("Performing kmeans...\n")
                    df_to_cluster['label'] = loaded_model.predict(df_to_cluster[["r_scaled", "g_scaled", "b_scaled"]])
                    

                    # ## inner join coordenadas radio y clasificacion
                    # # La variable df_annotated tendrá información acerca de en qué imagen se situan los circulos, sus coordenadas y su clasificacion
                    df_labelled = df_to_cluster[["Ctag", "label"]]
                    df_annotated = df_info_circles.merge(df_labelled, on="Ctag", how="inner")
                    df_annotated = df_annotated.drop_duplicates()
                    
                    # # Anotar frecuencia de cada label en cada foto
                    df_freq = df_annotated.groupby(by=["filename", "label"], as_index=False).size()
                    print(df_freq.head)

                    # # guarda los conteos en listas para anexarlos al dataframe en formato de columna. (PIVOTAR DATAFRAME en columna labels - valores counts)
                    # # indices de columna de las predicciones y los conteos de cada prediccion en iloclabel y ilocount
                    iloclabel = 1
                    ilocount = 2

                    n_circ_c1 = []
                    n_circ_c2 = []
                    n_circ_c3 = []
                    

                    # # add missing label counts - count 0
                    for filequarter in df_freq["filename"].unique().tolist():
                        for cluster in [0,1,2]:
                            if cluster not in df_freq[df_freq["filename"] == filequarter]["label"].values.tolist():
                                dict_fill = {"filename":[filequarter] , "label":[cluster], "size":[0]}
                                df_freq = pd.concat((df_freq, pd.DataFrame(data=dict_fill)), axis=0, ignore_index=True)
                    
                    # reformat size value to different categorical columns
                    for i in range(df_freq.shape[0]):
                        if df_freq.iloc[i, iloclabel] == 0:
                            n_circ_c1.append(df_freq.iloc[i, ilocount])

                        if df_freq.iloc[i, iloclabel] == 1:
                            n_circ_c2.append(df_freq.iloc[i, ilocount])
                        
                        if df_freq.iloc[i, iloclabel] == 2:
                            n_circ_c3.append(df_freq.iloc[i, ilocount])

                        

                    df_freq = df_freq[["filename"]].drop_duplicates()
                    df_freq["nº circulos c1"] = n_circ_c1
                    df_freq["nº circulos c2"] = n_circ_c2
                    df_freq["nº circulos c3"] = n_circ_c3
                    
                    
                    #
                    # guardar información en dataframe
                    df_final = df_info_foto.merge(df_freq, on = "filename" , how = "inner")
                    #df_final["Viabilidad P+I"] = df_final[""]
                    #df_final["Viabilidad I+N"] = df_final[""]
                    df_todas_fotos  = pd.concat((df_todas_fotos, df_final), ignore_index=True)
                    
                ###################################################################################
                # TERCERA PARTE - ANOTACION DE IMAGENES EN RUTA DE GUARDADO con el dataframe df_annotated

                print(df_freq[["filename"]].head)

                
                print("Annotating filename "+filename)

                # Leer la imagen
                image = cv2.imread(f"{dir_path}\\{experiment}\\{repname}\\{filename}", 1) #
                df_tmp = df_annotated[df_annotated["filename"] == filename][["x", "y", "r", "label"]]
                
                # iterar por circulo anotado
                for row_info_circle in df_tmp.values.tolist():

                    x, y, r, label_col  = row_info_circle 

                    # 1) ANOTAR CIRCULOS
                    # más adelante pongo el color del v de cada centroide
                    if label_col == 0:
                        cv2.circle(image, (x, y), r, (0, 0, 255), 2)  # Dibujar círculos rojos en la imagen
                    elif label_col == 1:
                        cv2.circle(image, (x, y), r, (0, 255, 0), 2)  # Dibujar círculos verdes en la imagen
                    elif label_col == 2:
                        cv2.circle(image, (x, y), r, (0, 255, 255), 2)  # Dibujar círculos amarillos en la imagen
            
                contador_c1 = df_freq[df_freq["filename"] == filename][["nº circulos c1"]].to_string()
                contador_c2 = df_freq[df_freq["filename"] == filename][["nº circulos c2"]].to_string()
                contador_c3 = df_freq[df_freq["filename"] == filename][["nº circulos c3"]].to_string()
                
                # 2) ANOTAR FRECUENCIA DE CLUSTERS
                cv2.putText(image, "Cluster 1 - R : "+str(contador_c1), (20, 50) , cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(image, "Cluster 2 - V : "+str(contador_c2), (20, 90) , cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(image, "Cluster 3 - A : "+str(contador_c3), (20, 130) , cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # 3) GUARDAR IMAGEN
                # Guardar la imagen en formato JPG
                cv2.imwrite(f"{ruta_guardar}\\{filename}_annot.jpg", image)

                
        else:
            next

    df_todas_fotos.to_excel(ruta_guardar+"\\resultados_colores_dominantes_clasificaciones.xlsx", index=False)
    # # Muestra un mensaje de confirmación
    # tk.messagebox.showinfo("Imagenes guardadas", f"La imagen y estadisticas de la imagen se ha guardado en el directorio {ruta_guardar}")
    return


# Crear la ventana principal
ventana = tk.Tk()
ventana.geometry("400x80")
ventana.title("ALGM para CIBMS-CSIC")

# Crear un menú
menu = tk.Menu(ventana)
ventana.config(menu=menu)

# Menú Archivo
menu_archivo = tk.Menu(menu, tearoff=0)
menu.add_cascade(label="Archivo", menu=menu_archivo)
menu_archivo.add_command(label="Entrenar modelo kmeans", command=entrenar_kmeans)
menu_archivo.add_command(label="Predecir viabilidad con modelo kmeans", command=predecir_cl)
menu_archivo.add_separator()
menu_archivo.add_command(label="Salir", command=ventana.quit)

# Crear una etiqueta para mostrar la imagen
#etiqueta_imagen = tk.Label(ventana)
#etiqueta_imagen.pack()

ventana.mainloop()










