import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.applications import ResNet50

# Función para procesar una imagen dada su ruta
def procesar_imagen(img_path):
    # Cargar la imagen con el tamaño adecuado para el modelo
    messagebox.showinfo("Cargandoimagen", "Espere mientras se procesa la imagen.")
    img = image.load_img(img_path, target_size=(224, 224))
    # Convertir la imagen en un array
    img_array = image.img_to_array(img)
    # Expandir las dimensiones de la imagen para que se ajuste al formato esperado por el modelo
    img_expanded = np.expand_dims(img_array, axis=0)
    # Preprocesar la imagen de la misma manera que se preprocesaron las imágenes originales
    img_ready = preprocess_input(img_expanded)
    return img_ready

# Función para mostrar la imagen en un tamaño específico
def mostrar_imagen(image_path):
    # Cargar la imagen
    img = Image.open(image_path)
    # Redimensionar la imagen a 100px x 100px
    img.thumbnail((100, 100))
    # Convertir la imagen a un objeto PhotoImage
    photo_img = ImageTk.PhotoImage(img)
    
    # Mostrar la imagen en un widget Label
    img_label.config(image=photo_img)
    # Mantener una referencia a la imagen para evitar que sea eliminada por el recolector de basura
    img_label.image = photo_img

# Función para realizar la predicción y mostrar el resultado
def predecir_imagen():
    global img_ready, preds, prediction_label
    
    # Abrir el cuadro de diálogo para seleccionar la imagen
    file_path = filedialog.askopenfilename()
    
    if file_path:
        # Procesar la imagen
        img_ready = procesar_imagen(file_path)
        
        # Mostrar la imagen en el lienzo
        mostrar_imagen(file_path)
        
        # Instanciar un modelo ResNet50 con pesos 'imagenet'
        model = ResNet50(weights='imagenet')
        
        # Realizar la predicción con ResNet50
        preds = model.predict(img_ready)
        
        # Decodificar la primera predicción
        prediction = decode_predictions(preds, top=1)[0][0]
    
        
        # Habilitar el botón 'Generate Prediction'
        generate_button.config(state=tk.NORMAL, bg="#1E90FF")
        # Deshabilitar el botón 'Cargar Imagen'
        load_button.config(state=tk.DISABLED, bg="#D3D3D3")

# Función para generar la predicción
def generar_prediccion():
    global prediction_label
    
    # Decodificar la primera predicción
    prediction = decode_predictions(preds, top=1)[0][0]
    
    # Actualizar la etiqueta de la predicción
    prediction_label.config(text=f"Objeto: {prediction[1]}\nProbabilidad: {prediction[2]}")

# Función para restaurar el programa a su estado original
def resetear_programa():
    # Habilitar el botón 'Cargar Imagen'
    load_button.config(state=tk.NORMAL, bg="#D3D3D3")
    # Deshabilitar el botón 'Generate Prediction'
    generate_button.config(state=tk.DISABLED, bg="#D3D3D3")
    # Limpiar la etiqueta de predicción
    prediction_label.config(text="")
    # Limpiar la imagen mostrada
    img_label.config(image="")

# Crear la ventana principal de la aplicación
root = tk.Tk()
root.title("Predicción de Imágenes")

# Crear un frame con un fondo homogéneo
canvas = tk.Canvas(root, width=400, height=500, bg="#FFFFFF", highlightthickness=0)
canvas.pack()

# Encabezado
header_label = tk.Label(canvas, text="Reconocedor de Imágenes",bg="#FFFFFF",  font=("Arial", 16))
header_label.place(relx=0.5, rely=0.1, anchor="center")

# Descripción del programa
descripcion_text = ("Este programa utiliza un modelo de red neuronal convolucional (CNN) llamado ResNet50 "
                    "pre-entrenado en el conjunto de datos ImageNet. Permite cargar una imagen y muestra "
                    "el objeto reconocido con su probabilidad asociada.")
descripcion_label = tk.Label(canvas, text=descripcion_text, font=("Arial", 10), justify="left", wraplength=280,bg="#FFFFFF")
descripcion_label.place(relx=0.5, rely=0.3, anchor="center")

# Crear un botón para cargar la imagen
load_button = tk.Button(canvas, text="Cargar Imagen", command=predecir_imagen, bg="#D3D3D3")
load_button.place(relx=0.3, rely=0.7, anchor="center")

# Crear un botón para generar la predicción
generate_button = tk.Button(canvas, text="Generate Prediction", command=generar_prediccion, bg="#D3D3D3", state=tk.DISABLED)
generate_button.place(relx=0.7, rely=0.7, anchor="center")

# Crear un botón para restaurar el programa a su estado original
refresh_button = tk.Button(canvas, text="Refrescar", command=resetear_programa)
refresh_button.place(relx=0.5, rely=0.8, anchor="center")

# Etiqueta para mostrar la imagen
img_label = tk.Label(canvas, bg="#FFFFFF")
img_label.place(relx=0.5, rely=0.5, anchor="center")

# Etiqueta para mostrar la predicción
prediction_label = tk.Label(canvas, text="", font=("Arial", 12), bg="#FFFFFF")
prediction_label.place(relx=0.5, rely=0.9, anchor="center")

# Ejecutar la aplicación
root.mainloop()
