#la etiqueta que sale de esta función es la que se usa para clasificar el género en el modelo de re-identificación
# esta salida debe reemplazar al género que predice age_gender.py, pero mantener la edad que predice

from keras.models import load_model
import numpy as np
import cv2
import sys

model = load_model("classification/clasificador_genero_latino_robust.h5")
CLASS_NAMES = ["Female", "Man"]

def clasificar_genero_latino(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (224, 224)) / 255.0
    input_tensor = np.expand_dims(resized, axis=0)
    pred = model.predict(input_tensor)[0]
    label = CLASS_NAMES[np.argmax(pred)]
    return label

if __name__ == "__main__":
    # Ruta fija a la imagen de prueba
    img_path = "/home/jared/Desktop/IAleph/classification/fairface_database/fairface_latino/Female/379_f.jpg"
    
    # Carga y chequeo
    img = cv2.imread(img_path)
    if img is None:
        print(f"No se pudo leer la imagen: {img_path}")
        sys.exit(1)
    
    # Inferencia
    label = clasificar_genero_latino(img)
    print(f"Predicción de género para '{img_path}': {label}")
