import os
# Filtra los mensajes INFO y WARNING de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from deepface import DeepFace
import cv2

def clasificar_edad_genero(face_img):
    """
    Clasifica la edad y el género a partir de una imagen que contenga el rostro.
    
    Parámetros:
      - face_img: Imagen del rostro (numpy array) extraída de la región de interés.
    
    Retorna:
      - edad: Edad predicha (o 'Desconocido' si no se detecta).
      - genero: Género predicho (o 'Desconocido' si no se detecta).
    """
    # Analiza la imagen usando DeepFace para extraer edad y género
    result = DeepFace.analyze(face_img, actions=['age', 'gender', 'emotion'], enforce_detection=False)
    edad = result[0]['age']
    genero = result[0]['dominant_gender']
    
    # return edad, genero
    return edad, genero
# Bloque de prueba (se ejecuta solo si se corre este archivo directamente)
if __name__ == "__main__":
    face = cv2.imread('/home/jared/Desktop/IAleph/samples/imagen_prueba.jpg')
    edad, genero = clasificar_edad_genero(face)
    print(f"Edad: {edad}, Género: {genero}")
    # print(clasificar_edad_genero(face))
