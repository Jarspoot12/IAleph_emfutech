import os
# Filtra los mensajes INFO y WARNING de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from deepface import DeepFace
import cv2

def reconocer_emocion(face_img):
    """
    Reconoce la emoción predominante en una imagen del rostro utilizando DeepFace.
    
    Parámetros:
      - face_img: Imagen del rostro (numpy array).
    
    Retorna:
      - emoción: La emoción detectada (por ejemplo, 'happy', 'sad', etc.) o "Sin detección" si falla.
    """
    try:
        resultado = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
        # Si DeepFace devuelve una lista, tomar el primer elemento
        if isinstance(resultado, list):
            resultado = resultado[0]
        # 'dominant_emotion' contiene la emoción con mayor probabilidad
        return resultado.get("dominant_emotion", "Sin detección")
    except Exception as e:
        print("Error en el análisis:", e)
        return "Sin detección"

if __name__ == "__main__":
    # Carga la imagen desde disco
    face = cv2.imread('/home/jared/Desktop/IAleph/samples/imagen_prueba.jpg')
    if face is not None:
        emocion = reconocer_emocion(face)
        print("Emoción:", emocion)
    else:
        print("No se pudo cargar la imagen.")
