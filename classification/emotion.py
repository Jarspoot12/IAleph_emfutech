from tensorflow.keras.models import load_model  # Usamos tf.keras para garantizar compatibilidad
from tensorflow.keras.layers import DepthwiseConv2D as BaseDepthwiseConv2D
from tensorflow.keras.utils import get_custom_objects
import cv2
import numpy as np

# Deshabilitar la notación científica para mayor claridad (opcional)
np.set_printoptions(suppress=True)

# Definimos una clase personalizada que ignora el argumento "groups"
class DepthwiseConv2DCompat(BaseDepthwiseConv2D):
    def __init__(self, **kwargs):
        # Eliminar "groups" de los argumentos si existe
        kwargs.pop("groups", None)
        super().__init__(**kwargs)

# Registrar la clase personalizada con el nombre que espera el modelo (usualmente "DepthwiseConv2D")
get_custom_objects()["DepthwiseConv2D"] = DepthwiseConv2DCompat

# Cargar el modelo usando el diccionario de objetos personalizados
model = load_model("keras_model_emotion.h5", compile=False)

# Cargar las etiquetas (asegúrate que "labels_emotion.txt" tenga una etiqueta por línea)
with open("labels_emotion.txt", "r") as f:
    emotion_labels = [line.strip() for line in f.readlines()]

def reconocer_emocion(face_img):
    """
    Detecta la emoción predominante en la imagen del rostro, usando el modelo Keras.
    
    Parámetros:
      - face_img: Imagen del rostro (numpy array). Se espera que sea la ROI extraída desde el frame.
    
    Retorna:
      - emoción: La emoción detectada (cadena de texto).
      - confidence: La confianza de la predicción (valor entre 0 y 1).
    """
    # Redimensionar la imagen a 224x224 (tamaño de entrada del modelo)
    img = cv2.resize(face_img, (224, 224), interpolation=cv2.INTER_AREA)
    # Convertir la imagen a float32 y cambiar la forma para tener dimensión de batch
    img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
    # Normalizar la imagen (esto depende de cómo se entrenó el modelo; aquí se asume una escala de -1 a 1)
    img = (img / 127.5) - 1
    # Ejecutar la predicción
    prediction = model.predict(img, verbose=0)
    index = int(np.argmax(prediction))
    emotion = emotion_labels[index]
    # confidence = float(prediction[0][index])
    return emotion

if __name__ == "__main__":
    # Bloque de prueba: Captura desde la cámara para ver el resultado en tiempo real.
    camera = cv2.VideoCapture(0)
    while True:
        ret, image = camera.read()
        if not ret:
            break
        # Opcional: redimensionar la imagen para visualización
        image_disp = cv2.resize(image, (224, 224))
        cv2.imshow("Webcam Image", image_disp)
        emotion, conf = reconocer_emocion(image)
        print("Emotion:", emotion, "Confidence:", conf)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    camera.release()
    cv2.destroyAllWindows()
