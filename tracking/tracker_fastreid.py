# tracker.py
from .reid_model_fast import FastReIDModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys
import os

# Obtener la ruta absoluta de la raíz del proyecto.
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construir la ruta absoluta al archivo de configuración de FastReID

config_path = os.path.join(current_dir, "fast-reid", "configs", "Market1501", "bagtricks_R50.yml")
# Construir la ruta absoluta a los pesos; asegúrate que el archivo se encuentre en esa ruta
weights_path = os.path.join(current_dir, "market_bot_R50.pth")



# Inicializar el modelo de reidentificación con FastReID.
# Asegúrate de que la ruta al archivo de configuración y pesos sean correctas.
reid_model = FastReIDModel(
    config_file=config_path,
    model_weights=weights_path 
)

# Diccionario global para almacenar los embeddings de las personas detectadas.
# Este diccionario es compartido entre todas las cámaras.
person_features = {}

def actualizar_tracker(detecciones, frame):
    """
    Actualiza el tracker con las detecciones del frame actual utilizando FastReID.
    Se realiza la reidentificación global entre cámaras comparando los embeddings.
    
    Parámetros:
      - detecciones: Lista de bounding boxes en el formato [x1, y1, x2, y2].
                     (Se asume que la detección es de personas.)
      - frame: Frame actual (de la cámara correspondiente) usado para extraer la ROI y calcular el embedding.
    
    Retorna:
      - personas: Lista de diccionarios con cada persona trackeada, con el ID asignado y su bounding box.
    """
    personas = []
    
    # Umbral de similitud para considerar dos embeddings como iguales.
    SIMILARITY_THRESHOLD = 0.80
    
    for detection in detecciones:
        # Extrae la caja, la confianza y la clase; nos interesa la caja.
        bbox, conf, cls = detection
        
        # Asegúrate de que bbox sea una lista con [x1, y1, x2, y2]
        # En caso de que se encuentre anidada, desanídala:
        if isinstance(bbox[0], list) or isinstance(bbox[0], tuple):
            bbox = bbox[0]
        
        try:
            x1, y1, x2, y2 = map(int, bbox)
        except Exception as e:
            print("Error al convertir la caja a enteros:", e)
            continue

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        
        # Extraer el embedding de la ROI usando FastReID.
        feature = reid_model.extract_features(roi)
        print("Embedding:", feature[:5])

        # Comparar el embedding extraído con los almacenados globalmente.
        matched_id = None
        max_sim = 0.0
        for pid, stored_feature in person_features.items():
            sim = cosine_similarity([feature], [stored_feature])[0][0]
            if sim > SIMILARITY_THRESHOLD and sim > max_sim:
                matched_id = pid
                max_sim = sim
        
        # Si no hay coincidencia (o la similitud es baja), se asigna un nuevo ID.
        if matched_id is None:
            matched_id = len(person_features) + 1
            person_features[matched_id] = feature
        else:
            # Si se coincide, opcionalmente actualizamos el embedding para adaptarnos a ligeros cambios.
            person_features[matched_id] = feature
        
        personas.append({'id': matched_id, 'bbox': bbox})
    
    return personas

# Bloque de prueba (opcional)
if __name__ == "__main__":
    print("Prueba de tracker con FastReID: módulo cargado correctamente.")
