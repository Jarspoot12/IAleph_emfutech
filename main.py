import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Deshabilita GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import time
import threading
import queue
import json

# Importar las funciones de cada módulo (ajusta las rutas según corresponda)
from detectors.yolo2 import detectar_personas
from tracking.tracker import actualizar_tracker
from classification.age_gender import clasificar_edad_genero
from classification.emotion2 import reconocer_emocion
from segmentation.segmentation2 import segmentar_productos

# Parámetros globales
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
PROCESS_WIDTH = 240    # Resolución baja para procesamiento pesado
PROCESS_HEIGHT = 180
DETECTION_EVERY_N_FRAME = 8    # Actualizar boxes cada 8 frames
CLASSIFICATION_EVERY_N_FRAME = 20  # Ejecutar inferencia pesada cada 20 frames
DISAPPEAR_THRESHOLD = 0.0  # Usaremos detección actual para dibujar boxes

# Cola para enviar frames para inferencia pesada
heavy_frame_queue = queue.Queue(maxsize=5)
# Variable global (protegida por lock) para almacenar los resultados pesados actuales
last_registros = []
lock = threading.Lock()
# Diccionario para cachear la clasificación por ID (edad y género se mantienen; emoción se actualiza)
person_cache = {}
# (Opcional) Lock para cajas si se requiere separar la actualización de "current_boxes"
boxes_lock = threading.Lock()
current_boxes = []  # Para actualización rápida de boxes (detección y tracking) cada DETECTION_EVERY_N_FRAME

def heavy_classification_worker():
    """
    Hilo que procesa frames para inferencia pesada (clasificación, segmentación) cada CLASSIFICATION_EVERY_N_FRAME.
    Actualiza la caché y genera una salida JSON con la información: ID, edad, género, emoción, productos.
    """
    global last_registros, person_cache
    MIN_ROI_SIZE = 20
    while True:
        try:
            frame = heavy_frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        detecciones, _ = detectar_personas(frame)
        personas = actualizar_tracker(detecciones, frame)
        resultados = []

        for persona in personas:
            x1, y1, x2, y2 = map(int, persona['bbox'])
            if (x2 - x1) < MIN_ROI_SIZE or (y2 - y1) < MIN_ROI_SIZE:
                continue
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            person_id = persona['id']

            # Actualizar clasificación:
            try:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print("Error en conversión de color (heavy):", e)
                roi_rgb = roi
            roi_resized = cv2.resize(roi_rgb, (112, 112))
            if person_id in person_cache:
                cached = person_cache[person_id]
                edad = cached['edad']
                genero = cached['genero']
                try:
                    emocion = reconocer_emocion(roi_resized)
                    if not emocion:
                        emocion = "Sin detección"
                except Exception as e:
                    print("Error en reconocimiento de emoción (heavy):", e)
                    emocion = "Sin detección"
                person_cache[person_id]['emocion'] = emocion
            else:
                try:
                    edad, genero = clasificar_edad_genero(roi_resized)
                except Exception as e:
                    print("Error en clasificación de edad/género (heavy):", e)
                    edad, genero = "Desconocido", "Desconocido"
                try:
                    emocion = reconocer_emocion(roi_resized)
                    if not emocion:
                        emocion = "Sin detección"
                except Exception as e:
                    print("Error en reconocimiento de emoción (heavy):", e)
                    emocion = "Sin detección"
                person_cache[person_id] = {'edad': edad, 'genero': genero, 'emocion': emocion}

            # Actualizar productos usando el modelo de Keras (entrenado con Teachable Machine o reentrenado para 5 clases)
            try:
                nuevos_productos = segmentar_productos(roi)
            except Exception as e:
                print("Error en segmentación de productos (heavy):", e)
                nuevos_productos = []
            # Reescribe la lista de productos con la detección actual:
            productos = nuevos_productos

            registro = {
                "id": person_id,
                "bbox": persona['bbox'],  # Coordenadas en el frame reducido
                "edad": edad,
                "genero": genero,
                "emocion": emocion,
                "productos": productos,
                "timestamp": time.time()
            }
            resultados.append(registro)
        with lock:
            last_registros = resultados

        # Genera salida JSON (puedes elegir imprimirla o guardarla en un archivo)
        output_json = json.dumps(
            [{"id": r["id"],
              "edad": r["edad"],
              "genero": r["genero"],
              "emocion": r["emocion"],
              "productos": r["productos"]}
             for r in resultados],
            indent=2
        )
        print("Heavy Frame JSON Output:")
        print(output_json)

        heavy_frame_queue.task_done()

def main():
    global last_registros, current_boxes, person_cache
    cap = cv2.VideoCapture(2)
    # Si usas una cámara secundaria, cambia el índice (ej: 2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    frame_count = 0

    heavy_thread = threading.Thread(target=heavy_classification_worker, daemon=True)
    heavy_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar el frame.")
            break

        frame_count += 1
        display_frame = frame.copy()

        # Actualización de boxes (detección y tracking) cada DETECTION_EVERY_N_FRAME
        if frame_count % DETECTION_EVERY_N_FRAME == 0:
            frame_proc = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
            detecciones, _ = detectar_personas(frame_proc)
            personas = actualizar_tracker(detecciones, frame_proc)
            with boxes_lock:
                current_boxes = personas

        # Envío de frame para inferencia pesada cada CLASSIFICATION_EVERY_N_FRAME
        if frame_count % CLASSIFICATION_EVERY_N_FRAME == 0:
            frame_proc = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
            try:
                heavy_frame_queue.put(frame_proc, timeout=0.05)
            except queue.Full:
                pass

        # Dibujar boxes de current_boxes (detección/tracking) para mayor fluidez
        with boxes_lock:
            boxes_to_draw = current_boxes.copy()
        scale_x = CAPTURE_WIDTH / PROCESS_WIDTH
        scale_y = CAPTURE_HEIGHT / PROCESS_HEIGHT
        for persona in boxes_to_draw:
            x1, y1, x2, y2 = map(int, persona["bbox"])
            x1 = int(x1 * scale_x)
            x2 = int(x2 * scale_x)
            y1 = int(y1 * scale_y)
            y2 = int(y2 * scale_y)
            # Dibujar solo si el box es válido (no dummy)
            if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                continue
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            pid = persona["id"]
            if pid in person_cache:
                info = person_cache[pid]
                etiqueta = f"ID: {pid} {info['genero']}, {info['edad']}, {info['emocion']}"
            else:
                etiqueta = f"ID: {pid} Cargando..."
            cv2.putText(display_frame, etiqueta, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Predicciones en Tiempo Real", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Se define boxes_lock para proteger current_boxes
    boxes_lock = threading.Lock()
    main()
