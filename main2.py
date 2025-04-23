# main.py
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Forzar CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Obtener la ruta absoluta de la raíz del proyecto.
current_dir = os.path.dirname(os.path.abspath(__file__))
# Agregar la ruta del repositorio fast-reid (se encuentra en "tracking/fast-reid")
fastreid_path = os.path.join(current_dir, "tracking", "fast-reid")
sys.path.append(fastreid_path)

import cv2
import time
import threading
import queue
import json

# Importar módulos (ajusta rutas según tu estructura)
from detectors.yolov8m import detectar_personas
# from tracking.tracker_fastreid import actualizar_tracker  # Tracker modificado que utiliza FastReID para asignar IDs globales.
from tracking.tracker_deepface import actualizar_tracker
from classification.age_gender import clasificar_edad_genero
from classification.emotion import reconocer_emocion
from segmentation.segmentation import segmentar_productos

# Parámetros de cámara y resolución
CAPTURE_WIDTH = 640   
CAPTURE_HEIGHT = 480
PROCESS_WIDTH = 640    
PROCESS_HEIGHT = 480
DETECTION_EVERY_N_FRAME = 10       
CLASSIFICATION_EVERY_N_FRAME = 40  # Frecuencia para enviar a heavy classification

# Lista de cámaras (índices)
CAMERAS = [0, 2]

# Estructuras globales para manejo de múltiples cámaras:
# Cola para inferencia pesada; ahora enviamos (cam_id, personas, frame_proc)
heavy_frame_queue = queue.Queue(maxsize=10)
# Diccionarios para almacenar resultados (por cada cámara)
last_registros_by_camera = {cam_id: [] for cam_id in CAMERAS}
lock = threading.Lock()  # Para sincronizar actualizaciones en la caché
# person_cache: almacena la clasificación pesada para cada ID por cámara
person_cache = {cam_id: {} for cam_id in CAMERAS}
boxes_lock = threading.Lock()
# current_boxes almacena las cajas (detecciones + tracking) actuales en cada cámara
current_boxes = {cam_id: [] for cam_id in CAMERAS}
frame_counts = {cam_id: 0 for cam_id in CAMERAS}

def heavy_classification_worker():
    """
    Hilo que procesa inferencia pesada (edad, género, emoción y segmentación) 
    para las detecciones ya trackeadas.
    Recibe una tupla: (cam_id, personas, frame_proc).
    """
    MIN_ROI_SIZE = 20
    while True:
        try:
            cam_id, personas, frame = heavy_frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        for persona in personas:
            bbox = persona.get("bbox")
            # Si la caja está anidada, extraer el primer elemento.
            if isinstance(bbox[0], (list, tuple)):
                bbox = bbox[0]
            try:
                x1, y1, x2, y2 = map(int, bbox)
            except Exception as e:
                print("Error al convertir bbox a enteros:", e)
                continue
            if (x2 - x1) < MIN_ROI_SIZE or (y2 - y1) < MIN_ROI_SIZE:
                continue
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                continue

            person_id = persona["id"]
            try:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print("Error en conversión de color (heavy):", e)
                roi_rgb = roi
            roi_resized = cv2.resize(roi_rgb, (112, 112))
            
            # Ejecutar inferencia pesada: edad, género y emoción.
            try:
                edad, genero = clasificar_edad_genero(roi_resized)
                # Puedes agregar un print debug si lo deseas:
                # print(f"ID {person_id} - Edad: {edad}, Género: {genero}")
            except Exception as e:
                print("Error en clasificación de edad/género:", e)
                edad, genero = "Desconocido", "Desconocido"
            try:
                emocion = reconocer_emocion(roi_resized)
                if not emocion:
                    emocion = "Sin detección"
                # print(f"ID {person_id} - Emoción: {emocion}")
            except Exception as e:
                print("Error en reconocimiento de emoción:", e)
                emocion = "Sin detección"
            
            # Actualizar la caché para este ID en la cámara correspondiente:
            with lock:
                person_cache[cam_id][person_id] = {
                    "edad": edad,
                    "genero": genero,
                    "emocion": emocion
                }
        heavy_frame_queue.task_done()

def main():
    # Abrir objetos de captura para cada cámara.
    caps = {cam_id: cv2.VideoCapture(cam_id) for cam_id in CAMERAS}
    for cam_id, cap in caps.items():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

    for cam_id in CAMERAS:
        cv2.namedWindow(f"Predicciones en Tiempo Real - Cámara {cam_id}", cv2.WINDOW_NORMAL)

    # Iniciar el hilo de inferencia pesada.
    heavy_thread = threading.Thread(target=heavy_classification_worker, daemon=True)
    heavy_thread.start()

    while True:
        frames = {}
        # Capturar frames de cada cámara.
        for cam_id, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                print(f"No se pudo capturar el frame de la cámara {cam_id}.")
                continue
            frame = cv2.flip(frame, 1)  # Efecto espejo
            frames[cam_id] = frame
            frame_counts[cam_id] += 1

        for cam_id, frame in frames.items():
            display_frame = frame.copy()
            # Cada DETECTION_EVERY_N_FRAME se actualizan las detecciones y se asignan IDs (tracking ligero)
            if frame_counts[cam_id] % DETECTION_EVERY_N_FRAME == 0:
                frame_proc = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
                detecciones = detectar_personas(frame_proc)
                personas = actualizar_tracker(detecciones, frame_proc, cam_id)
                with boxes_lock:
                    current_boxes[cam_id] = personas

                # Cada CLASSIFICATION_EVERY_N_FRAME se envía a heavy classification el par (cam_id, personas, frame_proc)
                if frame_counts[cam_id] % CLASSIFICATION_EVERY_N_FRAME == 0:
                    try:
                        heavy_frame_queue.put((cam_id, personas, frame_proc), timeout=0.05)
                    except queue.Full:
                        pass

            # Dibujo de resultados
            with boxes_lock:
                boxes_to_draw = current_boxes[cam_id].copy()
            scale_x = CAPTURE_WIDTH / PROCESS_WIDTH
            scale_y = CAPTURE_HEIGHT / PROCESS_HEIGHT
            for persona in boxes_to_draw:
                bbox = persona["bbox"]
                if isinstance(bbox[0], (list, tuple)):
                    bbox = bbox[0]
                try:
                    x1, y1, x2, y2 = map(int, bbox)
                except Exception as e:
                    print("Error al convertir bbox en el dibujo:", e)
                    continue
                # Escalar las coordenadas al tamaño de la captura original
                x1 = int(x1 * scale_x)
                x2 = int(x2 * scale_x)
                y1 = int(y1 * scale_y)
                y2 = int(y2 * scale_y)
                if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                    continue
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                pid = persona["id"]
                with lock:
                    info = person_cache[cam_id].get(pid, None)
                if info:
                    etiqueta = f"ID: {pid} {info['genero']}, {info['edad']}, {info['emocion']}"
                else:
                    etiqueta = f"ID: {pid} Cargando..."
                cv2.putText(display_frame, etiqueta, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow(f"Predicciones en Tiempo Real - Cámara {cam_id}", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cam_id, cap in caps.items():
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
