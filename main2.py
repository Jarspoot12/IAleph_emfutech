import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2, time, threading, queue, json
from detectors.yolov8m import detectar_personas
from tracking.tracker import actualizar_tracker
from classification.age_gender import clasificar_edad_genero
from classification.emotion import reconocer_emocion
from segmentation.segmentation import segmentar_productos

# Parámetros globales
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
PROCESS_WIDTH = 240
PROCESS_HEIGHT = 180
DETECTION_EVERY_N_FRAME = 4       # Cada 4 frames se actualizan las detecciones (boxes)
CLASSIFICATION_EVERY_N_FRAME = 10  # Cada 10 frames inferencia pesada

# Variables y colas globales
multi_cam_frame_queue = queue.Queue(maxsize=10)
last_registros = {}  # Diccionario indexado por cam_id
last_frame = {}      # Último frame por cam_id
person_cache = {}   # Cache para clasificaciones por ID

frame_lock = threading.Lock()
results_lock = threading.Lock()

running = True  # Flag de control

# --- Hilo de captura para cada cámara
def capture_camera(cam_id, cap):
    frame_count = 0
    while running:
        ret, frame = cap.read()
        if not ret:
            print(f"Cámara {cam_id}: no se capturó frame.")
            continue
        frame_count += 1
        with frame_lock:
            last_frame[cam_id] = frame.copy()
        if frame_count % CLASSIFICATION_EVERY_N_FRAME == 0:
            frame_proc = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
            try:
                multi_cam_frame_queue.put({"cam_id": cam_id, "frame": frame_proc}, timeout=0.05)
            except queue.Full:
                pass
        time.sleep(0.01)

# --- Hilo de inferencia pesada
def heavy_classification_worker():
    global last_registros, person_cache
    MIN_ROI_SIZE = 20
    while running:
        try:
            item = multi_cam_frame_queue.get(timeout=1)
        except queue.Empty:
            continue
        cam_id = item["cam_id"]
        frame = item["frame"]
        detecciones, _ = detectar_personas(frame)
        personas = actualizar_tracker(detecciones, frame)
        resultados = []
        for persona in personas:
            x1, y1, x2, y2 = map(int, persona["bbox"])
            if (x2 - x1) < MIN_ROI_SIZE or (y2 - y1) < MIN_ROI_SIZE:
                continue
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            person_id = persona["id"]
            try:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            except Exception as e:
                roi_rgb = roi
            roi_resized = cv2.resize(roi_rgb, (112, 112))
            if person_id in person_cache:
                cached = person_cache[person_id]
                edad = cached["edad"]
                genero = cached["genero"]
                try:
                    emocion, conf = reconocer_emocion(roi_resized)
                except Exception as e:
                    emocion = "Sin detección"
                person_cache[person_id]["emocion"] = emocion
            else:
                try:
                    edad, genero = clasificar_edad_genero(roi_resized)
                except Exception as e:
                    edad, genero = "Desconocido", "Desconocido"
                try:
                    emocion, conf = reconocer_emocion(roi_resized)
                except Exception as e:
                    emocion = "Sin detección"
                person_cache[person_id] = {"edad": edad, "genero": genero, "emocion": emocion}
            try:
                nuevos_productos = segmentar_productos(roi)
            except Exception as e:
                nuevos_productos = []
            productos = nuevos_productos

            registro = {
                "id": person_id,
                "bbox": persona["bbox"],
                "edad": edad,
                "genero": genero,
                "emocion": emocion,
                "productos": productos,
                "timestamp": time.time()
            }
            resultados.append(registro)
        with results_lock:
            last_registros[item["cam_id"]] = resultados
        multi_cam_frame_queue.task_done()

# --- Hilo principal (visualización)
def main():
    global running, last_frame, last_registros
    # Inicializar cámaras
    cap0 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(2)
    cap0.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    
    # Inicializar diccionarios
    with frame_lock:
        last_frame[0] = None
        last_frame[1] = None
    with results_lock:
        last_registros[0] = []
        last_registros[1] = []
    
    # Lanzar hilos de captura para cada cámara
    thread0 = threading.Thread(target=capture_camera, args=(0, cap0))
    thread1 = threading.Thread(target=capture_camera, args=(1, cap1))
    thread0.start()
    thread1.start()
    
    # Lanzar el hilo de inferencia pesada
    heavy_thread = threading.Thread(target=heavy_classification_worker)
    heavy_thread.start()
    
    try:
        while True:
            with frame_lock:
                frame0 = last_frame.get(0)
                frame1 = last_frame.get(1)
            if frame0 is not None:
                disp0 = frame0.copy()
                with results_lock:
                    regs0 = last_registros.get(0, [])
                scale_x = CAPTURE_WIDTH / PROCESS_WIDTH
                scale_y = CAPTURE_HEIGHT / PROCESS_HEIGHT
                for reg in regs0:
                    x1, y1, x2, y2 = map(int, reg["bbox"])
                    x1 = int(x1 * scale_x)
                    x2 = int(x2 * scale_x)
                    y1 = int(y1 * scale_y)
                    y2 = int(y2 * scale_y)
                    cv2.rectangle(disp0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    pid = reg["id"]
                    if pid in person_cache:
                        info = person_cache[pid]
                        label = f"ID: {pid} {info['genero']}, {info['edad']}, {info['emocion']}"
                    else:
                        label = f"ID: {pid} Cargando..."
                    cv2.putText(disp0, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow("Camera 0", disp0)
            if frame1 is not None:
                disp1 = frame1.copy()
                with results_lock:
                    regs1 = last_registros.get(1, [])
                scale_x = CAPTURE_WIDTH / PROCESS_WIDTH
                scale_y = CAPTURE_HEIGHT / PROCESS_HEIGHT
                for reg in regs1:
                    x1, y1, x2, y2 = map(int, reg["bbox"])
                    x1 = int(x1 * scale_x)
                    x2 = int(x2 * scale_x)
                    y1 = int(y1 * scale_y)
                    y2 = int(y2 * scale_y)
                    cv2.rectangle(disp1, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    pid = reg["id"]
                    if pid in person_cache:
                        info = person_cache[pid]
                        label = f"ID: {pid} {info['genero']}, {info['edad']}, {info['emocion']}"
                    else:
                        label = f"ID: {pid} Cargando..."
                    cv2.putText(disp1, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow("Camera 1", disp1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        running = False  # Señal para detener los hilos
        thread0.join()
        thread1.join()
        heavy_thread.join()
        cap0.release()
        cap1.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Inicializar lock de frames antes de usarlo
    frame_lock = threading.Lock()
    main()
