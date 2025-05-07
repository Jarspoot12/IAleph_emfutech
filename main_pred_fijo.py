import os, sys, cv2, threading, queue, json
from pymongo import MongoClient
from datetime import datetime, timezone

# ───────── Configuración de entorno ─────────────────────────────────#─
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "tracking", "fast-reid"))

from detectors.yolov8m          import detectar_personas
from tracking.tracker_deepface  import actualizar_tracker, enqueue_embed_job
from classification.age_gender  import clasificar_edad_genero
from classification.custom_gender_classifier import clasificar_genero_latino
from classification.emotion     import reconocer_emocion
from segmentation.segmentation  import segmentar_productos

# ───────── Parámetros globales ───────────────────────────────────────
CAPTURE_W, CAPTURE_H   = 640, 480
PROC_W,    PROC_H      = 640, 480
DETECT_EVERY_N_FRAME   = 3       # YOLO cada 3 frames
CLASSIFY_EVERY_N_FRAME = 30      # clasificación y segmentación cada 30 frames
PERSISTENCE_FRAMES     = 30      # persistencia de bbox sin detección
CAMERAS                = [2, 4]

# ───────── Estructuras compartidas ───────────────────────────────────
ageemo_queue   = queue.Queue()       # ilimitada
person_cache   = {}                  # {cam_id: {gid: info_dict}}
frame_counter  = {}
last_personas  = {}
last_det_frame = {}
lock_cache     = threading.Lock()

# ───────── Hilo de clasificación de atributos y segmentación ────────
def ageemo_worker():
    # Conexión a MongoDB
    connection_string = "mongodb+srv://ialeph:ialeph11@ialeph.yy6gxxd.mongodb.net/?retryWrites=true&w=majority&appName=IAleph"
    client = MongoClient(connection_string)
    ialeph_db = client["IAleph"]
    collection = ialeph_db["ialeph_main"]

    while True:
        cam_id, persona, frame = ageemo_queue.get()
        registros = []

        # Extraer bbox
        x1,y1,x2,y2 = map(int, persona["bbox"])
        h, w = frame.shape[:2]
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
        if x2<=x1 or y2<=y1:
            ageemo_queue.task_done()
            continue

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            ageemo_queue.task_done()
            continue

        # Preparar para edad y emoción
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_in  = cv2.resize(roi_rgb, (112,112))

        # 1) Consultar cache previa
        with lock_cache:
            cam_cache = person_cache.setdefault(cam_id, {})
            info = cam_cache.get(persona["id"])

        # 2) Si no existe, calculamos edad, género y emoción
        if info is None:
            # Edad
            try:
                edad, _ = clasificar_edad_genero(roi_in)
            except:
                edad = "-"
            # Género
            genero = clasificar_genero_latino(roi)
            # Emoción
            try:
                emocion = reconocer_emocion(roi_in) or "-"
            except:
                emocion = "-"
            # Crear el dict inicial
            info = {"edad": edad, "genero": genero, "emocion": emocion}
        else:
            # Reutilizar valores ya almacenados
            edad    = info["edad"]
            genero  = info["genero"]
            emocion = info["emocion"]

        # 3) Siempre recalculamos productos
        productos = segmentar_productos(roi_rgb, conf=0.65)

        # 4) Formar registro JSON usando valores fijos + nuevos productos
        ts = datetime.now(timezone.utc).isoformat()
        registro = {
            "camara":    cam_id,
            "id":        persona["id"],
            "bbox":      persona["bbox"],
            "edad":      edad,
            "genero":    genero,
            "emocion":   emocion,
            "productos": productos,
            "timestamp": ts
        }
        registros.append(registro)
        print(json.dumps(registro))

        # 5) Insertar en MongoDB
        if registros:
            try:
                collection.insert_many(registros)
            except Exception as e:
                print("Error insertando batch en Mongo:", e)

        # 6) Actualizar cache: preservando edad/género/emoción, actualizando productos
        with lock_cache:
            cam_cache[persona["id"]] = {
                "edad":      info["edad"],
                "genero":    info["genero"],
                "emocion":   info["emocion"],
                "productos": productos
            }

        ageemo_queue.task_done()

# Lanzar hilos
for _ in range(2):
    threading.Thread(target=ageemo_worker, daemon=True).start()

# ───────── Programa principal ────────────────────────────────────────
def main():
    caps = {}
    for cid in CAMERAS:
        cap = cv2.VideoCapture(cid)
        if not cap.isOpened():
            print(f"[WARN] cámara {cid} no disponible; se omite")
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
        caps[cid]           = cap
        person_cache[cid]   = {}
        frame_counter[cid]  = 0
        last_personas[cid]  = []
        last_det_frame[cid] = 0
        cv2.namedWindow(f"Cam {cid}", cv2.WINDOW_NORMAL)

    if not caps:
        print("No se abrió ninguna cámara.")
        return

    while True:
        for cam_id, cap in caps.items():
            ok, frame_cap = cap.read()
            if not ok:
                continue

            frame_cap  = cv2.flip(frame_cap, 1)
            frame_proc = cv2.resize(frame_cap, (PROC_W, PROC_H))
            fcount = frame_counter[cam_id] = frame_counter[cam_id] + 1

            # Detección
            run_det = (fcount % DETECT_EVERY_N_FRAME == 0)
            dets = detectar_personas(frame_proc) if run_det else []

            # Tracking y persistencia
            if run_det and dets:
                personas = actualizar_tracker(dets, frame_proc, cam_id)
                last_personas[cam_id]  = personas
                last_det_frame[cam_id] = fcount
            else:
                if (fcount - last_det_frame[cam_id]) <= PERSISTENCE_FRAMES:
                    personas = last_personas[cam_id]
                else:
                    personas = []

            # Encolar embedding
            for p in personas:
                enqueue_embed_job(cam_id, p, frame_proc, fcount)

            # Clasificación / segmentación cada N frames
            if fcount % CLASSIFY_EVERY_N_FRAME == 0 and personas:
                for p in personas:
                    ageemo_queue.put((cam_id, p, frame_proc))

            # Dibujo de cajas y etiquetas
            sx, sy = CAPTURE_W/PROC_W, CAPTURE_H/PROC_H
            for p in personas:
                x1, y1, x2, y2 = [int(v*sx) if i%2==0 else int(v*sy)
                                 for i, v in enumerate(p["bbox"])]
                cv2.rectangle(frame_cap, (x1, y1), (x2, y2), (0,255,0), 2)
                with lock_cache:
                    info = person_cache[cam_id].get(p["id"])
                if info:
                    prod_txt = ",".join(info["productos"]) or "-"
                    label = f"ID:{p['id']} {info['genero']} {info['edad']} {prod_txt}"
                else:
                    label = f"ID:{p['id']} ..."
                cv2.putText(frame_cap, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)

            cv2.imshow(f"Cam {cam_id}", frame_cap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
