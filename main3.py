import os, sys, cv2, threading, queue, json, time

# ───────── Configuración de entorno ─────────────────────────────────#─
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "tracking", "fast-reid"))

from detectors.yolov8m          import detectar_personas
from tracking.tracker_deepface  import actualizar_tracker, enqueue_embed_job
from classification.age_gender  import clasificar_edad_genero
from classification.emotion     import reconocer_emocion
from segmentation.segmentation  import segmentar_productos

# ───────── Parámetros globales ───────────────────────────────────────
CAPTURE_W, CAPTURE_H   = 640, 480
PROC_W,    PROC_H      = 640, 480
DETECT_EVERY_N_FRAME   = 3       # YOLO cada 3 frames
CLASSIFY_EVERY_N_FRAME = 30      # clasificación y segmentación cada 30 frames
PERSISTENCE_FRAMES     = 30      # persistencia de bbox sin detección
CAMERAS                = [0, 2]

# ───────── Estructuras compartidas ───────────────────────────────────
ageemo_queue   = queue.Queue()       # ilimitada
person_cache   = {}                  # {cam_id: {gid: info}}
frame_counter  = {}
last_personas  = {}
last_det_frame = {}
lock_cache     = threading.Lock()

# ───────── Hilo de clasificación de atributos y segmentación ────────
def ageemo_worker():
    while True:
        cam_id, persona, frame = ageemo_queue.get()
        x1, y1, x2, y2 = map(int, persona["bbox"])
        roi_rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        roi_in  = cv2.resize(roi_rgb, (112,112))

        try:
            edad, genero = clasificar_edad_genero(roi_in)
        except:
            edad, genero = "-", "-"
        try:
            emocion = reconocer_emocion(roi_in) or "-"
        except:
            emocion = "-"

        productos = segmentar_productos(roi_rgb, conf=0.65)

        registro = {
            "camara": cam_id,
            "id": persona["id"],
            "bbox": persona["bbox"],
            "edad": edad,
            "genero": genero,
            "emocion": emocion,
            "productos": productos,
            "timestamp": time.time()
        }
        print(json.dumps(registro))

        with lock_cache:
            person_cache[cam_id][persona["id"]] = {
                "edad": edad,
                "genero": genero,
                "emocion": emocion,
                "productos": productos
            }
        ageemo_queue.task_done()

# lanzar dos hilos para clasificación
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
            frame_counter[cam_id] += 1
            fcount = frame_counter[cam_id]

            # detección cada DETECT_EVERY_N_FRAME frames
            run_det = (fcount % DETECT_EVERY_N_FRAME == 0)
            dets = detectar_personas(frame_proc) if run_det else []

            # tracking y persistencia
            if run_det and dets:
                personas = actualizar_tracker(dets, frame_proc, cam_id)
                last_personas[cam_id]  = personas
                last_det_frame[cam_id] = fcount
            else:
                if (fcount - last_det_frame[cam_id]) <= PERSISTENCE_FRAMES:
                    personas = last_personas[cam_id]
                else:
                    personas = []

            # re-ID embedding jobs
            for p in personas:
                enqueue_embed_job(cam_id, p, frame_proc, fcount)

            # clasificación y segmentación cada CLASSIFY_EVERY_N_FRAME
            if fcount % CLASSIFY_EVERY_N_FRAME == 0 and personas:
                for p in personas:
                    ageemo_queue.put((cam_id, p, frame_proc))

            # dibujo de cajas y etiquetas
            sx, sy = CAPTURE_W/PROC_W, CAPTURE_H/PROC_H
            for p in personas:
                x1, y1, x2, y2 = [int(v*sx) if i%2==0 else int(v*sy)
                                 for i, v in enumerate(p["bbox"]) ]
                cv2.rectangle(frame_cap, (x1,y1), (x2,y2), (0,255,0), 2)
                with lock_cache:
                    info = person_cache[cam_id].get(p["id"])
                if info:
                    prod_txt = ",".join(it["label"] for it in info["productos"]) or "-"
                    label = f"ID:{p['id']} {info['genero']} {info['edad']} {info['emocion']} {prod_txt}"
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

