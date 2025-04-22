import os, sys, cv2, threading, queue

# Suprimir GPU y avisos verbosos
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "tracking", "fast-reid"))

from detectors.yolov8m         import detectar_personas
from tracking.tracker_deepface import actualizar_tracker
from classification.age_gender import clasificar_edad_genero
from classification.emotion    import reconocer_emocion

# ─── Parámetros ───────────────────────────────────────────────────────
CAPTURE_W, CAPTURE_H     = 640, 480
PROC_W,    PROC_H        = 640, 480
DETECT_EVERY_N_FRAME     = 3
CLASSIFY_EVERY_N_FRAME   = 30
PERSISTENCE_FRAMES       = 10    # cuántos frames sin detección mantenemos el bbox
CAMERAS                  = [0, 2]

# ─── Estructuras globales ─────────────────────────────────────────────
heavy_queue     = queue.Queue(maxsize=50)
person_cache    = {}    # para edad/género/emoción
frame_counter   = {}
lock_cache      = threading.Lock()

# para persistencia de boxes
last_personas   = {}
last_det_frame  = {}

def heavy_worker():
    """Hilo de clasificación pesada: edad, género y emoción."""
    MIN_ROI = 20
    while True:
        try:
            cam_id, personas, frame = heavy_queue.get(timeout=1)
        except queue.Empty:
            continue

        for p in personas:
            x1,y1,x2,y2 = map(int, p["bbox"])
            if min(x2-x1, y2-y1) < MIN_ROI:
                continue
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

            with lock_cache:
                person_cache[cam_id][p["id"]] = {
                    "edad": edad,
                    "genero": genero,
                    "emocion": emocion
                }
        heavy_queue.task_done()

def main():
    # 1) Abrir cámaras
    caps = {}
    for cid in CAMERAS:
        cap = cv2.VideoCapture(cid)
        if not cap.isOpened():
            print(f"[WARN] Cámara {cid} no disponible — se omite")
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
        print("No se abrió ninguna cámara :(")
        return

    threading.Thread(target=heavy_worker, daemon=True).start()

    # 2) Bucle principal
    while True:
        for cam_id, cap in caps.items():
            ret, frame_cap = cap.read()
            if not ret:
                continue

            frame_cap  = cv2.flip(frame_cap, 1)
            frame_proc = cv2.resize(frame_cap, (PROC_W, PROC_H))
            frame_counter[cam_id] += 1
            fcount = frame_counter[cam_id]

            # DETECCIÓN cada N frames
            run_detector = (fcount % DETECT_EVERY_N_FRAME == 0)
            detections   = detectar_personas(frame_proc) if run_detector else []

            # TRACKING
            if run_detector and detections:
                # nueva detección: actualizamos y reseteamos el contador
                personas = actualizar_tracker(detections, frame_proc, cam_id)
                last_personas[cam_id]  = personas
                last_det_frame[cam_id] = fcount
            else:
                # sin detección nueva: mantenemos hasta PERSISTENCE_FRAMES
                if (fcount - last_det_frame[cam_id]) <= PERSISTENCE_FRAMES:
                    personas = last_personas[cam_id]
                else:
                    personas = []

            # DEBUG rápido para verificar persistencia
            print(f"[DEBUG Cam {cam_id}] frame {fcount}, dets: {len(detections)}, personas: {len(personas)}")

            # Clasificación pesada solo con detección real
            if run_detector and detections and fcount % CLASSIFY_EVERY_N_FRAME == 0:
                heavy_queue.put_nowait((cam_id, personas, frame_proc))

            # Dibujo de cajas y etiquetas
            sx, sy = CAPTURE_W/PROC_W, CAPTURE_H/PROC_H
            for p in personas:
                x1,y1,x2,y2 = [int(v*sx) if i%2==0 else int(v*sy)
                               for i,v in enumerate(p["bbox"])]
                cv2.rectangle(frame_cap, (x1,y1), (x2,y2), (0,255,0), 2)
                with lock_cache:
                    info = person_cache[cam_id].get(p["id"])
                txt = f"ID:{p['id']} " + (f"{info['genero']},{info['edad']},{info['emocion']}"
                                          if info else "...")
                cv2.putText(frame_cap, txt, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            cv2.imshow(f"Cam {cam_id}", frame_cap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 3) Liberar recursos
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
