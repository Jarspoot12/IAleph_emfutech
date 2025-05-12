"""La función actualizar_tracker de tracker_deepface asigna un persona["id"] 
global que persiste incluso al cambiar de cámara, gracias a DeepSORT + DeepFace embeddings.
 Este gid es el que usas aquí para cachear la edad/género/emoción.
"""

from collections import deque
import cv2, threading, queue
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from deep_sort_realtime.deepsort_tracker import DeepSort
from mtcnn import MTCNN

# ─── parámetros ───────────────────────────────────────────────────────
SIM_THRESH       = 0.8       # más estricto
DS_N_INIT        = 1
DS_MAX_AGE       = 10          # 3× DETECT interval
DS_MAX_COS_DIST  = 0.25
MIN_FACE_PIXELS  = 10*10
RECHECK_EMBED    = 40         # re‑embeddings cada 40 frames
STALE_FRAMES = 15          # ≈ medio segundo con detector cada 3 frames
_mtcnn = MTCNN()



# ─── trackers por cámara ──────────────────────────────────────────────
_trackers = {}                                    # cam_id → DeepSort
embed_queue = queue.Queue()                       # jobs de DeepFace
embed_lock  = threading.Lock()                    # protege face_features

face_features = {}
local2global  = {}
last_seen     = {}         # gid → frame_global
next_gid      = 1
frame_global  = 0

def _get_tracker(cam):
    if cam not in _trackers:
        _trackers[cam] = DeepSort(
            max_age             = DS_MAX_AGE,
            n_init              = DS_N_INIT,
            max_cosine_distance = DS_MAX_COS_DIST
        )
    return _trackers[cam]

# ─── hilo dedicado a embeddings ───────────────────────────────────────
def _embed_worker():
    global next_gid, frame_global
    while True:
        cam_id, persona, frame = embed_queue.get()
        # 1) recorta la zona aproximada
        x1, y1, x2, y2 = map(int, persona["bbox"])
        patch = frame[y1:y2, x1:x2]

        # 2) detecta solo la cara dentro de ese parcho
        faces = _mtcnn.detect_faces(patch)
        if not faces:
            embed_queue.task_done()
            continue

        # 3) elige la cara con más confianza
        face_box = max(faces, key=lambda f: f['confidence'])['box']
        fx, fy, fw, fh = face_box
        face_crop = patch[fy:fy+fh, fx:fx+fw]
        

        # 1) Filtrar recortes pequeños
        if face_crop.size < MIN_FACE_PIXELS:
            embed_queue.task_done()
            continue

        # 2) Embedding con ArcFace
        try:
            rep = DeepFace.represent(
                    face_crop, model_name="ArcFace",
                    enforce_detection=True)[0]["embedding"]
        except Exception:
            embed_queue.task_done()
            continue

        key = (cam_id, persona["tid"])

        with embed_lock:
            # 3) Si el track ya tiene gid, refrescar embedding y last_seen
            if key in local2global:
                gid = local2global[key]
                face_features[gid] = rep
                last_seen[gid]    = frame_global
            else:
                # 4) Buscar gid candidato inactivo con similitud ≥ SIM_THRESH
                best_gid, best_sim = None, -1
                for g, emb in face_features.items():
                    # ¿cuánto hace que NO se ve?
                    inactive = (frame_global - last_seen.get(g, -1)) > STALE_FRAMES
                    if not inactive:
                        continue
                    sim = cosine_similarity([rep], [emb])[0][0]
                    if sim > best_sim and sim >= SIM_THRESH:
                        best_gid, best_sim = g, sim

                # 5) Asignar gid
                if best_gid is not None:        # reutilizar gid
                    gid = best_gid
                else:                           # crear nuevo gid
                    gid = next_gid
                    next_gid += 1
                face_features[gid] = rep
                local2global[key]  = gid
                last_seen[gid]     = frame_global

            # 6) Actualizar el diccionario persona in-place
            persona["id"] = gid

        embed_queue.task_done()

threading.Thread(target=_embed_worker, daemon=True).start()

# ─── API para main.py ─────────────────────────────────────────────────
def enqueue_embed_job(cam_id, persona, frame, fcount):
    """Encola un trabajo de DeepFace si toca re‑embedding."""
    if (fcount - persona.get("last_embed", -RECHECK_EMBED)) >= RECHECK_EMBED:
        persona["last_embed"] = fcount
        embed_queue.put((cam_id, persona.copy(), frame.copy()))

def actualizar_tracker(dets, frame, cam_id):
    tracker = _get_tracker(cam_id)
    global frame_global
    frame_global += 1

    ds_dets = []
    for bbox, conf, cls in dets:
        if int(cls) != 0: continue
        x1,y1,x2,y2 = map(int, bbox)
        ds_dets.append(([x1,y1,x2-x1,y2-y1], conf, cls))

    tracks = tracker.update_tracks(ds_dets, frame=frame)

    personas = []
    for t in tracks:
        if t.time_since_update > DS_MAX_AGE: continue
        x1,y1,x2,y2 = map(int, t.to_ltrb())
        tid = t.track_id
        key = (cam_id, tid)
        gid = local2global.get(key, f"{cam_id}-{tid}")

        personas.append({
            "id":  gid,
            "tid": tid,
            "bbox":[x1,y1,x2,y2]
        })
        if t.time_since_update == 0 and key in local2global:
            last_seen[local2global[key]] = frame_global

    return personas



