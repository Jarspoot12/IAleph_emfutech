"""
tracker_deepface.py ─────────────────────────────────────────────────────────────
ID provisional “cam‑tid” → ID global definitivo via DeepFace + re‑ID.
Cada cámara tiene su propio DeepSort (Kalman), pero los embeddings faciales
se comparten para mantener estabilidad de ID al pasar de una cámara a otra.
"""

import cv2
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from deep_sort_realtime.deepsort_tracker import DeepSort

# ─── Hiper‑parámetros ─────────────────────────────────────────────────────────
SIM_THRESH       = 0.80      # similitud mínima para fusionar con ID existente
DS_N_INIT        = 1         # confirma en el primer frame
DS_MAX_AGE       = 30        # retención Kalman (frames)
DS_MAX_COS_DIST  = 0.3       # umbral appearance‑cosine en DeepSORT
PERSIST_FRAMES   = 20        # reservado en tracker (se usa en main.py)
MIN_FACE_PIXELS  = 20 * 20   # área mínima de rostro para intentar embedding

# ─── Un DeepSORT por cámara ───────────────────────────────────────────────────
_trackers = {}

def _get_tracker(cam_id):
    if cam_id not in _trackers:
        _trackers[cam_id] = DeepSort(
            max_age             = DS_MAX_AGE,
            n_init              = DS_N_INIT,
            max_cosine_distance = DS_MAX_COS_DIST
        )
    return _trackers[cam_id]

# ─── Re‑ID global ─────────────────────────────────────────────────────────────
face_features = {}   # global_id → embedding
local2global  = {}   # (cam_id, track_id) → global_id
next_gid      = 1    # contador para nuevos IDs

def _crop_head(bbox, frame, head_ratio=0.35):
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    return frame[y1:y1+int(h * head_ratio), x1:x2]

def actualizar_tracker(detecciones, frame, cam_id=0):
    """
    Args
    ----
      detecciones: [(bbox, conf, cls), ...]
      frame      : imagen BGR de tamaño PROC_W×PROC_H
      cam_id     : identificador de cámara (int o str)
    Returns
    -------
      personas: [ {"id": <str|int>, "bbox":[x1,y1,x2,y2]} ]
    """
    global next_gid
    tracker_ds = _get_tracker(cam_id)

    # 1) Formatear detecciones para DeepSORT
    ds_dets = []
    for bbox, conf, cls in detecciones:
        if int(cls) != 0:
            continue
        x1,y1,x2,y2 = map(int, bbox)
        ds_dets.append(([x1,y1,x2-x1,y2-y1], conf, cls))

    # 2) Actualizar Kalman y asociación
    tracks = tracker_ds.update_tracks(ds_dets, frame=frame)

    # 3) Limpiar mappings huérfanos
    live_tids = {t.track_id for t in tracks if t.is_confirmed()}
    for key in list(local2global):
        if key[0] == cam_id and key[1] not in live_tids:
            del local2global[key]

    personas = []

    # 4) Procesar todos los tracks activos (sin filtrar is_confirmed)
    for t in tracks:
        if t.time_since_update > PERSIST_FRAMES:
            continue  # demasiado tiempo sin detección real

        tid = t.track_id
        x1,y1,x2,y2 = map(int, t.to_ltrb())
        key = (cam_id, tid)

        # 4.1 ID provisional (siempre dibujamos)
        gid_tmp = f"{cam_id}-{tid}"
        persona_dict = {"id": gid_tmp, "bbox": [x1,y1,x2,y2]}
        personas.append(persona_dict)

        # 4.2 Solo en frames con detección real → intentar embedding
        if t.time_since_update == 0:
            head = _crop_head((x1,y1,x2,y2), frame)
            rep = None
            if head.size >= MIN_FACE_PIXELS:
                try:
                    rep = DeepFace.represent(
                        head, model_name="Facenet",
                        enforce_detection=False
                    )[0]["embedding"]
                except:
                    rep = None

            if rep is not None:
                # a) mapping local→global ya existente?
                if key in local2global:
                    gid = local2global[key]
                    face_features[gid] = rep
                else:
                    # b) buscar coincidencia global
                    best_gid, best_sim = None, -1
                    for g,emb in face_features.items():
                        sim = cosine_similarity([rep],[emb])[0][0]
                        if sim > best_sim:
                            best_gid, best_sim = g, sim
                    if best_sim >= SIM_THRESH:
                        gid = best_gid
                        face_features[gid] = rep
                    else:
                        gid = next_gid
                        face_features[gid] = rep
                        next_gid += 1
                    local2global[key] = gid

                # c) sustituir provisional por ID global definitivo
                persona_dict["id"] = gid

    return personas
