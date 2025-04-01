from deep_sort_realtime.deepsort_tracker import DeepSort

# Inicializar el tracker Deep SORT con un parámetro 'max_age' para definir el tiempo de retención de tracks
tracker = DeepSort(max_age=30)

def actualizar_tracker(detecciones, frame):
    """
    Actualiza el tracker con las detecciones del frame actual.
    
    Parámetros:
      - detecciones: Lista de bounding boxes [x1, y1, x2, y2] provenientes de YOLOv8.
      - frame: Frame actual del video, usado para calcular las características.
    
    Retorna:
      - personas: Lista de diccionarios con cada persona trackeada, que incluye un ID único y su bounding box.
    """
    # Actualiza el tracker con las detecciones y el frame actual
    tracks = tracker.update_tracks(detecciones, frame=frame)
    
    personas = []
    for track in tracks:
        # Solo procesar tracks confirmados (evitar tracks inestables)
        if not track.is_confirmed():
            continue
        track_id = track.track_id              # Obtener el ID asignado al track
        bbox = track.to_ltrb()                 # Obtener el bounding box en formato [left, top, right, bottom]
        personas.append({'id': track_id, 'bbox': bbox})
    
    return personas

# Bloque de prueba (se ejecuta solo si se corre este archivo directamente)
if __name__ == "__main__":
    # Este bloque se integrará en main.py para pruebas reales
    pass
