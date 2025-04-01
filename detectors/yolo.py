import cv2
from ultralytics import YOLO

# Cargar el modelo YOLO (versión ligera para mayor velocidad)
model = YOLO('yolov8n.pt')

# Inicializar la captura de video con la cámara predeterminada
cap = cv2.VideoCapture(0)

# Reducir la resolución a 640x480 para disminuir la carga de procesamiento
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Crear una ventana única para mostrar los resultados
cv2.namedWindow('Detección YOLOv8', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar el frame.")
        break

    # Ejecutar el modelo sobre el frame actual (sin 'stream=True' para una sola predicción)
    results = model(frame)
    
    # Obtener el frame anotado (usamos el primer resultado)
    annotated_frame = results[0].plot()

    # Mostrar el frame anotado en la ventana única
    cv2.imshow('Detección YOLOv8', annotated_frame)

    # Espera 30 ms entre frames; presiona 'q' para salir
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar la ventana
cap.release()
cv2.destroyAllWindows()
