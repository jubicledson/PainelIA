from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

# Inicializar o modelo YOLOv11
model = YOLO("yolo11n.pt")  # Certifique-se de ter o modelo YOLOv11 correto

# Inicializar o DeepSORT
tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)

# Inicializar a câmera (vídeo ao vivo)
cap = cv2.VideoCapture(0)  # Substitua 0 pelo índice correto ou caminho do vídeo

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Erro ao acessar a câmera. Verifique a conexão.")
        break

    # Fazer predições com YOLO
    results = model.predict(source=frame, stream=True)

    # Processar as detecções de pessoas
    detections = []
    for result in results:
        for box in result.boxes:
            cls, conf = int(box.cls[0]), box.conf[0].item()  # Convertendo tensor para número
            if cls == 0 and conf > 0.5:  # Classe 0 é 'Pessoa'
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width, height = x2 - x1, y2 - y1
                detections.append([x1, y1, width, height, conf])  # [x, y, w, h, conf]

    # Validar as detecções antes de rastrear
    if len(detections) > 0:
        print("Detecções formatadas:", detections)
    else:
        print("Nenhuma detecção válida encontrada.")

    # Rastrear os objetos com DeepSORT
    try:
        tracks = tracker.update_tracks(detections, frame=frame)
    except AssertionError as e:
        print("Erro de formato nas detecções:", e)
        continue

    # Desenhar rastreamentos no quadro
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Desenhar caixa delimitadora e ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar o número total de pessoas rastreadas
    cv2.putText(frame, f"Pessoas rastreadas: {len(tracks)}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Exibir o feed com detecção e rastreamento
    cv2.imshow("Detecção e Rastreamento - YOLOv11 + DeepSORT", frame)

    # Finalizar com a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
