import cv2
import time
from yolo_detector import YoloDetector
from tracker import Tracker

MODEL_PATH = "yolo11n.pt"
VIDEO_PATH = 0


def main():
    
    # Inicializa variáveis
    detector = YoloDetector(model_path=MODEL_PATH, confidence=0.2)
    tracker = Tracker()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        exit()

    current_ids = set()  # IDs sendo rastreados no quadro atual
    total_ids = set()    # IDs únicos de todas as pessoas detectadas
    total_count = 0      # Contador total de pessoas que já passaram
    frame_count = 0      # Contador de quadros

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.perf_counter()

        # Detectar objetos no quadro
        detections = detector.detect(frame)
        tracking_ids, boxes = tracker.track(detections, frame)

        # Atualizar conjuntos de IDs
        new_ids = set(tracking_ids)  # IDs no quadro atual
        if not new_ids:  # Resetar contagem se nenhum ID estiver presente
            current_ids.clear()

        # Atualizar contagem de IDs únicos
        for tracking_id in new_ids:
            if tracking_id not in total_ids:
                total_ids.add(tracking_id)  # Adicionar ID ao conjunto único
                total_count += 1  # Incrementar o contador total

        current_ids = new_ids  # Atualizar os IDs rastreados no quadro atual

        # Desenhar caixas delimitadoras e IDs nos objetos rastreados
        for tracking_id, bounding_box in zip(tracking_ids, boxes):
            cv2.rectangle(
                frame,
                (int(bounding_box[0]), int(bounding_box[1])),
                (int(bounding_box[2]), int(bounding_box[3])),
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                f"ID: {tracking_id}",
                (int(bounding_box[0]), int(bounding_box[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )


        # Exibir informações no quadro
        cv2.putText(frame, f"Total Pessoas: {total_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Pessoas Atuais: {len(current_ids)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        end_time = time.perf_counter()
        fps = 1 / (end_time - start_time)
        print(f"Current fps: {fps}")

        # Mostrar o vídeo com as detecções
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Total IDs únicos detectados:", total_ids)
    print("Contagem total de pessoas:", total_count)
    print(detections)


if __name__ == "__main__":
    main()
