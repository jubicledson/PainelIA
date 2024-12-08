from datetime import datetime, timedelta
import cv2
import time
from threading import Thread, Lock
import pymysql
import torch
from yolo_detector import YoloDetector
from tracker import Tracker

MODEL_PATH = "C:\\Users\\Usuario\\Desktop\\Painel\\DeteccaoIA\\yoloD\\yolo11n_ncnn_model"
VIDEO_PATH = 0  # Pode ser substituído por um URL ou arquivo de vídeo.

# Variáveis globais
frame_buffer = []
buffer_lock = Lock()

# Função para leitura assíncrona de frames
def read_frames(cap):
    global frame_buffer
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        with buffer_lock:
            frame_buffer.append(frame)

# Função para envio ao banco
def send_to_database(people_info, db_config):
    filtered_info = {
        k: v for k, v in people_info.items()
        if v["exit_time"] and (datetime.strptime(v["exit_time"], "%Y-%m-%d %H:%M:%S") - datetime.strptime(v["entry_time"], "%Y-%m-%d %H:%M:%S")).total_seconds() >= 3
    }

    if not filtered_info:
        return

    try:
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()
        for person_id, info in filtered_info.items():
            cursor.execute(
                "INSERT INTO registros (painel_id, person_id, entry_time, exit_time, face_detected) VALUES (%s, %s, %s, %s, %s)",
                (1, person_id, info["entry_time"], info["exit_time"], info["face_detected"]),
            )
        conn.commit()
        for person_id in filtered_info:
            del people_info[person_id]
    except Exception as e:
        print(f"Erro no banco: {e}")
    finally:
        conn.close()

def main():
    db_config = {"host": "localhost", "user": "root", "password": "", "database": "dbpaineis"}
    send_interval = timedelta(seconds=60)  # Alterado para 60 segundos
    last_send_time = datetime.now()

    # Passando o dispositivo (GPU/CPU) para o YoloDetector
    detector = YoloDetector(model_path=MODEL_PATH, confidence=0.5)
    tracker = Tracker()
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    # Configurar buffers para otimização de leitura
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    read_thread = Thread(target=read_frames, args=(cap,))
    read_thread.start()

    people_info = {}
    frame_skip = 4  # Pular 3 quadros entre processamentos
    frame_counter = 0

    while True:
        with buffer_lock:
            if frame_buffer:
                frame = frame_buffer.pop(0)
            else:
                continue

        frame_counter += 1
        if frame_counter % frame_skip != 0:
            continue

        frame = cv2.resize(frame, (620, 480))  # Resolução reduzida
        roi = frame

        # Detectar rostos
        faces = face_classifier.detectMultiScale(roi, 1.1, 5, minSize=(40, 40))
        face_detected = len(faces) > 0

        # Desenhar os retângulos de rosto
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Detectar e rastrear pessoas
        detections = detector.detect(roi)
        tracking_ids, boxes = tracker.track(detections, roi)

        current_ids = set(tracking_ids)
        for person_id, box in zip(tracking_ids, boxes):
            # Desenhar o retângulo ao redor das pessoas
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {person_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            if person_id not in people_info:
                people_info[person_id] = {"entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "exit_time": None, "face_detected": False}

            if face_detected:
                people_info[person_id]["face_detected"] = True

        for person_id in list(people_info.keys()):
            if person_id not in current_ids and people_info[person_id]["exit_time"] is None:
                people_info[person_id]["exit_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Enviar ao banco de dados a cada 60 segundos
        if datetime.now() - last_send_time >= send_interval:
            send_to_database(people_info, db_config)
            last_send_time = datetime.now()

        # Exibir o frame
        cv2.imshow("Detecção e Rastreamento", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Enviar registros restantes ao banco antes de sair
    send_to_database(people_info, db_config)

if __name__ == "__main__":
    main()
