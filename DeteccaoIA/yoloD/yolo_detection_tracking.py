from datetime import datetime, timedelta
import cv2
import time
from threading import Thread, Lock
import pymysql
from yolo_detector import YoloDetector
from tracker import Tracker

MODEL_PATH = "yolo11n.pt"
VIDEO_PATH = 0  # Pode ser substituído por um URL de vídeo ou arquivo de vídeo.

# Variáveis globais para leitura assíncrona
frame_buffer = []
buffer_lock = Lock()

# Função para ler frames do vídeo em um thread separado
def read_frames(cap):
    global frame_buffer
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        with buffer_lock:
            frame_buffer.append(frame)

# Função para enviar registros ao banco de dados
def send_to_database(people_info, db_config):
    # Filtrar registros com `exit_time` definido e duração superior a 3 segundos
    filtered_info = {}
    for person_id, info in people_info.items():
        # Garantir que both `entry_time` e `exit_time` sejam válidos
        if info["exit_time"] is not None:
            entry_time = datetime.strptime(info["entry_time"], "%Y-%m-%d %H:%M:%S")
            exit_time = datetime.strptime(info["exit_time"], "%Y-%m-%d %H:%M:%S")
            duration = (exit_time - entry_time).total_seconds()

            # Incluir no filtro apenas se a duração for maior que 3 segundos
            if duration >= 3:
                filtered_info[person_id] = info

    if not filtered_info:
        return

    try:
        conn = pymysql.connect(
            host=db_config["host"],
            user=db_config["user"],
            password=db_config["password"],
            database=db_config["database"],
        )
        cursor = conn.cursor()

        for person_id, info in filtered_info.items():
            entry_time = info["entry_time"]
            exit_time = info["exit_time"]
            face_detected = info["face_detected"]
            cursor.execute(
                "INSERT INTO registros (painel_id, person_id, entry_time, exit_time, face_detected) VALUES (%s, %s, %s, %s, %s)",
                (1, person_id, entry_time, exit_time, face_detected),
            )
        conn.commit()
        print(f"{len(filtered_info)} registros enviados ao banco de dados.")

        # Limpar os registros que já foram enviados
        for person_id in filtered_info:
            del people_info[person_id]  # Excluindo os registros enviados

    except Exception as e:
        print(f"Erro ao enviar registros para o banco de dados: {e}")
    finally:
        conn.close()

def main():
    # Configuração do banco de dados
    db_config = {
        "host": "localhost",
        "user": "root",
        "password": "",  # Substitua pela senha do seu banco
        "database": "dbpaineis",
    }

    # Configurações de envio
    send_interval = timedelta(seconds=10)  # Intervalo de envio
    last_send_time = datetime.now()

    # Inicializar detector e rastreador
    detector = YoloDetector(model_path=MODEL_PATH, confidence=0.5)
    tracker = Tracker()
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Abrir o vídeo
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Erro: Não foi possível abrir o vídeo.")
        return

    # Iniciar thread para leitura assíncrona
    read_thread = Thread(target=read_frames, args=(cap,))
    read_thread.start()

    # Dicionário para armazenar informações de pessoas
    people_info = {}

    # Variáveis de controle de processamento
    frame_skip = 2
    frame_counter = 0

    while True:
        with buffer_lock:
            if frame_buffer:
                frame = frame_buffer.pop(0)
            else:
                continue

        frame_counter += 1
        if frame_counter % frame_skip != 0:
            continue  # Pular este quadro

        # Reduzir a resolução do frame
        frame = cv2.resize(frame, (640, 480))

        # Delimitar área de interesse (ROI)
        roi = frame

        start_time = time.perf_counter()

        # Detectar faces frontais
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_roi, 1.1, 5, minSize=(40, 40))
        face_detected = len(faces) > 0

        # Desenhar retângulos ao redor de faces detectadas
        for (x, y, w, h) in faces:
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Detectar e rastrear pessoas
        detections = detector.detect(roi)
        tracking_ids, boxes = tracker.track(detections, roi)

        # Atualizar informações de IDs rastreados
        current_ids = set(tracking_ids)

        for person_id, box in zip(tracking_ids, boxes):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Retângulo seguindo as pessoas
            cv2.putText(frame, f"ID: {person_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            if person_id not in people_info:
                people_info[person_id] = {
                    "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "exit_time": None,
                    "face_detected": False,
                }

            if face_detected:
                people_info[person_id]["face_detected"] = True

        for person_id in list(people_info.keys()):
            if person_id not in current_ids and people_info[person_id]["exit_time"] is None:
                people_info[person_id]["exit_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Mostrar FPS
        end_time = time.perf_counter()
        fps = 1 / (end_time - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Exibir número de pessoas rastreadas
        cv2.putText(frame, f"Pessoas: {len(current_ids)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Exibir vídeo
        cv2.imshow("Detecção e Rastreamento", frame)

        # Enviar dados ao banco em intervalos definidos
        if datetime.now() - last_send_time >= send_interval:
            send_to_database(people_info, db_config)

            # Remover registros com `exit_time` como None ou duração inferior a 3 segundos
            people_info = {k: v for k, v in people_info.items() if v["exit_time"] is not None}
            people_info = {k: v for k, v in people_info.items() if (datetime.strptime(v["exit_time"], "%Y-%m-%d %H:%M:%S") - datetime.strptime(v["entry_time"], "%Y-%m-%d %H:%M:%S")).total_seconds() >= 3}
            last_send_time = datetime.now()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Enviar registros restantes antes de sair
    send_to_database(people_info, db_config)

if __name__ == "__main__":
    main()
