import cv2

cap = cv2.VideoCapture(0)  # Substitua por VIDEO_PATH, se necessário.
if not cap.isOpened():
    print("Erro ao acessar a câmera ou vídeo.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar frame.")
        break

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
