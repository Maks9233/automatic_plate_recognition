
import cv2
from ultralytics import YOLO
from easyocr import Reader
from datetime import datetime
import sqlite3
import os

# Ініціалізація моделі YOLOv8 (використовуй свою натреновану модель)
model = YOLO('best.pt')  # заміни на актуальний шлях до моделі
ocr = Reader(['en', 'uk'], gpu=False)

# Підключення до SQLite бази даних або створення нової
if not os.path.exists("plates.db"):
    conn = sqlite3.connect("plates.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE recognition_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_text TEXT,
            confidence REAL,
            timestamp TEXT,
            image_path TEXT
        )
    """)
    conn.commit()
else:
    conn = sqlite3.connect("plates.db")

# Відкриття відео з камери (можна змінити на rtsp або відеофайл)
video_source = cv2.VideoCapture(0)

# Каталог для збереження кадрів
os.makedirs("recognized", exist_ok=True)


def recognize_plate(frame):
    results = model(frame)[0]
    plates = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = frame[y1:y2, x1:x2]

        ocr_result = ocr.readtext(cropped)
        for text in ocr_result:
            plate_text = text[1]
            confidence = text[2]
            if confidence > 0.5:
                filename = f"recognized/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{plate_text}.jpg"
                cv2.imwrite(filename, cropped)

                # Зберегти в БД
                conn.execute(
                    "INSERT INTO recognition_results (plate_text, confidence, timestamp, image_path) VALUES (?, ?, ?, ?)",
                    (plate_text, confidence, datetime.now().isoformat(), filename)
                )
                conn.commit()

                # Малюємо рамку і підпис
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, plate_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return frame


while True:
    ret, frame = video_source.read()
    if not ret:
        break

    annotated_frame = recognize_plate(frame)

    cv2.imshow("License Plate Recognition", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_source.release()
cv2.destroyAllWindows()
conn.close()
