from flask import Flask, render_template, request, jsonify
import cv2
import torch
from easyocr import Reader

app = Flask(__name__)

# Завантаження моделі YOLO
model = torch.hub.load('ultralytics/yolov5', 'yolov8', pretrained=True)

# Ініціалізація EasyOCR
reader = Reader(['en'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_plate', methods=['POST'])
def detect_plate():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Детекція номерного знака за допомогою YOLO
    results = model(img)
    plates = results.pandas().xywh[results.pandas().xywh['class'] == 2]  # Class 2 for vehicle

    if plates.empty:
        return jsonify({'error': 'No license plate detected'}), 400

    plate_image = img[plates.iloc[0]['ymin']:plates.iloc[0]['ymax'], plates.iloc[0]['xmin']:plates.iloc[0]['xmax']]

    # Розпізнавання символів на номерному знаку
    result_text = reader.readtext(plate_image)
    plate_number = ' '.join([text[1] for text in result_text])

    return jsonify({'plate_number': plate_number})

if __name__ == '__main__':
    app.run(debug=True)
