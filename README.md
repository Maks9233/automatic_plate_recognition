# Automatic License Plate Recognition System

Цей проєкт реалізує систему автоматичного розпізнавання номерних знаків транспортних засобів з використанням YOLOv8 та EasyOCR.

## 🔧 Встановлення

1. Клонуйте репозиторій:
   ```bash
   git clone https://github.com/Maks9233/automatic_plate_recognition.git
   cd automatic_plate_recognition
   ```

2. Створіть та активуйте віртуальне середовище (опціонально):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Для Linux/MacOS
   venv\Scripts\activate   # Для Windows
   ```

3. Встановіть залежності:
   ```bash
   pip install -r requirements.txt
   ```

4. Запустіть програму:
   ```bash
   python main.py
   ```

## 📂 Структура проєкту

- `main.py`: Головний скрипт для запуску системи.
- `best.pt`: Натренована модель YOLOv8.
- `plates.db`: База даних SQLite для збереження результатів.
- `recognized/`: Каталог для збереження зображень розпізнаних номерів.

## 📄 Ліцензія

Цей проєкт ліцензований під MIT License.
