import base64
import threading
import time
import io
import cv2
import dlib
import numpy as np
from PIL import Image, ExifTags
from flask import Flask, request, jsonify
import face_recognition

# RTSP stream
RTSP_URL = "rtsp://admin:12345678@192.168.0.54:554/LowResolutionVideo"
MATCH_THRESHOLD = 0.6
FRAME_INTERVAL = 5  # Process every Nth frame

# Dlib models
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
shape_predictor = dlib.shape_predictor(shape_predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

app = Flask(__name__)

# Shared variables
current_reference_encoding = None
reference_lock = threading.Lock()
match_result = None  # True / False
display_frame = None

# ========== Helper functions ==========

def fix_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif:
            orientation_value = exif.get(orientation)
            if orientation_value == 3:
                image = image.rotate(180, expand=True)
            elif orientation_value == 6:
                image = image.rotate(270, expand=True)
            elif orientation_value == 8:
                image = image.rotate(90, expand=True)
    except Exception:
        pass
    return image

def extract_face_encoding(image_np):
    face_locations = face_recognition.face_locations(image_np)
    if not face_locations:
        return None

    # Только ближайшее лицо
    top, right, bottom, left = face_locations[0]
    dlib_rect = dlib.rectangle(left, top, right, bottom)
    shape = shape_predictor(image_np, dlib_rect)
    encoding = np.array(face_rec_model.compute_face_descriptor(image_np, shape, 1))
    return encoding

def decode_base64_image(base64_string):
    try:
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_bytes))
        image = fix_orientation(image)
        return np.array(image.convert('RGB'))
    except Exception as e:
        print(f"Ошибка при декодировании изображения: {e}")
        return None

# ========== RTSP Thread ==========

def rtsp_face_check_loop():
    global match_result, display_frame

    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("❌ Не удалось подключиться к RTSP потоку")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.2)
            continue

        display_frame = frame.copy()
        frame_count += 1
        if frame_count % FRAME_INTERVAL != 0:
            continue

        with reference_lock:
            ref = current_reference_encoding

        if ref is None:
            continue

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        if not face_locations:
            match_result = False
            continue

        # Увеличиваем координаты к оригинальному размеру
        top, right, bottom, left = [v * 4 for v in face_locations[0]]
        dlib_rect = dlib.rectangle(left, top, right, bottom)
        shape = shape_predictor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dlib_rect)
        encoding = np.array(face_rec_model.compute_face_descriptor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), shape, 1))

        distance = np.linalg.norm(encoding - ref)
        if distance < MATCH_THRESHOLD:
            match_result = True
        else:
            match_result = False

        # Draw box
        color = (0, 255, 0) if match_result else (0, 0, 255)
        cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
        cv2.putText(display_frame, f'Distance: {distance:.2f}', (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# ========== Video display ==========

def video_display_loop():
    global display_frame
    while True:
        if display_frame is not None:
            show = display_frame.copy()
            result_text = "MATCH" if match_result else "NO MATCH"
            color = (0, 255, 0) if match_result else (0, 0, 255)
            cv2.putText(show, result_text, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.imshow("Live RTSP Face Match", show)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.01)
    cv2.destroyAllWindows()

# ========== Flask Route ==========

@app.route('/face-check', methods=['POST'])
def face_check():
    global current_reference_encoding, match_result

    data = request.get_json()
    if not data or 'image_base64' not in data or 'timeout' not in data:
        return jsonify({"error": "Required fields: image_base64 and timeout"}), 400

    timeout = float(data['timeout'])
    image_np = decode_base64_image(data['image_base64'])
    if image_np is None:
        return jsonify({"error": "Could not decode image"}), 400

    encoding = extract_face_encoding(image_np)
    if encoding is None:
        return jsonify({"error": "No face found in uploaded image"}), 400

    with reference_lock:
        current_reference_encoding = encoding
        match_result = None

    start_time = time.time()
    result = None

    while time.time() - start_time < timeout:
        if match_result is True:
            result = "MATCH"
            break
        elif match_result is False:
            pass  # continue waiting
        time.sleep(0.1)

    if result is None:
        result = "NO MATCH (timeout)"

    # === СБРОС состояния после проверки ===
    with reference_lock:
        current_reference_encoding = None
        match_result = None

    return jsonify({"result": result})


# ========== Run server and threads ==========

if __name__ == '__main__':
    threading.Thread(target=rtsp_face_check_loop, daemon=True).start()
    threading.Thread(target=video_display_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)
