from flask import Flask, request, jsonify
import base64
import numpy as np
from PIL import Image, ExifTags
import io
import dlib
import cv2
import face_recognition
app = Flask(__name__)

shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"

shape_predictor = dlib.shape_predictor(shape_predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

MATCH_THRESHOLD = 0.6

# Загружаем эталонное фото один раз при старте
def load_reference_encoding(path):
    image = Image.open(path)
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

    image_np = np.array(image)
    if image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]

    face_locations = dlib.rectangle()
    face_locations = face_recognition.face_locations(image_np)
    if not face_locations:
        raise Exception(f"❌ Лицо не найдено в эталонном фото")

    top, right, bottom, left = face_locations[0]
    dlib_rect = dlib.rectangle(left, top, right, bottom)
    shape = shape_predictor(image_np, dlib_rect)
    encoding = np.array(face_rec_model.compute_face_descriptor(image_np, shape, 1))
    return encoding

reference_encoding = load_reference_encoding("reference.jpg")

def get_face_encoding_from_image_pil(image_pil):
    # Пробуем повороты 0, 90, 180, 270, ищем ближайшее лицо
    for angle in [0, 90, 180, 270]:
        rotated = image_pil.rotate(angle, expand=True)
        image_np = np.array(rotated)
        face_locations = face_recognition.face_locations(image_np)
        if face_locations:
            # Выбираем ближайшее (максимальная площадь)
            def area(loc):
                top, right, bottom, left = loc
                return (right - left) * (bottom - top)
            largest = max(face_locations, key=area)
            top, right, bottom, left = largest
            dlib_rect = dlib.rectangle(left, top, right, bottom)
            shape = shape_predictor(image_np, dlib_rect)
            encoding = np.array(face_rec_model.compute_face_descriptor(image_np, shape, 1))
            return encoding
    return None

@app.route('/check_face', methods=['POST'])
def check_face():
    data = request.get_json()
    img_b64 = data.get("image_base64")
    if not img_b64:
        return jsonify({"error": "image_base64 is required"}), 400
    try:
        img_bytes = base64.b64decode(img_b64)
        image_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image data: {e}"}), 400

    try:
        incoming_encoding = get_face_encoding_from_image_pil(image_pil)
        if incoming_encoding is None:
            return jsonify({"result": "no_face_found"}), 200
    except Exception as e:
        return jsonify({"error": f"Error processing image: {e}"}), 500

    dist = np.linalg.norm(incoming_encoding - reference_encoding)
    matched = dist < MATCH_THRESHOLD

    return jsonify({
        "result": "match" if matched else "no_match",
        "distance": dist
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
