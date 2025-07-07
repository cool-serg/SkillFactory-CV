import cv2
import face_recognition
import numpy as np
from PIL import Image, ExifTags
import tkinter as tk
from tkinter import filedialog, messagebox
import threading

# Путь к dlib моделям — если надо, скачай и укажи пути!
import dlib
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"

shape_predictor = dlib.shape_predictor(shape_predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

RTSP_URL = "rtsp://admin:12345678@192.168.0.54:554/LowResolutionVideo"
FRAME_INTERVAL = 10
MATCH_THRESHOLD = 0.6

reference_encoding = None
reference_lock = threading.Lock()

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

    face_locations = face_recognition.face_locations(image_np)
    if not face_locations:
        raise Exception(f"❌ Лицо не найдено в {path}")

    top, right, bottom, left = face_locations[0]
    dlib_rect = dlib.rectangle(left, top, right, bottom)
    shape = shape_predictor(image_np, dlib_rect)
    encoding = np.array(face_rec_model.compute_face_descriptor(image_np, shape, 1))
    return encoding

def get_face_encodings_dlib(image_bgr, face_locations):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    encodings = []
    for (top, right, bottom, left) in face_locations:
        dlib_rect = dlib.rectangle(left, top, right, bottom)
        shape = shape_predictor(image_rgb, dlib_rect)
        encoding = np.array(face_rec_model.compute_face_descriptor(image_rgb, shape, 1))
        encodings.append(encoding)
    return encodings

class FaceCheckApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RTSP Face Check")

        # Кнопка загрузки файла
        self.load_btn = tk.Button(root, text="Загрузить фото", command=self.load_photo)
        self.load_btn.pack(pady=5)

        # Поле для отображения пути
        self.file_path_var = tk.StringVar()
        self.file_path_entry = tk.Entry(root, textvariable=self.file_path_var, width=50)
        self.file_path_entry.pack(pady=5)

        # Кнопка применения нового эталона
        self.apply_btn = tk.Button(root, text="Применить", command=self.apply_new_reference)
        self.apply_btn.pack(pady=5)

        self.status_var = tk.StringVar(value="Статус: Загрузка эталонного фото...")
        self.status_label = tk.Label(root, textvariable=self.status_var)
        self.status_label.pack(pady=5)

        self.cap = cv2.VideoCapture(RTSP_URL)
        if not self.cap.isOpened():
            messagebox.showerror("Ошибка", "Не удалось подключиться к камере")
            root.destroy()
            return

        self.frame_count = 0
        self.last_result = "WAITING..."
        self.last_color = (255, 255, 255)

        # Изначально загружаем reference.jpg
        try:
            global reference_encoding
            reference_encoding = load_reference_encoding("reference.jpg")
            self.status_var.set("Статус: Эталонное фото загружено (reference.jpg)")
        except Exception as e:
            self.status_var.set(f"Ошибка загрузки эталона: {e}")

        self.update_frame()

    def load_photo(self):
        filepath = filedialog.askopenfilename(filetypes=[("Изображения", "*.jpg *.jpeg *.png")])
        if filepath:
            self.file_path_var.set(filepath)

    def apply_new_reference(self):
        filepath = self.file_path_var.get()
        if not filepath:
            messagebox.showwarning("Внимание", "Выберите файл для загрузки")
            return
        try:
            new_encoding = load_reference_encoding(filepath)
            with reference_lock:
                global reference_encoding
                reference_encoding = new_encoding
            self.status_var.set(f"Эталонное фото обновлено: {filepath}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить фото:\n{e}")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.status_var.set("Ошибка получения кадра с камеры")
            self.root.after(100, self.update_frame)
            return

        self.frame_count += 1

        if self.frame_count % FRAME_INTERVAL == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations_small = face_recognition.face_locations(rgb_small_frame)

            face_locations = []
            for (top, right, bottom, left) in face_locations_small:
                face_locations.append((
                    top * 4,
                    right * 4,
                    bottom * 4,
                    left * 4
                ))

            face_encodings = get_face_encodings_dlib(frame, face_locations)

            with reference_lock:
                current_ref = reference_encoding

            if current_ref is None:
                self.last_result = "NO REFERENCE"
                self.last_color = (255, 255, 0)
            else:
                match_found = False
                for face_encoding in face_encodings:
                    distance = np.linalg.norm(face_encoding - current_ref)
                    if distance < MATCH_THRESHOLD:
                        match_found = True
                        self.last_result = f"MATCH ({distance:.2f})"
                        self.last_color = (0, 255, 0)
                        break

                if not match_found:
                    self.last_result = "NO MATCH"
                    self.last_color = (0, 0, 255)

            for (top, right, bottom, left) in face_locations:
                color = (0, 255, 0) if self.last_color == (0, 255, 0) else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        cv2.putText(frame, self.last_result, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.last_color, 3)

        cv2.imshow("RTSP Face Check", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            self.root.destroy()
            return

        self.root.after(10, self.update_frame)


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceCheckApp(root)
    root.mainloop()
