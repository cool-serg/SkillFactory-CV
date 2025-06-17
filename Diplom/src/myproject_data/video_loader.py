import cv2

def extract_frames(video_path, step=5):
    """
    Извлекает кадры из видео с заданным шагом.
    :param video_path: путь к видеофайлу
    :param step: каждый сколько кадров сохранять
    :return: список RGB-кадров
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        i += 1
    cap.release()
    return frames