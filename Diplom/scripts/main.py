from myproject_data.video_loader import extract_frames
from myproject_models.keypoint_rcnn import load_model, get_keypoints
from myproject.affine import align_keypoints
from myproject.distance import weight_distance, cosine_distance
from myproject.visualization import plot_comparison_graphs
import numpy as np

def compare_videos(video_path_model, video_path_input, step=5):
    """
    Сравнивает два видео покадрово по позам.
    Строит графики метрик сходства.
    """
    model = load_model()
    frames_model = extract_frames(video_path_model, step=step)
    frames_input = extract_frames(video_path_input, step=step)

    distances, cosines = [], []
    for i in range(min(len(frames_model), len(frames_input))):
        k1, c1 = get_keypoints(frames_model[i], model, model.device)
        k2, c2 = get_keypoints(frames_input[i], model, model.device)

        if k1 is None or k2 is None or np.sum(c1) == 0:
            distances.append(None)
            cosines.append(None)
            continue

        k2_aligned = align_keypoints(k2, k1)
        dist = weight_distance(k1, k2_aligned, c1)
        cos = cosine_distance(k1, k2_aligned)

        distances.append(dist)
        cosines.append(cos)

    plot_comparison_graphs(distances, cosines)

# Пример вызова:
compare_videos("data/1.mp4", "data/2.mp4")