import numpy as np

def weight_distance(pose1, pose2, conf1):
    """Взвешенное расстояние между позами с учетом доверий к точкам"""
    p1 = pose1.flatten()
    p2 = pose2.flatten()
    weights = np.repeat(conf1, 2)
    diff = np.abs(p1 - p2)
    return float(np.sum(weights * diff) / np.sum(conf1))

def cosine_distance(pose1, pose2):
    """Косинусное сходство между позами"""
    v1 = pose1.flatten()
    v2 = pose2.flatten()
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))