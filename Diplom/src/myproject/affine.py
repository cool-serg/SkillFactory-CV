import numpy as np

def pad(x):
    return np.hstack([x, np.ones((x.shape[0], 1))])

def unpad(x):
    return x[:, :-1]

def align_keypoints(source, target):
    """Выполняет аффинное преобразование source под target"""
    A, _, _, _ = np.linalg.lstsq(pad(source), pad(target), rcond=None)
    return unpad(np.dot(pad(source), A))