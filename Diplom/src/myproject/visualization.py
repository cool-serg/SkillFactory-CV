import matplotlib.pyplot as plt
import numpy as np

def plot_comparison_graphs(distances, cosines):
    valid_distances = [d for d in distances if d is not None]
    valid_cosines = [c for c in cosines if c is not None]

    mean_dist = np.mean(valid_distances)
    mean_cos = np.mean(valid_cosines)

    # Интерпретация
    if mean_dist < 20:
        dist_msg = "Позы совпадают хорошо"
    elif mean_dist < 50:
        dist_msg = "Есть заметные различия"
    else:
        dist_msg = "Сильное отклонение"

    if mean_cos > 0.98:
        cos_msg = "Форма позы точная"
    elif mean_cos > 0.95:
        cos_msg = "Поза похожа, но не идеально"
    else:
        cos_msg = "Плохое совпадение формы"

    plt.figure(figsize=(14, 6))

    # График расстояния
    plt.subplot(1, 2, 1)
    plt.plot(valid_distances, color='red')
    plt.title(f"Взвешенное расстояние\nСреднее: {mean_dist:.2f} — {dist_msg}")
    plt.xlabel("Кадр")
    plt.ylabel("Расстояние (px)")
    plt.grid(True)

    # График косинусного сходства
    plt.subplot(1, 2, 2)
    plt.plot(valid_cosines, color='green')
    plt.title(f"Косинусное сходство\nСреднее: {mean_cos:.4f} — {cos_msg}")
    plt.xlabel("Кадр")
    plt.ylabel("Сходство (1=идеал)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
