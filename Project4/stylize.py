import tensorflow.compat.v1 as tf
import tf_slim as slim
import numpy as np
import skimage.io
from skimage.transform import resize
import os
import tempfile

from magenta.models.image_stylization import image_utils
from magenta.models.arbitrary_image_stylization import arbitrary_image_stylization_build_model as build_model

# ⚙️ Временная директория
tempfile.tempdir = "D:/SF-Projects/computervision/project4/tmp"
os.makedirs(tempfile.tempdir, exist_ok=True)

# 🛠 Windows-safe временные файлы
_original_named_tempfile = tempfile.NamedTemporaryFile
def safe_named_tempfile(*args, **kwargs):
    kwargs["delete"] = False
    return _original_named_tempfile(*args, **kwargs)
tempfile.NamedTemporaryFile = safe_named_tempfile

tf.disable_v2_behavior()

# Пути к файлам
CONTENT_IMAGE = "content.jpg"
STYLE_IMAGE = "style.jpg"
CHECKPOINT_DIR = "checkpoints/train_dir"
OUTPUT_IMAGE = "stylized_output.jpg"
IMAGE_SIZE = 256

def main():
    with tf.Graph().as_default(), tf.Session() as sess:
        # Загрузка изображений и масштабирование
        content_img = image_utils.load_np_image(CONTENT_IMAGE)
        style_img = image_utils.load_np_image(STYLE_IMAGE)

        content_img = resize(content_img, (IMAGE_SIZE, IMAGE_SIZE), preserve_range=True).astype(np.float32)
        style_img = resize(style_img, (IMAGE_SIZE, IMAGE_SIZE), preserve_range=True).astype(np.float32)

        content_input = tf.placeholder(tf.float32, shape=content_img.shape, name='content_input')
        style_input = tf.placeholder(tf.float32, shape=style_img.shape, name='style_input')

        content_batch = tf.expand_dims(content_input, 0)
        style_batch = tf.expand_dims(style_input, 0)

        stylized_images, _, _, _ = build_model.build_model(
            content_batch,
            style_batch,
            trainable=False,
            is_training=False,
            inception_end_point='Mixed_6e',
            style_prediction_bottleneck=100,
            content_weights={"vgg_16/conv3": 1.0},
            style_weights={
                "vgg_16/conv1": 0.5e-3,
                "vgg_16/conv2": 0.5e-3,
                "vgg_16/conv3": 0.5e-3,
                "vgg_16/conv4": 0.5e-3
            },
            total_variation_weight=1e4
        )

        saver = tf.train.Saver()
        checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
        if checkpoint is None:
            raise ValueError("❌ Checkpoint не найден в папке: " + CHECKPOINT_DIR)
        saver.restore(sess, checkpoint)

        result = sess.run(stylized_images, feed_dict={
            content_input: content_img,
            style_input: style_img
        })

        output_img = result
        if output_img.ndim == 4:
            output_img = output_img[0]
        if output_img.shape[-1] == 1:
            output_img = np.repeat(output_img, 3, axis=-1)

        from PIL import Image
        
        print("📊 Output image stats:")
        print("Shape:", output_img.shape)
        print("Min:", output_img.min())
        print("Max:", output_img.max())
        print("Mean:", output_img.mean())

       
        import matplotlib.pyplot as plt

        plt.imshow(np.clip(output_img, 0, 1))
        plt.title("Preview: Stylized Image")
        plt.axis("off")
        plt.show()

        
        
        output_img_uint8 = np.uint8(np.clip(output_img * 255.0, 0, 255))
        Image.fromarray(output_img_uint8).save(OUTPUT_IMAGE)

        print(f"✅ Stylized image saved to {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()
