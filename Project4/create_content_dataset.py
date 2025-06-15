import os
import tensorflow.compat.v1 as tf
import skimage.io
import io

tf.disable_v2_behavior()

input_dir = 'data/content/images'
output_file = 'data/content/content.tfrecord'

writer = tf.python_io.TFRecordWriter(output_file)

for i, filename in enumerate(os.listdir(input_dir)):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    filepath = os.path.join(input_dir, filename)
    image = skimage.io.imread(filepath)

    if image.ndim == 2:
        # grayscale → RGB
        image = image[..., None].repeat(3, axis=2)

    buf = io.BytesIO()
    skimage.io.imsave(buf, image, format='JPEG')
    buf.seek(0)
    image_raw = buf.getvalue()

    example = tf.train.Example(features=tf.train.Features(feature={
        'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
    }))

    writer.write(example.SerializeToString())
    print(f"Added: {filename}")

writer.close()
print(f"\n TFRecord saved to: {output_file}")
