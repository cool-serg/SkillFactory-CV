import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from absl import app
from magenta.models.arbitrary_image_stylization import arbitrary_image_stylization_build_model as build_model

CHECKPOINT_DIR = 'checkpoints/train_dir'
EXPORT_DIR = 'saved_model'
IMAGE_SIZE = 256

def export_model(sess, export_dir, content_input, style_input, output_tensor):
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

    tensor_info_content = tf.saved_model.utils.build_tensor_info(content_input)
    tensor_info_style = tf.saved_model.utils.build_tensor_info(style_input)
    tensor_info_output = tf.saved_model.utils.build_tensor_info(output_tensor)

    signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={
            'content_image': tensor_info_content,
            'style_image': tensor_info_style
        },
        outputs={'stylized_image': tensor_info_output},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )

    builder.add_meta_graph_and_variables(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
        }
    )
    builder.save()
    print(f"✅ SavedModel exported to: {export_dir}")

def main(argv):
    tf.reset_default_graph()

    checkpoint_path = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if not checkpoint_path:
        raise FileNotFoundError("Checkpoint не найден.")

    with tf.Session() as sess:
        content_input = tf.placeholder(tf.float32, shape=[1, IMAGE_SIZE, IMAGE_SIZE, 3], name='content_input')
        style_input = tf.placeholder(tf.float32, shape=[1, IMAGE_SIZE, IMAGE_SIZE, 3], name='style_input')

        stylized_images, _, _, _ = build_model.build_model(
            content_input,
            style_input,
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
        saver.restore(sess, checkpoint_path)

        export_model(sess, EXPORT_DIR, content_input, style_input, stylized_images)

if __name__ == '__main__':
    app.run(main)
