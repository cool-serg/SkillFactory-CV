import os
import numpy as np
from flask import Flask, request, render_template, send_file
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PIL import Image

app = Flask(__name__)
MODEL_PATH = "saved_model"
IMAGE_SIZE = 256

def load_and_process_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img).astype(np.float32)  
    return np.expand_dims(img, axis=0)  # [1, H, W, 3]

@app.route("/", methods=["GET", "POST"])
def stylize():
    if request.method == "POST":
        content_file = request.files["content"]
        style_file = request.files["style"]

        os.makedirs("uploads", exist_ok=True)
        content_path = os.path.join("uploads", "content.jpg")
        style_path = os.path.join("uploads", "style.jpg")
        content_file.save(content_path)
        style_file.save(style_path)

        content_img = load_and_process_image(content_path)
        style_img = load_and_process_image(style_path)

        tf.reset_default_graph()
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], MODEL_PATH)
            graph = tf.get_default_graph()

            
            content_input = graph.get_tensor_by_name("content_input:0")
            style_input = graph.get_tensor_by_name("style_input:0")
            output_tensor = graph.get_tensor_by_name("transformer/expand/conv3/conv/Sigmoid:0")  

            output = sess.run(output_tensor, feed_dict={
                content_input: content_img,
                style_input: style_img
            })

     
        stylized_img = np.uint8(np.clip(output[0] * 255.0, 0, 255))
        result_path = "static/stylized.jpg"
        Image.fromarray(stylized_img).save(result_path)

        return render_template("index.html", result_url=result_path)

    return render_template("index.html")

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
