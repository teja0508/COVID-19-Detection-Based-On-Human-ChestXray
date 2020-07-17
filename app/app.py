import os

import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory
import numpy as np

app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

# Load model
cnn_model = tf.keras.models.load_model(STATIC_FOLDER + "/model/" + "Corona.h5")

IMAGE_SIZE = 224


# Preprocess an image
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image /= 255.0  # normalize to [0,1] range

    return image


# Read the image from path and preprocess
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)

    return preprocess_image(image)


# Predict & classify image
def classify(model, image_path):

    preprocessed_imgage = load_and_preprocess_image(image_path)
    preprocessed_imgage = tf.reshape(
        preprocessed_imgage, (1, IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    prob = cnn_model.predict(preprocessed_imgage)
    observations = ['Covid +ve','Normal']
                 
    class_idxs_sorted = np.argsort(prob.flatten())[::-1]
    topNclass         = 2

    label = []
    classified_prob = []

    for i, idx in enumerate(class_idxs_sorted[:topNclass]):
        label.append(observations[idx])
        classified_prob.append(round(prob[0,idx]*100))

    label1 = label[0]
    #label2 = label[1]
   

    prob1 = classified_prob[0]
    #prob2 = classified_prob[1]
  


    return label1, prob1

# home page
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/classify", methods=["POST", "GET"])
def upload_file():

    if request.method == "GET":
        return render_template("home.html")

    else:
        file = request.files["myFile"]
        upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        print(upload_image_path)
        file.save(upload_image_path)

        label1, prob1 = classify(cnn_model, upload_image_path)

    return render_template(
        "classify.html", image_file_name=file.filename, label1=label1, prob1=prob1 
    )

@app.route("/classify/<filename>")
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.debug = True
    app.run(debug=True)
    app.debug = True