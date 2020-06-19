import argparse
import json
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sys import argv

json_path = None

parser = argparse.ArgumentParser(description='option parser')

parser.add_argument('image_path', action="store")
parser.add_argument('saved_model_path', action="store")
parser.add_argument('--top_k', action="store",
                    dest="top_k", type=int, default=1)
parser.add_argument('--category_names', action="store",
                    dest="json_path", default=None)
args = parser.parse_args()

image_path = args.image_path
saved_model_path = args.saved_model_path
top_k = args.top_k
json_path = args.json_path

def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224 , 224))
    image /= 255
    return image

def predict(image_path, model, top_k=1):

    image = np.asarray(Image.open(image_path))
    processed_image = process_image(image)
    prediction = model.predict(np.expand_dims(processed_image, axis=0))
    probs = np.sort(prediction)[0][::-1][:top_k]
    class_probs = np.argsort(prediction)[0][::-1][:top_k]
    return probs, class_probs, image

def main(image_path, saved_model_path, top_k=1, json_path=None):
            
    model = tf.keras.models.load_model(saved_model_path, custom_objects={'KerasLayer':hub.KerasLayer})
    probs, class_probs, image = predict(image_path, model, top_k)

    if json_path:
        with open(json_path, 'r') as f:
            class_names = json.load(f)
            print('Here is the flower name:',class_names[str(class_probs[0] + 1)],', And the Probability',probs)
    else:
        print('Here is the top flower classes:',class_probs,', And the Probability',probs)
            
            
            
if __name__ == "__main__":
    main(image_path, saved_model_path, top_k, json_path)

