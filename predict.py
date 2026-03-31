import argparse
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image


def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0
    return image.numpy()


def predict(image_path, model, top_k=1):
    im = Image.open(image_path)
    image = np.asarray(im)
    processed_image = process_image(image)
    expanded_image = np.expand_dims(processed_image, axis=0)
    
    predictions = model.predict(expanded_image)
    probs, classes = tf.math.top_k(predictions[0], k=top_k)
    
    return probs.numpy(), classes.numpy()

def main():
    parser = argparse.ArgumentParser(description='Predict flower class from image')
    
    parser.add_argument('image_path', type=str, help='Path to image')
    parser.add_argument('saved_model', type=str, help='Path to saved Keras model')
    parser.add_argument('--top_k', type=int, default=1, help='Return top K classes')
    parser.add_argument('--category_names', type=str,default='label_map.json',help='Path to JSON label map')
    
    args = parser.parse_args()
    
    # load the model
    model = tf.keras.models.load_model(
        args.saved_model,
        custom_objects={'KerasLayer': hub.KerasLayer}
    )
    
    # predict
    probs, classes = predict(args.image_path, model, args.top_k)
    
    if args.category_names:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
        labels = [class_names[str(c)] for c in classes]
    else:
        labels = classes
    
    # results
    for label, prob in zip(labels, probs):
        print(f'{label}: {prob:.4f}')


if __name__ == '__main__':
    main()




