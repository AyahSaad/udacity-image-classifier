# Image Classifier Project

This project is part of the Udacity Deep Learning Nanodegree.  
It demonstrates how to build and train a deep learning model to classify flower images, and provides a command-line application for making predictions.

---

## Project Overview

The project has two main parts:

1. **Part 1 - Development**
   - Implemented an image classifier using TensorFlow and Keras.
   - Trained a neural network on a flower dataset.
   - Added validation and data augmentation to prevent overfitting.
   - Saved the trained model for future use.

2. **Part 2 - Command Line Application**
   - Built a Python script (`predict.py`) to predict flower classes from an image.
   - Supports top-K predictions and mapping class indices to flower names using a JSON file (`label_map.json`).
   - Command-line usage:

**Files**
my_model.h5 - The trained Keras model 

predict.py - Command line script to make predictions 

label_map.json - Mapping from class indices to flower names

test_images/ - Example images for testing

Image_Classifier_Part1.ipynb - Jupyter notebook 

**Installation**

Make sure you have Python 3 and the following packages installed:

pip install tensorflow tensorflow-hub numpy pillow

```bash
# Basic usage
python predict.py /path/to/image saved_model.h5

# Return top 3 predictions
python predict.py /path/to/image saved_model.h5 --top_k 3

# Map class indices to flower names
python predict.py /path/to/image saved_model.h5 --category_names label_map.json

Notes

Make sure my_model.h5 and label_map.json are in the same folder as predict.py.

Test images are available in the test_images/ folder.
