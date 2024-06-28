import tensorflow as tf
import numpy as np
import glob
import os
import cv2
from sklearn.preprocessing import LabelEncoder
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

def preprocess_images(images, img_size=(128, 128)):
    """
    Preprocesses a batch of images using TensorFlow operations for optimal GPU utilization.
    Resizes images to the specified size and normalizes pixel values.
    """
    images = tf.image.resize(images, img_size)
    images = tf.cast(images, tf.float32) / 255.0  # Normalize pixel values
    return images

def encode_labels(labels):
    """
    Encodes textual labels into numeric format using scikit-learn's LabelEncoder.
    Returns both the encoded labels and the encoder instance for inverse transformations.
    """
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    return encoded_labels, encoder

def extract_labels(image_folder, xml_folder):
    """
    Extracts image file paths and labels from XML annotations.
    Assumes image and XML files are matched by filename and sorted.
    """
    image_files = sorted(glob.glob(os.path.join(image_folder, '*.bmp')))
    xml_files = sorted(glob.glob(os.path.join(xml_folder, '*.xml')))

    if len(image_files) != len(xml_files):
        raise ValueError("Mismatched number of image and XML files.")

    labels = []

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        label = root.find('.//name').text  # Using a more robust XML path
        labels.append(label)

    return image_files, labels

def plot_history(history):
    """
    Plots the training history of a model, including accuracy and loss over epochs.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history['accuracy'], label='Training Accuracy')
    ax1.plot(history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')

    ax2.plot(history['loss'], label='Training Loss')
    ax2.plot(history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def load_and_preprocess_image(file_path, img_size=(224, 224)):
    """
    Loads and preprocesses a single image from disk.
    """
    img = tf.io.read_file(file_path)
    img = tf.image.decode_bmp(img, channels=3)  # BMP-specific decoding
    img = tf.image.resize(img, img_size)
    img = img / 255.0  # Normalize to [0, 1]
    return img
