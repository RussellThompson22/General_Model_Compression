import tensorflow as tf
import os
import cv2
from General_Model_Compression import *

def setup_data(t_d, hp):
    """ Set up training, validation, and test datasets with batch processing. """
    return {k: v.batch(hp['batch_size']) for k, v in t_d.items()}

def get_directory(base_path):
    """ Retrieve the directory for images and XML annotations. """
    image_folder = os.path.join(base_path, 'images')
    xml_folder = os.path.join(base_path, 'annotations')
    return image_folder, xml_folder

def load_and_preprocess_image(path):
    """ Load and preprocess images using TensorFlow functions for optimal performance. """
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, [224, 224])  # Example resizing, adjust as necessary
    return img

def create_dataset(image_folder, xml_folder, max_samples=1500):
    """ Create a dataset from image and XML annotations. """
    image_files, labels = extract_labels(image_folder, xml_folder)
    image_files = image_files[:max_samples]
    labels = labels[:max_samples]
    
    dataset = tf.data.Dataset.from_tensor_slices((image_files, labels))
    dataset = dataset.map(lambda x, y: (load_and_preprocess_image(x), y))
    return dataset

def get_dataset_info(dataset):
    """ Retrieve information about the dataset such as input shape and unique labels. """
    for images, labels in dataset.take(1):
        input_shape = images.shape
        unique_labels = tf.unique(labels)
    return input_shape, unique_labels
