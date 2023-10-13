import tensorflow as tf
import socket
import os
import cv2
from General_Model_Compression import *

def setup_data(t_d, hp):
    train_dataset = t_d['train_dataset'].batch(hp['batch_size'])
    val_dataset = t_d['val_dataset'].batch(hp['batch_size'])
    test_dataset = t_d['test_dataset'].batch(hp['batch_size'])

    return train_dataset, val_dataset, test_dataset

def get_local_directory():
    print("Please input the path of the image folder in your Google Drive:")
    image_folder = '/content/drive/MyDrive/UMASSD/Model Compression/UATD/UATD_Training/images'
    print("Please input the path of the XML folder in your Google Drive:")
    xml_folder = '/content/drive/MyDrive/UMASSD/Model Compression/UATD/UATD_Training/annotations'
    return image_folder, xml_folder

def create_training_dataset(image_folder, xml_folder, max_samples=1500):
    image_files, labels = extract_labels(image_folder, xml_folder)

    # Limit the number of samples
    image_files = image_files[:max_samples]
    labels = labels[:max_samples]

    dataset = []

    for image_file, label in zip(image_files, labels):
        img = cv2.imread(image_file)
        dataset.append((img, label))

    return dataset

def create_test_dataset(image_folder, xml_folder):
    image_files, labels = extract_labels(image_folder, xml_folder)

    dataset = []

    for image_file, label in zip(image_files, labels):
        img = cv2.imread(image_file)
        dataset.append((img, label))

    return dataset

def get_dataset_info(dataset):
    unique_labels = set()
    for _, labels in dataset:
        unique_labels.update([labels.numpy()])

    # Assuming dataset is batched, take one batch to get input shape
    for images, _ in dataset.take(1):
        input_shape = images.shape

    return input_shape, len(unique_labels)

# def create_datasets():
#     # Create the training dataset
#     image_folder, xml_folder = get_local_directory()
#     training_data = create_training_dataset(image_folder, xml_folder, max_samples=1500)
#     print("Training dataset created.")

#     # Split the data into images and labels
#     images, labels = zip(*training_data)
#     images = preprocess_images(images)
#     labels, classes = encode_labels(labels)

#     # Split the data into training and validation sets
#     X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
#     tr_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
#     v_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

#     # Load the test dataset
#     test_image_folder = "/content/drive/MyDrive/UMASSD/Model Compression/UATD/UATD_Test_1/images"
#     test_xml_folder = "/content/drive/MyDrive/UMASSD/Model Compression/UATD/UATD_Test_1/annotations"
#     test_data = create_test_dataset(test_image_folder, test_xml_folder)
#     print("Test dataset loaded.")

#     # Split the test data into images and labels, preprocess the images, and encode the labels
#     test_images, test_labels = zip(*test_data)
#     test_images = preprocess_images(test_images)
#     test_labels_encoded = encode_labels(test_labels)[0]  # Only need the encoded labels
#     te_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels_encoded))

#     save_dir = '/content/drive/MyDrive/UMASSD/Model Compression/UATD/UATD_Training'
#     te_dataset.save(os.path.join(save_dir, "uatd_train_dataset"))
#     v_dataset.save(os.path.join(save_dir, "uatd_val_dataset"))
#     tr_dataset.save(os.path.join(save_dir, "uatd_test_dataset"))

#     return tr_dataset, v_dataset, te_dataset

def get_datasets():
    host_name = socket.gethostname()
    if host_name =='jetson-desktop':
        save_dir = './Documents/Data/UATD'
    elif os.path.exists('/content/drive/MyDrive/Model Compression/UATD/UATD_Training'):
        save_dir = '/content/drive/MyDrive/Model Compression/UATD/UATD_Training'
    elif os.path.exists('/content/drive/MyDrive/UMASSD/Model Compression/UATD/UATD_Training'):
        save_dir = '/content/drive/MyDrive/UMASSD/Model Compression/UATD/UATD_Training'
    else:
        raise ValueError("UNKNOWN SAVE DIRECTORY")

    te_dataset = tf.data.Dataset.load(os.path.join(save_dir, "uatd_train_dataset"))
    v_dataset = tf.data.Dataset.load(os.path.join(save_dir, "uatd_val_dataset"))
    tr_dataset = tf.data.Dataset.load(os.path.join(save_dir, "uatd_test_dataset"))

    return tr_dataset, v_dataset, te_dataset



def setup_data(t_d, hp):
    train_dataset = t_d['train_dataset'].batch(hp['batch_size'])
    val_dataset = t_d['val_dataset'].batch(hp['batch_size'])
    test_dataset = t_d['test_dataset'].batch(hp['batch_size'])

    return train_dataset, val_dataset, test_dataset

