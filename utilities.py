import tensorflow as tf

def preprocess_images(images, img_size=(128, 128)):
    preprocessed_images = [cv2.resize(img, img_size) for img in images]
    return np.array(preprocessed_images)

def encode_labels(labels):
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    return encoded_labels, encoder.classes_

def extract_labels(image_folder, xml_folder):
    image_files = sorted(glob.glob(os.path.join(image_folder, '*.bmp')))
    xml_files = sorted(glob.glob(os.path.join(xml_folder, '*.xml')))

    print(f"Found {len(image_files)} image files and {len(xml_files)} XML files.")

    labels = []

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        label = root.find('object').find('name').text
        labels.append(label)

    return image_files, labels

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot training and validation accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Plot training and validation loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.show()

def get_dataset_info(dataset):
    unique_labels = set()
    for _, labels in dataset:
        unique_labels.update([labels.numpy()])

    # Assuming dataset is batched, take one batch to get input shape
    for images, _ in dataset.take(1):
        input_shape = images.shape

    return input_shape, len(unique_labels)


