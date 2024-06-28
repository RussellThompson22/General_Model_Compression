# General_Model_Compression


# TensorFlow Model Compression

This repository contains a set of tools and scripts designed to help compress deep learning models using TensorFlow. It includes utilities for pruning, quantizing, and clustering models to achieve smaller model sizes and potentially faster inference times without significant loss of accuracy.

## Features

- **Model Pruning**: Reduce the complexity of a neural network by cutting less important connections.
- **Quantization**: Reduce the precision of the numbers used to represent model parameters, which can decrease model size and improve inference performance with minimal impact on accuracy.
- **Clustering**: Group weights of the neural network into fewer distinct values, which can reduce model complexity and size.
- **Training and Evaluation Scripts**: Train and evaluate the performance of compressed models.
- **Utilities**: Functions for data preprocessing, label encoding, and training history visualization.

## Installation

To set up this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/tf-model-compression.git
   cd tf-model-compression
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Preparing Data

Use the utilities provided in `data_tools.py` to set up your datasets:

```python
from data_tools import setup_data, create_dataset

# Setup your data paths and hyperparameters
image_folder, xml_folder = get_directory('/path/to/your/data')
dataset = create_dataset(image_folder, xml_folder)
```

### Training a Model

To train a model with model compression techniques such as pruning:

```python
from train import train_model, initialize_metrics
from models import mobilenet

# Initialize model
model = mobilenet(input_shape=(224, 224, 3), num_classes=10)

# Get datasets
train_dataset, val_dataset = setup_data()

# Train the model
metrics = initialize_metrics()
trained_model = train_model(train_dataset, val_dataset, model, epochs=10, optimizer=tf.keras.optimizers.Adam(), loss_object=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics)
```

### Evaluating Model Performance

Evaluate the performance of your compressed models using the evaluation functions in `train.py`:

```python
from train import evaluate_model

# Evaluate the model
evaluate_model(val_dataset, trained_model)
```

## Contributing

Contributions to this repository are welcome. To contribute, please fork the repository, make your changes, and submit a pull request.

## License

Distributed under the MIT License. See `LICENSE` for more information.
