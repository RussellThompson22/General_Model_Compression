import tensorflow as tf
import numpy as np
from tqdm.auto import tqdm
from tensorflow_model_optimization.sparsity.keras import prune_low_magnitude
from tensorflow_model_optimization.sparsity.keras import PolynomialDecay
from tensorflow_model_optimization.quantization.keras import quantize_model
from tensorflow_model_optimization.clustering.keras import cluster_weights
from tensorflow_model_optimization.clustering.keras import CentroidInitialization

def initialize_metrics():
    """
    Initializes and returns common metrics used during training and validation.
    """
    return {
        'train_loss': tf.keras.metrics.Mean(name='train_loss'),
        'train_accuracy': tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy'),
        'test_loss': tf.keras.metrics.Mean(name='test_loss'),
        'test_accuracy': tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    }

def train_loop(dataset, model, loss_object, optimizer, metrics, training=True):
    """
    Generic training/validation loop to be used for either training or validation.
    """
    for x, y in dataset:
        with tf.GradientTape() as tape:
            predictions = model(x, training=training)
            loss = loss_object(y, predictions)

        if training:
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        metrics['train_loss'].update_state(loss)
        metrics['train_accuracy'].update_state(y, predictions)

def apply_pruning(model, initial_sparsity, final_sparsity, begin_step, end_step):
    """
    Apply pruning to the model.
    """
    pruning_params = {
        'pruning_schedule': PolynomialDecay(initial_sparsity=initial_sparsity,
                                            final_sparsity=final_sparsity,
                                            begin_step=begin_step,
                                            end_step=end_step)
    }
    return prune_low_magnitude(model, **pruning_params)

def apply_quantization(model):
    """
    Apply quantization to the model.
    """
    return quantize_model(model)

def apply_clustering(model, number_of_clusters):
    """
    Apply clustering to the model.
    """
    return cluster_weights(model, number_of_clusters=number_of_clusters,
                           cluster_centroids_init=CentroidInitialization.LINEAR)

def train_model(model, train_dataset, val_dataset, epochs, optimizer, loss_object, apply_compression=None, compression_params=None):
    """
    Train the model with optional compression techniques.
    """
    metrics = initialize_metrics()

    if apply_compression:
        model = apply_compression(model, **compression_params)

    for epoch in tqdm(range(epochs), desc="Training"):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Training phase
        train_loop(train_dataset, model, loss_object, optimizer, metrics['train'], training=True)

        # Validation phase
        train_loop(val_dataset, model, loss_object, optimizer, metrics['test'], training=False)

        print(f"Train Loss: {metrics['train_loss'].result()}, Train Accuracy: {metrics['train_accuracy'].result()}")
        print(f"Validation Loss: {metrics['test_loss'].result()}, Validation Accuracy: {metrics['test_accuracy'].result()}")

        # Reset metrics at the end of epoch
        for metric in metrics.values():
            metric.reset_states()

    if hasattr(model, 'strip_clustering'):
        model = model.strip_clustering()

    return model

# Example usage:
# model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=True, classes=10)
# optimizer = tf.keras.optimizers.Adam()
# loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
# trained_model = train_model(model, train_dataset, val_dataset, 10, optimizer, loss_object,
#                             apply_compression=apply_pruning, compression_params={'initial_sparsity': 0.1, 'final_sparsity': 0.5, 'begin_step': 0, 'end_step': 1000})
