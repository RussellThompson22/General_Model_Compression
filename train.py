import tensorflow as tf
import numpy as np
from tqdm.auto import tqdm
from General_Model_Compression.data_tools import get_dataset_info, setup_data

def reset_metrics(metrics):
    """
    Resets all passed Keras metrics.
    """
    for metric in metrics.values():
        metric.reset_states()

def update_metrics(metrics, losses, labels, predictions):
    """
    Updates the training and validation metrics based on the batch results.
    """
    metrics['train_loss'].update_state(losses)
    metrics['train_accuracy'].update_state(labels, predictions)

def run_epoch(dataset, model, metrics, loss_object, optimizer=None, data_augmentation=None, training=True):
    """
    Runs a single epoch of training or validation.
    """
    for x, y in dataset:
        if data_augmentation and training:
            x = data_augmentation(x)

        with tf.GradientTape() as tape:
            predictions = model(x, training=training)
            loss = loss_object(y, predictions)

        if training:
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        update_metrics(metrics, loss, y, predictions)

    return metrics

def train_model(train_dataset, val_dataset, model, epochs, loss_object, optimizer, metrics, data_augmentation=None, callbacks=None):
    """
    Trains the model for a given number of epochs.
    """
    for epoch in tqdm(range(epochs), desc="Training"):
        print(f"Epoch {epoch+1}/{epochs}")

        # Training phase
        train_metrics = run_epoch(train_dataset, model, metrics, loss_object, optimizer, data_augmentation, training=True)
        print(f"Training - Loss: {train_metrics['train_loss'].result():.4f}, Accuracy: {train_metrics['train_accuracy'].result() * 100:.2f}%")

        # Validation phase
        val_metrics = run_epoch(val_dataset, model, metrics, loss_object, training=False)
        print(f"Validation - Loss: {val_metrics['test_loss'].result():.4f}, Accuracy: {val_metrics['test_accuracy'].result() * 100:.2f}%")

        # Reset metrics after each epoch
        reset_metrics(metrics)

        if callbacks:
            for callback in callbacks:
                callback(model, epoch)

        if 'early_stopping' in callbacks and callbacks['early_stopping'].stop_training:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    return model

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
