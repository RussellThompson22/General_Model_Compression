import tensorflow as tf
from tqdm.auto import tqdm
from tensorflow_model_optimization.sparsity.keras import prune_low_magnitude, PolynomialDecay
from tensorflow_model_optimization.quantization.keras import quantize_model
from tensorflow_model_optimization.clustering.keras import cluster_weights, CentroidInitialization

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

def train_model(model, train_dataset, val_dataset, epochs, optimizer, loss_object):
    """
    Train the model sequentially with different compression techniques:
    1. No compression
    2. Quantization
    3. Weight pruning
    4. Clustering
    """
    metrics = initialize_metrics()

    # Initial training without compression
    print("Training without compression...")
    for epoch in tqdm(range(epochs), desc="Training - No Compression"):
        train_loop(train_dataset, model, loss_object, optimizer, metrics['train'], training=True)
        train_loop(val_dataset, model, loss_object, optimizer, metrics['test'], training=False)

    # Apply quantization and retrain
    print("Training with quantization...")
    model = quantize_model(model)
    for epoch in tqdm(range(epochs), desc="Training - Quantization"):
        train_loop(train_dataset, model, loss_object, optimizer, metrics['train'], training=True)
        train_loop(val_dataset, model, loss_object, optimizer, metrics['test'], training=False)

    # Apply pruning and retrain
    print("Training with pruning...")
    pruning_params = {
        'pruning_schedule': PolynomialDecay(initial_sparsity=0.1, final_sparsity=0.5, begin_step=0, end_step=200)
    }
    model = prune_low_magnitude(model, **pruning_params)
    for epoch in tqdm(range(epochs), desc="Training - Pruning"):
        train_loop(train_dataset, model, loss_object, optimizer, metrics['train'], training=True)
        train_loop(val_dataset, model, loss_object, optimizer, metrics['test'], training=False)

    # Apply clustering and retrain
    print("Training with clustering...")
    model = cluster_weights(model, number_of_clusters=16,
                            cluster_centroids_init=CentroidInitialization.LINEAR)
    for epoch in tqdm(range(epochs), desc="Training - Clustering"):
        train_loop(train_dataset, model, loss_object, optimizer, metrics['train'], training=True)
        train_loop(val_dataset, model, loss_object, optimizer, metrics['test'], training=False)

    # Optional: strip model of pruning and clustering artifacts for final use
    model = tfmot.sparsity.keras.strip_pruning(model)
    model = tfmot.clustering.keras.strip_clustering(model)

    return model

# Example usage:
# Assuming model, train_dataset, val_dataset, optimizer, and loss_object are properly defined
# trained_model = train_model(model, train_dataset, val_dataset, 10, optimizer, loss_object)
