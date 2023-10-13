import numpy as np
import tensorflow_model_optimization as tfmot
import wandb
from tqdm.auto import tqdm
from General_Model_Compression.data_tools import get_dataset_info, setup_data
from General_Model_Compression import *

def standardize_model_output(output):
    try:
        output = output.logits
        return output
    except:
        output
        return output

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.best = float('inf')
        self.stop_training = False

    def on_epoch_end(self, epoch, current_value):
        if self.best - current_value > self.min_delta:
            self.best = current_value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True


def train_loop(train_dataset, num_classes, model, loss_object, optimizer, data_augmentation = None):
    losses = np.array([])
    labels = np.array([]).reshape(0,)
    predictions = np.array([]).reshape(0,num_classes)

    for x, y in train_dataset:
        if data_augmentation:
            x = data_augmentation(x)

        with tf.GradientTape() as tape:
            preds = model(x, training = True)
            preds = standardize_model_output(preds)
            loss = loss_object(y, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        predictions = np.vstack((predictions, preds))
        labels = np.append(labels, y)

        losses = np.append(losses, np.array(loss))

    return losses, labels, predictions


def test_loop(test_dataset, num_classes, model, loss_object):
    losses = np.array([])
    labels = np.array([]).reshape(0,)
    predictions = np.array([]).reshape(0,num_classes)

    for x, y in test_dataset:

        preds = model(x, training = True)
        preds = standardize_model_output(preds)
        loss = loss_object(y, preds)

        predictions = np.vstack((predictions, preds))
        labels = np.append(labels, y)
        losses = np.append(losses, np.array(loss))

    return losses, labels, predictions

def prune_train(hp, t_f, t_d, model, record_results = False):
    print(f"\n\n PRUNE TRAINING MODEL {hp['model']}{'-'*100}\n")

    #Setup Data
    input_shape, num_classes = get_dataset_info(t_d['train_dataset'])
    hp['input_shape'] = input_shape
    hp['num_classes'] = num_classes

    train_dataset, val_dataset, test_dataset = setup_data(t_d, hp)
    data_augmentation = t_f['data_augmentation']

    EPOCHS = hp['epochs']
    loss_object = t_f['loss_func']()
    optimizer = t_f['optimizer'](learning_rate = hp['learning_rate'])

    # Define our metrics
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

    early_stopping = EarlyStopping(patience = hp['early_stop_patience'], min_delta = hp['early_stop_min_delta'])

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    end_step = np.ceil(len(train_dataset)).astype(np.int32) * EPOCHS

    pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                            final_sparsity=0.80,
                                                            begin_step=0,
                                                            end_step=end_step)
    }
    model = prune_low_magnitude(model, **pruning_params)

    step_callback = tfmot.sparsity.keras.UpdatePruningStep()
    step_callback.set_model(model)

    optimizer = t_f['optimizer'](learning_rate = hp['learning_rate'])


    #Training Loop
    step_callback.on_train_begin()
    model.optimizer = optimizer

    for epoch in tqdm(range(EPOCHS), desc = "Prune Training"):
        step_callback.on_epoch_begin(epoch)
        step_callback.on_train_batch_begin(batch = epoch)

        losses, labels, predictions = train_loop(train_dataset,
                                                num_classes,
                                                model,
                                                loss_object,
                                                optimizer,
                                                data_augmentation)

        train_loss(losses)
        train_accuracy(labels, predictions)

        losses, labels, predictions = test_loop(val_dataset,
                                                num_classes,
                                                model,
                                                loss_object)

        test_loss(losses)
        test_accuracy(labels, predictions)

        #display and record results
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print (template.format(epoch+1,
                                train_loss.result(),
                                train_accuracy.result()*100,
                                test_loss.result(),
                                test_accuracy.result()*100))
        logs = dict(
            loss = train_loss.result(),
            accuracy = train_accuracy.result(),
            val_loss = test_loss.result(),
            val_accuracy = test_accuracy.result(),
        )
        if record_results: wandb.log(logs)

        early_stopping.on_epoch_end(epoch, test_loss.result())

        step_callback.on_epoch_end(batch = epoch)

        if early_stopping.stop_training:
            print(f"Early stopping at epoch {epoch}")
            break

        # Reset metrics every epoch
        train_loss.reset_states()
        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()

    model = tfmot.sparsity.keras.strip_pruning(model)

    return model

def cluster_train(hp, t_f, t_d, model, record_results = False):
    print(f"\n\n CLUSTER TRAINING MODEL {hp['model']}{'-'*100}\n")

    #Setup Data
    input_shape, num_classes = get_dataset_info(t_d['train_dataset'])
    hp['input_shape'] = input_shape
    hp['num_classes'] = num_classes

    train_dataset, val_dataset, test_dataset = setup_data(t_d, hp)
    data_augmentation = t_f['data_augmentation']

    EPOCHS = hp['epochs']
    loss_object = t_f['loss_func']()
    optimizer = t_f['optimizer'](learning_rate = hp['learning_rate'])

    # Define our metrics
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

    early_stopping = EarlyStopping(patience = hp['early_stop_patience'], min_delta = hp['early_stop_min_delta'])

    cluster_weights = tfmot.clustering.keras.cluster_weights
    CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

    clustering_params = {
      'number_of_clusters': 16,
      'cluster_centroids_init': CentroidInitialization.LINEAR
    }
    model = cluster_weights(model, **clustering_params)

    optimizer = t_f['optimizer'](learning_rate = hp['learning_rate'])


    #Training Loop

    for epoch in tqdm(range(EPOCHS), desc = "Cluster Training"):

        losses, labels, predictions = train_loop(train_dataset,
                                                num_classes,
                                                model,
                                                loss_object,
                                                optimizer,
                                                data_augmentation)

        train_loss(losses)
        train_accuracy(labels, predictions)

        losses, labels, predictions = test_loop(val_dataset,
                                                num_classes,
                                                model,
                                                loss_object)

        test_loss(losses)
        test_accuracy(labels, predictions)

        #display and record results
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print (template.format(epoch+1,
                                train_loss.result(),
                                train_accuracy.result()*100,
                                test_loss.result(),
                                test_accuracy.result()*100))
        logs = dict(
            loss = train_loss.result(),
            accuracy = train_accuracy.result(),
            val_loss = test_loss.result(),
            val_accuracy = test_accuracy.result(),
        )
        if record_results: wandb.log(logs)

        early_stopping.on_epoch_end(epoch, test_loss.result())

        if early_stopping.stop_training:
            print(f"Early stopping at epoch {epoch}")
            break

        # Reset metrics every epoch
        train_loss.reset_states()
        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()

    model = tfmot.clustering.keras.strip_clustering(model)

    return model

def quantize_train(hp, t_f, t_d, model, record_results = False):
    print(f"\n\n QUANTIZE TRAINING MODEL {hp['model']}{'-'*100}\n")

    #Setup Data
    input_shape, num_classes = get_dataset_info(t_d['train_dataset'])
    hp['input_shape'] = input_shape
    hp['num_classes'] = num_classes

    train_dataset, val_dataset, test_dataset = setup_data(t_d, hp)
    data_augmentation = t_f['data_augmentation']

    EPOCHS = hp['epochs']
    loss_object = t_f['loss_func']()
    optimizer = t_f['optimizer'](learning_rate = hp['learning_rate'])

    # Define our metrics
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

    early_stopping = EarlyStopping(patience = hp['early_stop_patience'], min_delta = hp['early_stop_min_delta'])

    optimizer = t_f['optimizer'](learning_rate = hp['learning_rate'])

    quantize_model = tfmot.quantization.keras.quantize_model
    model = quantize_model(model)

    #Training Loop

    for epoch in tqdm(range(EPOCHS), desc = "Quantize Training"):

        losses, labels, predictions = train_loop(train_dataset,
                                                num_classes,
                                                model,
                                                loss_object,
                                                optimizer,
                                                data_augmentation)

        train_loss(losses)
        train_accuracy(labels, predictions)

        losses, labels, predictions = test_loop(val_dataset,
                                                num_classes,
                                                model,
                                                loss_object)

        test_loss(losses)
        test_accuracy(labels, predictions)

        #display and record results
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print (template.format(epoch+1,
                                train_loss.result(),
                                train_accuracy.result()*100,
                                test_loss.result(),
                                test_accuracy.result()*100))
        logs = dict(
            loss = train_loss.result(),
            accuracy = train_accuracy.result(),
            val_loss = test_loss.result(),
            val_accuracy = test_accuracy.result(),
        )
        if record_results: wandb.log(logs)

        early_stopping.on_epoch_end(epoch, test_loss.result())

        if early_stopping.stop_training:
            print(f"Early stopping at epoch {epoch}")
            break

        # Reset metrics every epoch
        train_loss.reset_states()
        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()

    print(f"Test accuracy: {test_accuracy.result() * 100:.2f}%")

    return model


def compress_train(hp, t_f, t_d, quantize = False, prune = False, cluster = False, record_results = False):
    print(f"\n\n TRAINING MODEL {hp['model']}{'-'*100}\n")

    #Setup Data
    input_shape, num_classes = get_dataset_info(t_d['train_dataset'])

    train_dataset, val_dataset, test_dataset = setup_data(t_d, hp)
    data_augmentation = t_f['data_augmentation']

    #Setup Model
    model = t_f['model'](input_shape, num_classes)

    EPOCHS = hp['epochs']
    loss_object = t_f['loss_func']()
    optimizer = t_f['optimizer'](learning_rate = hp['learning_rate'])

    # Define our metrics
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

    early_stopping = EarlyStopping(patience = hp['early_stop_patience'], min_delta = hp['early_stop_min_delta'])

    #Base Training
    for epoch in tqdm(range(EPOCHS), desc = "Pretraining"):

        losses, labels, predictions = train_loop(train_dataset,
                                                 num_classes,
                                                 model,
                                                 loss_object,
                                                 optimizer,
                                                 data_augmentation)

        train_loss(losses)
        train_accuracy(labels, predictions)

        losses, labels, predictions = test_loop(val_dataset,
                                                 num_classes,
                                                 model,
                                                 loss_object)

        test_loss(losses)
        test_accuracy(labels, predictions)

        #display and record results
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print (template.format(epoch+1,
                                train_loss.result(),
                                train_accuracy.result()*100,
                                test_loss.result(),
                                test_accuracy.result()*100))
        logs = dict(
            loss = train_loss.result(),
            accuracy = train_accuracy.result(),
            val_loss = test_loss.result(),
            val_accuracy = test_accuracy.result(),
        )

        if record_results: wandb.log(logs)

        early_stopping.on_epoch_end(epoch, np.mean(losses))

        if early_stopping.stop_training:
            print(f"Early stopping at epoch {epoch}")
            break

    # Reset metrics every epoch
    train_loss.reset_states()
    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()

    if prune:
        try:
            model = prune_train(hp, t_f, t_d, model, record_results = record_results)
        except Exception as error:
            print("Did not prune due to:")
            print(error)

    if cluster:
        try:
            model = cluster_train(hp, t_f, t_d, model, record_results = record_results)
        except Exception as error:
            print("Did not cluster due to:")
            print(error)

    if quantize:
        try:
            model = quantize_train(hp, t_f, t_d, model, record_results = record_results)
        except Exception as error:
            print("Did not quantize due to:")
            print(error)


    losses, labels, predictions = test_loop(test_dataset,
                                            num_classes,
                                            model,
                                            loss_object)

    test_loss(losses)
    test_accuracy(labels, predictions)

    print(f"Test accuracy: {test_accuracy.result() * 100:.2f}%")

    if record_results:
        wandb.log({"Model": wandb.Graph.from_keras(model)})
        wandb.log({"Model Summary": model.summary()})

    return model
