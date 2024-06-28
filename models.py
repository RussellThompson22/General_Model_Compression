import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, concatenate, GlobalAveragePooling2D, Dense

from tensorflow.keras.applications import MobileNet, EfficientNetB0
from tensorflow.keras.utils import plot_model

def add_top_layers(base_model, num_classes):
    """
    Adds custom top layers to the base model for classification.
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)  # Optional, can be adjusted based on the model's needs.
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def mobilenet(input_shape, num_classes, weights=None):
    """
    Constructs a MobileNet model with a custom top layer for classification.
    """
    base_model = MobileNet(input_shape=input_shape, include_top=False, weights=weights)
    return add_top_layers(base_model, num_classes)

def efficientnet(input_shape, num_classes, weights=None):
    """
    Constructs an EfficientNetB0 model with a custom top layer for classification.
    """
    base_model = EfficientNetB0(input_shape=input_shape, include_top=False, weights=weights)
    return add_top_layers(base_model, num_classes)

def fire_module(x, squeeze_channels, expand_channels):
    """
    Fire module used in the SqueezeNet architecture.
    """
    squeeze = Conv2D(squeeze_channels, (1, 1), activation='relu')(x)
    expand1 = Conv2D(expand_channels, (1, 1), activation='relu')(squeeze)
    expand3 = Conv2D(expand_channels, (3, 3), padding='same', activation='relu')(squeeze)
    return concatenate([expand1, expand3], axis=-1)

def squeezenet(input_shape, num_classes):
    """
    Constructs a SqueezeNet model with specified input shape and number of classes.
    """
    input_img = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', activation='relu')(input_img)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = fire_module(x, 16, 64)
    x = fire_module(x, 16, 64)
    x = fire_module(x, 32, 128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = fire_module(x, 32, 128)
    x = fire_module(x, 48, 192)
    x = fire_module(x, 48, 192)
    x = fire_module(x, 64, 256)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = fire_module(x, 64, 256)
    x = Conv2D(num_classes, (1, 1), activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax')(x)
    model = Model(inputs=input_img, outputs=output)
    return model

def print_model_summary(model):
    """
    Prints the model summary.
    """
    model.summary()

def visualize_model(model, filename='model.png'):
    """
    Visualizes the model architecture.
    """
    plot_model(model, to_file=filename, show_shapes=True)
