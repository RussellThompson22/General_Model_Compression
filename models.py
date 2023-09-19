import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, concatenate, GlobalAveragePooling2D, Dense, layers

from tensorflow.keras.applications import MobileNet

def mobilenet(input_shape, num_classes, weights = None):

    # Load the MobileNet model without its top (final) layers
    base_model = MobileNet(input_shape=input_shape, include_top=False, weights=weights)

    # Add custom layers on top of the base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)

    # Construct the full model
    model = Model(inputs=base_model.input, outputs=x)

    return model
  

def efficientnet(input_shape, num_classes, weights = None):
    # Load the EfficientNetB0 model without the top (classification) layers
    base_model = EfficientNetB0(weights=weights, include_top=False, input_shape=input_shape)

    # Freeze the layers of the base model (optional, if you want to use it as a feature extractor)
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers on top of the base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Create the full model
    model = Model(inputs=base_model.input, outputs=predictions)

    return model


def fire_module(x, squeeze_channels, expand_channels):
    """
    Fire module for SqueezeNet.
    """
    squeeze = Conv2D(squeeze_channels, (1, 1), activation='relu')(x)
    expand1 = Conv2D(expand_channels, (1, 1), activation='relu')(squeeze)
    expand3 = Conv2D(expand_channels, (3, 3), padding='same', activation='relu')(squeeze)
    return concatenate([expand1, expand3], axis=-1)

def squeezenet(input_shape, num_classes):
    """
    SqueezeNet model.
    """
    input_img = Input(shape=input_shape)

    # Initial convolution layer
    x = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', activation='relu')(input_img)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # Fire modules
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

    # Final layers
    x = Conv2D(num_classes, (1, 1), activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax')(x)

    model = Model(inputs=input_img, outputs=output)

    return model


def channel_shuffle(x, groups):
    """Channel shuffle operation."""
    _, height, width, channels = x.shape.as_list()
    group_ch = channels // groups

    # Reshape
    x = tf.reshape(x, [-1, height, width, group_ch, groups])
    x = tf.transpose(x, [0, 1, 2, 4, 3])

    # Flatten
    x = tf.reshape(x, [-1, height, width, channels])
    return x

def shuffle_unit(x, out_channels, bottleneck_ratio, groups, stride):
    """ShuffleNet unit with strided convolution and channel shuffle."""
    bottleneck_channels = int(out_channels * bottleneck_ratio)
    in_channels = x.shape.as_list()[-1]

    if stride == 2:
        out_channels -= in_channels

    # 1x1 Group Convolution
    x = layers.Conv2D(bottleneck_channels, 1, groups=groups, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Channel Shuffle
    x = tf.keras.layers.Lambda(channel_shuffle, arguments={'groups': groups})(x)

    # 3x3 Depthwise Convolution
    x = layers.DepthwiseConv2D(3, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # 1x1 Group Convolution
    x = layers.Conv2D(out_channels, 1, groups=groups, padding='same')(x)
    x = layers.BatchNormalization()(x)

    if stride == 2:
        residual = layers.AveragePooling2D(pool_size=3, strides=2, padding='same')(x)
        x = layers.concatenate([residual, x], axis=-1)
    else:
        x = layers.add([x, residual])

    x = layers.ReLU()(x)
    return x

def shufflenet(input_shape, num_classes, scale_factor=1.0, groups=3):
    """Build ShuffleNet model."""
    input_tensor = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(24, 3, strides=2, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    out_channels = [int(144 * scale_factor), int(288 * scale_factor), int(576 * scale_factor)]
    repetitions = [4, 8, 4]

    for rep, channels in zip(repetitions, out_channels):
        for i in range(rep):
            if i == 0:
                x = shuffle_unit(x, channels, 0.25, groups, 2)
            else:
                x = shuffle_unit(x, channels, 0.25, groups, 1)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_tensor, outputs=x)
    return model



def conv_block(x, filters=16, kernel_size=3, strides=2):
    conv_layer = layers.Conv2D(
        filters, kernel_size, strides=strides, activation=tf.nn.swish, padding="same"
    )
    return conv_layer(x)


# Reference: https://git.io/JKgtC


def inverted_residual_block(x, expanded_channels, output_channels, strides=1):
    m = layers.Conv2D(expanded_channels, 1, padding="same", use_bias=False)(x)
    m = layers.BatchNormalization()(m)
    m = tf.nn.swish(m)

    if strides == 2:
        m = layers.ZeroPadding2D(padding=imagenet_utils.correct_pad(m, 3))(m)
    m = layers.DepthwiseConv2D(
        3, strides=strides, padding="same" if strides == 1 else "valid", use_bias=False
    )(m)
    m = layers.BatchNormalization()(m)
    m = tf.nn.swish(m)

    m = layers.Conv2D(output_channels, 1, padding="same", use_bias=False)(m)
    m = layers.BatchNormalization()(m)

    if tf.math.equal(x.shape[-1], output_channels) and strides == 1:
        return layers.Add()([m, x])
    return m

from transformers import TFEfficientFormerForImageClassification, EfficientFormerConfig, EfficientFormerImageProcessor

def efficientformer(input_shape, num_classes):
    # Create a configuration for the model
    config = EfficientFormerConfig()

    # Create the TFEfficientFormerModel
    efficientformer = TFEfficientFormerForImageClassification(config)
    efficientformer.classifier = tf.keras.layers.Dense(num_classes, activation='softmax', name = 'classifier')

    efficientformer.build(input_shape)

    model = tf.keras.Sequential([
        tf.keras.layers.Resizing(224, 224,
                                interpolation="bilinear",
                                crop_to_aspect_ratio=False),
        tf.keras.layers.Permute(dims=(3, 1, 2)),
        tf.keras.layers.Normalization(),
        efficientformer,
        ])

    return model

from transformers import TFMobileViTForImageClassification, MobileViTConfig, MobileViTImageProcessor

def mobilevit(input_shape, num_classes):
    # Create a configuration for the model
    config = MobileViTConfig()

    # Create the TFMobileViTModel
    MobileViT = TFMobileViTForImageClassification(config)
    MobileViT.classifier = tf.keras.layers.Dense(num_classes, activation='softmax', name = 'classifier')

    model = tf.keras.Sequential([
        tf.keras.layers.Resizing(224, 224,
                                interpolation="bilinear",
                                crop_to_aspect_ratio=False),
        tf.keras.layers.Permute(dims=(3, 1, 2)),
        tf.keras.layers.Normalization(),
        MobileViT,
        ])

    return model


