'''
Squeeze and excitation UNet


Dependencies:
    Tensorflow 2.16

'''


import tensorflow as tf
from tensorflow.keras import layers, models

tf.keras.backend.set_image_data_format('channels_last')
# input data
INPUT_SIZE = 512
INPUT_CHANNEL = 3  # 1-grayscale, 3-RGB scale
OUTPUT_MASK_CHANNEL = 1


NUM_FILTER = 32
FILTER_SIZE = 3
UP_SAMP_SIZE = 2

def dice_coef(y_true, y_pred):
    # Umwandlung von NCHW zu NHWC, falls nötig
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)

    # Sicherstellen, dass beide Tensors die gleiche Form haben
    tf.debugging.assert_shapes([
        (y_true, (None, INPUT_SIZE, INPUT_SIZE, 1)),
        (y_pred, (None, INPUT_SIZE, INPUT_SIZE, 1))
    ])
    
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1.0)


def jacard_coef(y_true, y_pred):
    tf.debugging.assert_shapes([
        (y_true, 'BHW1'), 
        (y_pred, 'BHW1')
    ])
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    return 1.0 - jacard_coef(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def squeeze_and_excitation_block(input_X, reduction_ratio=16):
    """
    SE-Block: Betont relevante Feature Maps durch globales Kontextverständnis
    :param input_X: Eingabetensor (Feature-Map)
    :param reduction_ratio: Reduktion der Kanalanzahl für den Squeeze-Schritt
    """
    channels = input_X.shape[-1]  # Anzahl der Kanäle
    squeeze = tf.reduce_mean(input_X, axis=[1, 2], keepdims=True)
    
    excitation = layers.Dense(units=channels // reduction_ratio, activation='relu')(squeeze)
    excitation = layers.Dense(units=channels, activation='sigmoid')(excitation)
    
    return input_X * excitation

def conv_block(x, filters):
    """
    Standard Convolution Block: 2x Convolution + BatchNorm + ReLU
    """
    x = layers.Conv2D(filters, (FILTER_SIZE, FILTER_SIZE), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters, (FILTER_SIZE, FILTER_SIZE), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    return x

def upsample_block(x, filters):
    """
    Upsampling Block: UpSampling2D + Convolution zur Kanalanpassung
    """
    x = layers.UpSampling2D((UP_SAMP_SIZE, UP_SAMP_SIZE))(x)
    x = layers.Conv2D(filters, (UP_SAMP_SIZE, UP_SAMP_SIZE), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    return x

def gating_signal(input_tensor, out_size):
    """
    Gating Signal: 1x1 Convolution zum Anpassen der Kanäle
    """
    gating = layers.Conv2D(out_size, kernel_size=(1, 1), padding="same")(input_tensor)
    gating = layers.BatchNormalization()(gating)
    gating = layers.Activation('relu')(gating)
    return gating

def Attention_ResUNet_SE(input_shape=( INPUT_SIZE , INPUT_SIZE , INPUT_CHANNEL )):
    inputs = tf.keras.Input(input_shape)

    # Encoder-Pfad
    c1 = conv_block(inputs, NUM_FILTER)
    p1 = layers.MaxPooling2D((UP_SAMP_SIZE, UP_SAMP_SIZE), padding="same")(c1)
    
    c2 = conv_block(p1, NUM_FILTER * 2)
    p2 = layers.MaxPooling2D((UP_SAMP_SIZE, UP_SAMP_SIZE), padding="same")(c2)
    
    c3 = conv_block(p2, NUM_FILTER * 4)
    p3 = layers.MaxPooling2D((UP_SAMP_SIZE, UP_SAMP_SIZE), padding="same")(c3)
    
    c4 = conv_block(p3, NUM_FILTER * 8)
    p4 = layers.MaxPooling2D((UP_SAMP_SIZE, UP_SAMP_SIZE), padding="same")(c4)
    
    # Bottleneck
    c5 = conv_block(p4, NUM_FILTER * 16)
    gating = gating_signal(c5, NUM_FILTER * 16)

    # Decoder-Pfad mit SE-Blocks und Skip-Connections
    u6 = upsample_block(gating, NUM_FILTER * 8)
    se6 = squeeze_and_excitation_block(u6)
    c6 = conv_block(layers.concatenate([se6, c4]), NUM_FILTER * 8)
    
    u7 = upsample_block(c6, NUM_FILTER * 4)
    se7 = squeeze_and_excitation_block(u7)
    c7 = conv_block(layers.concatenate([se7, c3]), NUM_FILTER * 4)
    
    u8 = upsample_block(c7, NUM_FILTER * 2)
    se8 = squeeze_and_excitation_block(u8)
    c8 = conv_block(layers.concatenate([se8, c2]), NUM_FILTER * 2)
    
    u9 = upsample_block(c8, NUM_FILTER)
    se9 = squeeze_and_excitation_block(u9)
    c9 = conv_block(layers.concatenate([se9, c1]), NUM_FILTER)

    # Ausgabeschicht
    conv_final = layers.Conv2D(OUTPUT_MASK_CHANNEL, (1, 1), activation="sigmoid")(c9)
    
 
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('sigmoid')(conv_final)

    model = models.Model(inputs=[inputs], outputs=conv_final , name="Attention_ResUNet_SE")
    return model

