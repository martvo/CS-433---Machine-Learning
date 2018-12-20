import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPool2D
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.backend import resize_images


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True, dilation_rate=1):
    """
    This function creates a the "layers" for the unet. Connects the different layers to the input_tensor and returns the added layers
    If batchnorm = True this code will add a barchnormalization layer after the conv2d layer
    """
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same", dilation_rate=dilation_rate)(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same", dilation_rate=dilation_rate)(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def create_model(input_img, n_filters=16, dropout=0.5, batchnorm=True, kernel_size=3, dilation_rate=1):
    # contracting path of the model
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=kernel_size, batchnorm=batchnorm, dilation_rate=dilation_rate)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout * 0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # expansive path of the model
    u6 = Conv2DTranspose(n_filters * 8, (kernel_size, kernel_size), strides=(2, 2), padding='same', dilation_rate=dilation_rate)(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=kernel_size, batchnorm=batchnorm, dilation_rate=dilation_rate)

    u7 = Conv2DTranspose(n_filters * 4, (kernel_size, kernel_size), strides=(2, 2), padding='same', dilation_rate=dilation_rate)(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=kernel_size, batchnorm=batchnorm, dilation_rate=dilation_rate)

    u8 = Conv2DTranspose(n_filters * 2, (kernel_size, kernel_size), strides=(2, 2), padding='same', dilation_rate=dilation_rate)(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=kernel_size, batchnorm=batchnorm, dilation_rate=dilation_rate)

    u9 = Conv2DTranspose(n_filters * 1, (kernel_size, kernel_size), strides=(2, 2), padding='same', dilation_rate=dilation_rate)(c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=kernel_size, batchnorm=batchnorm, dilation_rate=dilation_rate)
    
    # Out put layer, uses sigmoid to get the pixels in a range of [0, 1]
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=input_img, outputs=outputs)
    return model
