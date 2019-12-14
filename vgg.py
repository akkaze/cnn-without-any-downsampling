import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Input, Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, DepthwiseConv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D


def vgg(input_shape, num_classes, use_larger_kernel=False, use_downsampling=False):
    inp = Input(input_shape)
    kernel_size = 3
    dilation_rate = 1
    strides = 1
    x = Conv2D(64, (kernel_size, kernel_size),
               padding='same',
               strides=strides,
               dilation_rate=dilation_rate,
               input_shape=input_shape)(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    if not use_downsampling:
        if use_larger_kernel:
            kernel_size += 2
        else:
            dilation_rate *= 2
    else:
        strides = 2
    x = Conv2D(128, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    if not use_downsampling:
        if use_larger_kernel:
            kernel_size += 2
        else:
            dilation_rate *= 2
    else:
        strides = 2
    x = Conv2D(256, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    if not use_downsampling:
        if use_larger_kernel:
            kernel_size += 2
        else:
            dilation_rate *= 2
    else:
        strides = 2
    x = Conv2D(512, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes)(x)
    out = Activation('softmax')(x)
    model = tf.keras.models.Model([inp], [out])
    return model


def vgg_3x3(input_shape, num_classes):
    inp = Input(input_shape)
    kernel_size = 3
    dilation_rate = 1
    strides = 1
    x = Conv2D(16, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate)(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(20, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(24, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(28, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(30, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(32, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(36, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(38, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(42, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(48, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes)(x)
    out = Activation('softmax')(x)
    model = tf.keras.models.Model([inp], [out])
    return model


def vgg_dep_wise_3x3(input_shape, num_classes):
    inp = Input(input_shape)
    kernel_size = 3
    dilation_rate = 1
    strides = 1
    x = Conv2D(16, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate)(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv2D((kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(24, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv2D((kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv2D((kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(32, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv2D((kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv2D((kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv2D((kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(48, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes)(x)
    out = Activation('softmax')(x)
    model = tf.keras.models.Model([inp], [out])
    return model
    return model