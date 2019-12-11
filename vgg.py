from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D


def vgg(input_shape, num_classes, use_larger_kernel=False, use_downsampling=False):
    model = Sequential()
    kernel_size = 3
    dilation_rate = 1
    strides = 1
    model.add(
        Conv2D(64, (kernel_size, kernel_size),
               padding='same',
               strides=strides,
               dilation_rate=dilation_rate,
               input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    if not use_downsampling:
        if use_larger_kernel:
            kernel_size += 2
        else:
            dilation_rate *= 2
    else:
        strides = 2
    model.add(Conv2D(128, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    if not use_downsampling:
        if use_larger_kernel:
            kernel_size += 2
        else:
            dilation_rate *= 2
    else:
        strides = 2
    model.add(Conv2D(256, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    if not use_downsampling:
        if use_larger_kernel:
            kernel_size += 2
        else:
            dilation_rate *= 2
    else:
        strides = 2
    model.add(Conv2D(512, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model


def vgg_3x3(input_shape, num_classes):
    model = Sequential()
    kernel_size = 3
    dilation_rate = 1
    strides = 1
    model.add(
        Conv2D(16, (kernel_size, kernel_size),
               padding='same',
               strides=strides,
               dilation_rate=dilation_rate,
               input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(16, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(16, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(16, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(16, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(18, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(18, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(20, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(20, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model


def vgg_3x3_v2(input_shape, num_classes):
    model = Sequential()
    kernel_size = 3
    dilation_rate = 1
    strides = 1
    model.add(
        Conv2D(16, (kernel_size, kernel_size),
               padding='same',
               strides=strides,
               dilation_rate=dilation_rate,
               input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D((kernel_size, kernel_size), padding='same', strides=strides,
                              dilation_rate=dilation_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(24, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D((kernel_size, kernel_size), padding='same', strides=strides,
                              dilation_rate=dilation_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D((kernel_size, kernel_size), padding='same', strides=strides,
                              dilation_rate=dilation_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(36, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D((kernel_size, kernel_size), padding='same', strides=strides,
                              dilation_rate=dilation_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D((kernel_size, kernel_size), padding='same', strides=strides,
                              dilation_rate=dilation_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D((kernel_size, kernel_size), padding='same', strides=strides,
                              dilation_rate=dilation_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(48, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model


def vgg_3x3_v3(input_shape, num_classes):
    model = Sequential()
    kernel_size = 3
    dilation_rate = 1
    strides = 1
    model.add(
        Conv2D(16, (kernel_size, kernel_size),
               padding='same',
               strides=strides,
               dilation_rate=dilation_rate,
               input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D((kernel_size, kernel_size), padding='same', strides=strides,
                              dilation_rate=dilation_rate))
    model.add(Conv2D(16, (1, 1), padding='same', strides=strides, dilation_rate=dilation_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(24, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D((kernel_size, kernel_size), padding='same', strides=strides,
                              dilation_rate=dilation_rate))
    model.add(Conv2D(24, (1, 1), padding='same', strides=strides, dilation_rate=dilation_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D((kernel_size, kernel_size), padding='same', strides=strides,
                              dilation_rate=dilation_rate))
    model.add(Conv2D(32, (1, 1), padding='same', strides=strides, dilation_rate=dilation_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(32, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D((kernel_size, kernel_size), padding='same', strides=strides,
                              dilation_rate=dilation_rate))
    model.add(Conv2D(32, (1, 1), padding='same', strides=strides, dilation_rate=dilation_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D((kernel_size, kernel_size), padding='same', strides=strides,
                              dilation_rate=dilation_rate))
    model.add(Conv2D(32, (1, 1), padding='same', strides=strides, dilation_rate=dilation_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(DepthwiseConv2D((kernel_size, kernel_size), padding='same', strides=strides,
                              dilation_rate=dilation_rate))
    model.add(Conv2D(32, (1, 1), padding='same', strides=strides, dilation_rate=dilation_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(48, (kernel_size, kernel_size), padding='same', strides=strides, dilation_rate=dilation_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model