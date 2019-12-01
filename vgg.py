from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D


def vgg(input_shape, num_classes, use_larger_kernel=False):
    model = Sequential()
    kernel_size = 3
    dilation_rate = 1
    model.add(
        Conv2D(16, (kernel_size, kernel_size), padding='same', dilation_rate=dilation_rate, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    if use_larger_kernel:
        kernel_size += 2
    else:
        dilation_rate *= 2
    model.add(Conv2D(24, (kernel_size, kernel_size), padding='same', dilation_rate=dilation_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    if use_larger_kernel:
        kernel_size += 2
    else:
        dilation_rate *= 2
    model.add(Conv2D(32, (kernel_size, kernel_size), padding='same', dilation_rate=dilation_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    if use_larger_kernel:
        kernel_size += 2
    else:
        dilation_rate *= 2
    model.add(Conv2D(48, (kernel_size, kernel_size), padding='same', dilation_rate=dilation_rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model