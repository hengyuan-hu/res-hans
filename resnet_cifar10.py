import os

from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten,
    Dropout
)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D
    #GlobalAveragePooling2D
)
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils.visualize_util import plot


# Helper to build a conv -> BN -> relu block
def _conv_bn_relu(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init="he_normal", border_mode="same")(input)
        norm = BatchNormalization(mode=0, axis=1)(conv)
        return Activation("relu")(norm)

    return f


# Helper to build a BN -> relu -> conv block
# This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        norm = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation("relu")(norm)
        return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init="he_normal", border_mode="same")(activation)

    return f


# Bottleneck architecture for > 34 layer resnet.
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
# Returns a final conv layer of nb_filters * 4
def _bottleneck(nb_filters, init_subsample=(1, 1)):
    def f(input):
        conv_1_1 = _bn_relu_conv(nb_filters, 1, 1, subsample=init_subsample)(input)
        conv_3_3 = _bn_relu_conv(nb_filters, 3, 3)(conv_1_1)
        residual = _bn_relu_conv(nb_filters * 4, 1, 1)(conv_3_3)
        return _shortcut(input, residual)

    return f


# Basic 3 X 3 convolution blocks.
# Use for resnet with layers <= 34
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
def _basic_block(nb_filters, init_subsample=(1, 1)):
    def f(input):
        conv1 = _bn_relu_conv(nb_filters, 3, 3, subsample=init_subsample)(input)
        residual = _bn_relu_conv(nb_filters, 3, 3)(conv1)
        return _shortcut(input, residual)

    return f


# Adds a shortcut between input and residual block and merges them with "sum"
def _shortcut(input, residual):
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    stride_width = input._keras_shape[2] / residual._keras_shape[2]
    stride_height = input._keras_shape[3] / residual._keras_shape[3]
    equal_channels = residual._keras_shape[1] == input._keras_shape[1]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Convolution2D(nb_filter=residual._keras_shape[1], nb_row=1, nb_col=1,
                                 subsample=(stride_width, stride_height),
                                 init="he_normal", border_mode="valid")(input)

    return merge([shortcut, residual], mode="sum")


# Builds a residual block with repeating bottleneck blocks.
def _residual_block(block_function, nb_filters, repetations, is_first_layer=False):
    def f(input):
        for i in range(repetations):
            init_subsample = (1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (2, 2)
            input = block_function(nb_filters=nb_filters, init_subsample=init_subsample)(input)
        return input

    return f


# http://arxiv.org/pdf/1512.03385v1.pdf
def resnet_cifar10(repetations):
    model_name = 'resnet_cifar10_repetation_%d' % repetations
    input = Input(shape=(3, 32, 32))

    conv1 = _conv_bn_relu(nb_filter=64, nb_row=3, nb_col=3)(input)
    # feature map size (32, 32)
    # pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="same")(conv1)

    # Build residual blocks..
    block_fn = _basic_block
    block1 = _residual_block(block_fn, nb_filters=16, repetations=repetations)(conv1)
    # feature map size (16, 16)
    block2 = _residual_block(block_fn, nb_filters=32, repetations=repetations)(block1)
    # feature map size (8, 8)
    block3 = _residual_block(block_fn, nb_filters=64, repetations=repetations)(block2)
    # block4 = _residual_block(block_fn, nb_filters=512, repetations=3)(block3)

    # Classifier block
    # pool2 = AveragePooling2D(pool_size=(4, 4), border_mode="same")(block3)
    pool2 = GlobalAveragePooling2D()(block3)
    # flatten1 = Flatten()(pool2)
    dense = Dense(output_dim=10, init="he_normal", activation="softmax")(pool2)

    model = Model(input=input, output=dense)
    return model, model_name


def main():
    import time
    start = time.time()
    model, model_name = resnet_cifar10(repetations=2)
    duration = time.time() - start
    print "{} s to make model".format(duration)

    start = time.time()
    model.output
    duration = time.time() - start
    print "{} s to get output".format(duration)

    start = time.time()
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    duration = time.time() - start
    print "{} s to get compile".format(duration)

    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(current_dir, "resnet_cifar10.png")
    plot(model, to_file=model_path, show_shapes=True)


if __name__ == '__main__':
    main()
