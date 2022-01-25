import numpy as np
import tensorflow as tf
# print(tf.test.is_gpu_available())

# @tf.function
def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * tf.abs(x)

def Darknet53Conv(inputs, filter_num, filter_size, strides=1, batch_norm=True, activation='elu'):
    if strides == 1:
        x = inputs
        padding = 'same'
    else:
        x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(inputs)
        padding = 'valid'
    x = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=filter_size, strides=strides,
                               padding=padding, use_bias= not batch_norm,
                               kernel_regularizer=tf.keras.regularizers.l2(l=0.0005))(x)
    print(x.shape, 'Darknet53conv2D')
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
        if activation == 'lrelu':
            x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        elif activation == 'elu':
            x = tf.keras.layers.ELU()(x)
        elif activation == 'relu':
            x = tf.keras.layers.ReLU()(x)
        print(x.shape, 'batch_norm')
    return x

def Darknet53Residual(inputs, filter_num, activation='elu'):
    x = Darknet53Conv(inputs=inputs, filter_num=filter_num//2, filter_size=1, activation=activation)  # divide filterNum by 2
    x = Darknet53Conv(x, filter_num=filter_num, filter_size=3)
    x = inputs + x
    print(x.shape, 'residual')
    return x

def Darknet53ConvResBlock(inputs, filter_num, block_num, activation='elu'):
    x = Darknet53Conv(inputs=inputs, filter_num=filter_num, filter_size=3, strides=2, activation=activation)
    for _ in range(block_num):
        x = Darknet53Residual(x, filter_num=filter_num, activation=activation)
    return x

def Darknet53(name=None, activation='elu'):
    print('Darknet53', name)
    x = inputs = tf.keras.layers.Input([None, None, 3])
    x = Darknet53Conv(inputs=x, filter_num=32, filter_size=3, activation=activation)
    x = Darknet53ConvResBlock(inputs=x, filter_num=64, block_num=1, activation=activation)
    x = Darknet53ConvResBlock(inputs=x, filter_num=128, block_num=2, activation=activation)  # skip connection
    x = x_36 = Darknet53ConvResBlock(inputs=x, filter_num=256, block_num=8, activation=activation)
    x = x_61 = Darknet53ConvResBlock(inputs=x, filter_num=512, block_num=8, activation=activation)
    x = Darknet53ConvResBlock(inputs=x, filter_num=1024, block_num=4, activation=activation)
    print('end Darknet53')
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)

def Darknet53Tiny(name=None, activation='elu'):
    print('Darknet53Tiny', name)
    x = inputs = tf.keras.layers.Input([None, None, 3])
    x = Darknet53Conv(inputs=x, filter_num=16, filter_size=3, activation=activation)
    x = tf.keras.layers.MaxPool2D(2, 2, 'same')(x)
    print(x.shape, 'maxPool')
    x = Darknet53Conv(inputs=x, filter_num=32, filter_size=3, activation=activation)
    x = tf.keras.layers.MaxPool2D(2, 2, 'same')(x)
    print(x.shape, 'maxPool')
    x = Darknet53Conv(inputs=x, filter_num=64, filter_size=3, activation=activation)
    x = tf.keras.layers.MaxPool2D(2, 2, 'same')(x)
    print(x.shape, 'maxPool')
    x = Darknet53Conv(inputs=x, filter_num=128, filter_size=3, activation=activation)
    x = tf.keras.layers.MaxPool2D(2, 2, 'same')(x)
    print(x.shape, 'maxPool')
    x = x_8 = Darknet53Conv(inputs=x, filter_num=256, filter_size=3, activation=activation)
    x = tf.keras.layers.MaxPool2D(2, 2, 'same')(x)
    print(x.shape, 'maxPool')
    x = Darknet53Conv(inputs=x, filter_num=512, filter_size=3, activation=activation)
    x = tf.keras.layers.MaxPool2D(2, 1, 'same')(x)
    print(x.shape, 'maxPool')
    x = Darknet53Conv(inputs=x, filter_num=1024, filter_size=3, activation=activation)
    print('end Darknet53Tiny')
    return tf.keras.Model(inputs, (x_8, x), name=name)

def Darknet19Conv(inputs, filter_num, filter_size, activation):
    x = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=filter_size, strides=1, padding='same',
                               activation=None, use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    if activation=='lrelu':
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    elif activation == 'elu':
        x = tf.keras.layers.ELU()(x)
    elif activation == 'relu':
        x = tf.keras.layers.ReLU()(x)
    print(x.shape, 'darknet19Conv2D')
    return x

def Darknet19(name=None, activation='elu'):
    print('Darknet19', name)
    x = inputs = tf.keras.layers.Input(shape=[None, None, 3])
    x = Darknet19Conv(inputs=x, filter_num=32, filter_size=3, activation=activation)
    x = tf.keras.layers.MaxPool2D(2, 2, 'same')(x)
    print(x.shape, 'maxPool')

    x = Darknet19Conv(inputs=x, filter_num=64, filter_size=3, activation=activation)
    x = tf.keras.layers.MaxPool2D(2, 2, 'same')(x)
    print(x.shape, 'maxPool')

    x = Darknet19Conv(inputs=x, filter_num=128, filter_size=3, activation=activation)
    x = Darknet19Conv(inputs=x, filter_num=64, filter_size=1, activation=activation)
    x = Darknet19Conv(inputs=x, filter_num=128, filter_size=3, activation=activation)
    x = tf.keras.layers.MaxPool2D(2, 2, 'same')(x)
    print(x.shape, 'maxPool')

    x = Darknet19Conv(inputs=x, filter_num=256, filter_size=3, activation=activation)
    x = Darknet19Conv(inputs=x, filter_num=128, filter_size=1, activation=activation)
    x = Darknet19Conv(inputs=x, filter_num=256, filter_size=3, activation=activation)
    x = tf.keras.layers.MaxPool2D(2, 2, 'same')(x)
    print(x.shape, 'maxPool')

    x = Darknet19Conv(inputs=x, filter_num=512, filter_size=3, activation=activation)
    x = Darknet19Conv(inputs=x, filter_num=256, filter_size=1, activation=activation)
    x = Darknet19Conv(inputs=x, filter_num=512, filter_size=3, activation=activation)
    x = Darknet19Conv(inputs=x, filter_num=256, filter_size=1, activation=activation)
    x = Darknet19Conv(inputs=x, filter_num=512, filter_size=3, activation=activation)
    x = tf.keras.layers.MaxPool2D(2, 2, 'same')(x)
    print(x.shape, 'maxPool')

    x = Darknet19Conv(inputs=x, filter_num=1024, filter_size=3, activation=activation)
    x = Darknet19Conv(inputs=x, filter_num=512, filter_size=1, activation=activation)
    x = Darknet19Conv(inputs=x, filter_num=1024, filter_size=3, activation=activation)
    x = Darknet19Conv(inputs=x, filter_num=512, filter_size=1, activation=activation)
    x = Darknet19Conv(inputs=x, filter_num=1024, filter_size=3, activation=activation)
    print('end Darknet19')
    return tf.keras.Model(inputs, x, name=name)

def convHead(inputs, filter_num, filter_size, act):
    x = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=filter_size,
                               strides=1, padding='same', activation=None, use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(l=0.0005))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    if act == 'lrelu':
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    elif act == 'elu':
        x = tf.keras.layers.ELU()(x)
    elif act == 'relu':
        x = tf.keras.layers.ReLU()(x)
    print(x.shape, 'conv2D head')
    return x

def head2D(name, input_shape, output_dim,
                   filter_num_list, filter_size_list, last_pooling=None, activation='elu'):
    print('head start')
    x = inputs = tf.keras.layers.Input(shape=input_shape)
    for filter_num, filter_size in zip(filter_num_list, filter_size_list):
        x = convHead(inputs=x, filter_num=filter_num, filter_size=filter_size,act=activation)
    x = tf.keras.layers.Conv2D(filters=output_dim, kernel_size=1,
                               strides=1, padding='same', activation=None, use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(l=0.0005))(x)
    print(x.shape, 'lastConv2D')
    if last_pooling == 'max':
        x = tf.reduce_max(input_tensor=x, axis=[1, 2])
        print(x.shape, 'maxpool')
    elif last_pooling == 'average':
        x = tf.reduce_mean(input_tensor=x, axis=[1, 2])
        print(x.shape, 'average pool')
    else:
        pass
    print('end head2D')
    return tf.keras.Model(inputs, x, name=name)





