import numpy as np
import tensorflow as tf

priornet_structure = {
    'name' : 'priornet',
    'input_dim' : 40,  # class num (one-hot vector)
    'unit_num_list' : [64, 32, 16],
    'core_activation' : 'elu',
    'const_log_var' : None,
}

def priorDense(inputs, output_dim, activation='elu'):
    x = tf.keras.layers.Dense(units=output_dim, use_bias=True,
                              kernel_regularizer=tf.keras.regularizers.l2(l=0.0005))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    if activation == 'lrelu':
        x = tf.keras.layers.LeakyReLU()(x)
    elif activation == 'relu':
        x = tf.keras.layers.ReLU()(x)
    elif activation == 'elu':
        x = tf.keras.layers.ELU()(x)
    print(x.shape, 'prior dense layer')
    return x

def priornet(structure):
    name = structure['name']
    input_dim = structure['input_dim']
    unit_num_list = structure['unit_num_list']
    act = structure['core_activation']
    const_log_var = structure['const_log_var']
    if const_log_var is None:
        const_log_var = 'None'
    print('priornet', name)
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    print(inputs.shape, 'input shape')
    print('x_mean')
    x_mean = 2.0 * inputs - 1.0
    for unit_num in unit_num_list[:-1]:
        x_mean = priorDense(inputs=x_mean, output_dim=unit_num, activation=act)
    x_mean = tf.keras.layers.Dense(units=unit_num_list[-1], use_bias=True,
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.0005))(x_mean)
    print(x_mean.shape)

    if const_log_var == 'None':
        print('x_log variance')
        x_log_var = 2.0 * inputs - 1.0
        for unit_num in unit_num_list[:-1]:
            x_log_var = priorDense(inputs=x_log_var, output_dim=unit_num, activation=act)
        x_log_var = tf.keras.layers.Dense(units=unit_num_list[-1], use_bias=True,
                                          kernel_regularizer=tf.keras.regularizers.l2(l=0.0005))(x_log_var)
    elif const_log_var == const_log_var:
        print('constant log variance :', const_log_var)
        x_log_var = const_log_var * tf.ones_like(x_mean)
    else:
        print('enforce const log var to 0.0')
        x_log_var = tf.zeros_like(x_mean)

    return tf.keras.Model(inputs, (x_mean, x_log_var), name=name)
