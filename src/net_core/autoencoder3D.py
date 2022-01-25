import numpy as np
import tensorflow as tf

#=========== autoencoder architecture example (from 3D GAN) ===============
encoder_structure = {
    'name':'encoder',
    'input_shape':[64,64,64,1], # or [None,None,None,1]
    'filter_num_list':[64,128,256,512,400],
    'filter_size_list':[4,4,4,4,4],
    'strides_list':[2,2,2,2,1],
    'final_pool':'average',
    'activation':'elu',
    'final_activation':'None',
}
decoder_structure = {
    'name':'docoder',
    'input_dim' : 200,
    'output_shape':[64,64,64,1],
    'filter_num_list':[512,256,128,64,1],
    'filter_size_list':[4,4,4,4,4],
    'strides_list':[1,2,2,2,2],
    'activation':'elu',
    'final_activation':'sigmoid'
}

def conv3DEnc(inputs, filter_num, filter_size, strides=2, padding='same', activation='elu'):
    x = tf.keras.layers.Conv3D(filters=filter_num, kernel_size=filter_size,
                               strides=strides, padding=padding, activation=None, use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(l=0.0005)
                               )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    print(x.shape, 'conv3DEnc')
    if activation == 'lrelu':
        x = tf.keras.layers.LeakyReLU()(x)
    elif activation == 'relu':
        x = tf.keras.layers.ReLU()(x)
    elif activation == 'elu':
        x = tf.keras.layers.ELU()(x)
    return x

def conv3DDec(inputs, filter_num, filter_size, strides=2, padding='same', activation='elu'):
    x = tf.keras.layers.Conv3DTranspose(filters=filter_num, kernel_size=filter_size,
                                        strides=strides, padding=padding, activation=None, use_bias=False,
                                        kernel_regularizer=tf.keras.regularizers.l2(l=0.0005)
                                        )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    print(x.shape, 'conv3DDec')
    if activation == 'lrelu':
        x = tf.keras.layers.LeakyReLU()(x)
    elif activation == 'relu':
        x = tf.keras.layers.ReLU()(x)
    elif activation == 'elu':
        x = tf.keras.layers.ELU()(x)
    return x

def linearTransform(inputs, output_dim, activation='None'):
    # print(inputs.get_shape().as_list())
    x = tf.reshape(inputs, (-1, np.prod(inputs.get_shape().as_list()[1:])))
    x = tf.keras.layers.Dense(units=output_dim, use_bias=True,
                              bias_regularizer=tf.keras.regularizers.l2(l=0.0005),
                              kernel_regularizer=tf.keras.regularizers.l2(l=0.0005))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if activation == 'lrelu':
        x = tf.keras.layers.LeakyReLU()(x)
    elif activation == 'relu':
        x = tf.keras.layers.ReLU()(x)
    elif activation == 'elu':
        x = tf.keras.layers.ELU()(x)
    print(x.shape, 'linear transform')
    return x

def encoder3D(structure):
    name = structure['name']
    input_shape = structure['input_shape']
    filter_num_list = structure['filter_num_list']
    filter_size_list = structure['filter_size_list']
    strides_list = structure['strides_list']
    final_pool = structure['final_pool']
    act = structure['activation']
    final_act = structure['final_activation']
    print('encoder3D', name)
    x = inputs = tf.keras.Input(shape=input_shape)
    print(x.shape, 'input shape')
    for filter_num, filter_size, strides in zip(filter_num_list[:-1], filter_size_list[:-1], strides_list[:-1]):
        x = conv3DEnc(inputs=x, filter_num=filter_num, filter_size=filter_size, strides=strides, activation=act)
    x = tf.keras.layers.Conv3D(filters=filter_num_list[-1], kernel_size=filter_size_list[-1], strides=strides_list[-1],
                               padding='same', activation=None, use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(l=0.0005))(x)
    print(x.shape, 'conv3DEnc final')
    if final_pool == 'average':
        x = tf.reduce_mean(x, axis=[1,2,3])
    elif final_pool == 'max':
        x = tf.reduce_max(x, axis=[1,2,3])
    if final_pool != 'None' or final_pool != None:
        pass
    print('encoder pool', final_pool)
    if final_act == 'sigmoid':
        x = tf.nn.sigmoid(x)
        print('encoder last activation : sigmoid')
    elif final_act == None or final_act == 'None' or final_act == 'linear':
        print('encoder last activation :', final_act)
    return tf.keras.Model(inputs, x, name=name)

def decoder3D(structure):
    name = structure['name']
    input_dim = structure['input_dim']
    output_shape = structure['output_shape']
    filter_num_list = structure['filter_num_list']
    filter_size_list = structure['filter_size_list']
    strides_list = structure['strides_list']
    act = structure['activation']
    final_act = structure['final_activation']

    print('decoder3D', name)
    conv_input_dim_wo_ch = output_shape[:-1]/np.prod(strides_list)
    conv_input_ch = filter_num_list[0] / 64
    if conv_input_ch < 8:
        conv_input_ch = 8  # fixed with arbitrary value
    conv_input_dim = np.concatenate([[-1], conv_input_dim_wo_ch, [conv_input_ch]])
    linear_output_dim = np.prod(conv_input_dim_wo_ch)*conv_input_ch

    x = inputs = tf.keras.layers.Input(shape=[input_dim,])
    print(x.shape, 'input shape')
    x = linearTransform(inputs=x, output_dim=linear_output_dim, activation=act)
    x = tf.reshape(x, shape=conv_input_dim)
    print(x.shape, 'reshape')
    for filter_num, filter_size, strides in zip(filter_num_list[:-1],filter_size_list[:-1],strides_list[:-1]):
        x = conv3DDec(inputs=x, filter_num=filter_num, filter_size=filter_size, strides=strides, activation=act)
    x = tf.keras.layers.Conv3DTranspose(filters=filter_num_list[-1],kernel_size=filter_size_list[-1],
                                        strides=strides_list[-1], padding='same', activation=None, use_bias=False,
                                        kernel_regularizer=tf.keras.regularizers.l2(l=0.0005)
                                        )(x)
    print(x.shape, 'conv3DDec final')
    if final_act == 'sigmoid':
        print('last layer activation : sigmoid')
        x = tf.sigmoid(x)
    elif final_act == 'None' or final_act == None or final_act == 'linear':
        pass
    return tf.keras.Model(inputs, x, name=name)






