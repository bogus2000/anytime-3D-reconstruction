import numpy as np
import time, sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 0 = all messages are logged(default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages  arenot printed
# 3 = INFO, WARNING, and ERROR messages  arenot printed

import src.dataset_loader.KITTI_dataset as KITTI
from src.dataset_loader.modelnet_dataset import dataLoader
import src.net_core.darknet as Darknet
import src.module.nolbo as nolbo
import tensorflow as tf

tf.get_logger().warning('test')
# WARNING:tensorflow:test
tf.get_logger().setLevel('ERROR')
tf.get_logger().warning('test')

imageSizeAndBatchListKITTI = [
    # [576,192,42],
    # [672,224,40],
    # [768,256,38],
    [864,288,40],
    [960,288,38],
    [960,320,36],
    [1056,352,36],
    [1152,384,32],
    [1216,384,32],
    # [1248,448,20],
]

imageSizeAndBatchListPascal = [
    [320,256,28],
    [480,384,20],
    [640,512,18],
    [320,320,28],
    [352,352,28],
    [384,384,28],
    [416,416,28],
    [448,448,26],
    [480,480,24],
    [512,512,20],
    [544,544,20],
    # [576,576,18],
    # [608,608,16],
]

imageSizeAndBatchList = [
    # [224,224,36],
    [256,256,72],
    # [288,288,30],
    # [320,256,28],
    # [480,384,20],
    # [640,512,18],
    # [320,320,28],
    # [352,352,28],
    # [384,384,28],
    # [416,416,28],
    # [448,448,26],
    # [480,480,24],
    # [512,512,20],
    # [544,544,20],
    # [576,576,18],
    # [608,608,16],
]

def train(
        learning_rate = 1e-4,
        config = None, dataset_path = None,
        load_path = None,
        isVAE=False,
):
    if isVAE:
        model = nolbo.nolboSingleObject_modelnet_category_VAE(nolbo_structure=config,
                                                              learning_rate=learning_rate)
    else:
        model = nolbo.nolboSingleObject_modelnet_category_AE(nolbo_structure=config,
                                                          learning_rate=learning_rate)

    data_loader_test = dataLoader(data_path=dataset_path, trainortest='test')


    if load_path != None:
        print('load weights...')
        model.loadModel(load_path=load_path)
        print('done!')

    epoch, epoch_curr = 0., 0.
    iteration, run_time = 0., 0.

    category_vectors = np.zeros((40, 64))
    index_numbers = np.zeros((40, 1))

    print('start training...')
    while epoch < 1:
        epoch_curr = data_loader_test.epoch
        data_start = data_loader_test.batchStart
        data_length = data_loader_test.dataLength

        batch_data = data_loader_test.getNextBatch(batchSize=72)
        inst_list, category_list, input_images, output_images = batch_data['inst_list'], batch_data['class_list'], batch_data['input_images'], batch_data['input_images']
        inputs = input_images

        if epoch!=epoch_curr and iteration!=0:
            break
        epoch = epoch_curr

        latents = model.getLatent(inputs=inputs)
        for latent, category in zip(latents, category_list):
            category_vectors[np.argmax(category)] += latent
            index_numbers[np.argmax(category)] += 1.
    category_vectors = category_vectors / index_numbers
    np.save(os.path.join(load_path, 'category_vectors.npy'), category_vectors)


latent_dim = 64
config = {
    'z_category_dim': latent_dim,
    'encoder': {
        'name':'encoder3D',
        'input_shape': [64,64,64,1], # or [None,None,None,1]
        'filter_num_list': [64,128,256,512, 2*latent_dim],
        'filter_size_list': [4,4,4,4,4],
        'strides_list': [2,2,2,2,1],
        'final_pool': 'average',
        'activation': 'elu',
        'final_activation': 'None',
    },
    'decoder':{
        'name':'decoder',
        'input_dim': latent_dim,
        'output_shape': [64,64,64,1],
        'filter_num_list': [512,256,128,64,1],
        'filter_size_list': [4,4,4,4,4],
        'strides_list': [1,2,2,2,2],
        'activation': 'elu',
        'final_activation': 'sigmoid'
    },
}


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# if __name__ == '__main__':
#     sys.exit(train(
#         config=config, dataset_path='/media/yonsei/4TB_HDD/dataset/modelNet/',
#         load_path='./weights/modelnet_category_AE/',
#         isVAE=False,
#     ))

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# if __name__ == '__main__':
#     sys.exit(train(
#         config=config, dataset_path='/media/yonsei/4TB_HDD/dataset/modelNet/',
#         load_path='./weights/modelnet_category_AE_dr/',
#         isVAE=False,
#     ))
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# if __name__ == '__main__':
#     sys.exit(train(
#         config=config, dataset_path='/media/yonsei/4TB_HDD/dataset/modelNet/',
#         load_path='./weights/modelnet_category_VAE/',
#         isVAE=True,
#     ))
#
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
if __name__ == '__main__':
    sys.exit(train(
        config=config, dataset_path='/media/yonsei/4TB_HDD/dataset/modelNet/',
        load_path='./weights/modelnet_category_VAE_dr/',
        isVAE=True,
    ))



