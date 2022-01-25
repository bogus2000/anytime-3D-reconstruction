import numpy as np
import time, sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 0 = all messages are logged(default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages  arenot printed
# 3 = INFO, WARNING, and ERROR messages  arenot printed

import src.dataset_loader.KITTI_dataset as KITTI
import src.dataset_loader.pascal3D as pascal3D
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
        config = None,
        load_path = None,
        load_encoder_backbone_path = None, load_encoder_backbone_name = None,
        load_decoder_path = None, load_decoder_name = None,
        isVAE=False,
):

    if isVAE:
        model = nolbo.nolboSingleObject_VAE(nolbo_structure=config,
                                        backbone_style=Darknet.Darknet19,
                                        learning_rate=learning_rate)
    else:
        model = nolbo.nolboSingleObject_AE(nolbo_structure=config,
                                        backbone_style=Darknet.Darknet19,
                                        learning_rate=learning_rate)
    # data_loader_kitti = KITTI.dataLoaderSingleObject(trainOrVal='train')
    data_loader_pascal = pascal3D.dataLoaderSingleObject(trainOrVal='val',
                                                         Pascal3DDataPath='/media/yonsei/4TB_HDD/dataset/PASCAL3D+_release1.1/')


    if load_path != None:
        print('load weights...')
        model.loadModel(load_path=load_path)
        # model.loadEncoder(load_path=load_path)
        # model.loadEncoderBackbone(load_path=load_path)
        # model.loadEncoderHead(load_path=load_path)
        # model.loadDecoder(load_path=load_path)
        # model.loadPriornet(load_path=load_path)
        print('done!')

    if load_encoder_backbone_path != None:
        print('load encoder backbone weights...')
        model.loadEncoderBackbone(
            load_path=load_encoder_backbone_path,
            file_name=load_encoder_backbone_name
        )
        print('done!')

    if load_decoder_path != None:
        print('load decoder weights...')
        model.loadDecoder(
            load_path=load_decoder_path,
            file_name=load_decoder_name
        )
        print('done!')
    loss = np.zeros(3)
    epoch, epoch_curr = 0., 0.
    iteration, run_time = 0., 0.

    category_vectors = np.zeros((12, 16))
    index_numbers = np.zeros((12, 1))

    print('start training...')
    while epoch < 1:
        start_time = time.time()
        periodOfImageSize = 3
        if int(iteration) % (periodOfImageSize * len(imageSizeAndBatchList)) == 0:
            np.random.shuffle(imageSizeAndBatchList)
        image_col, image_row, batch_size = imageSizeAndBatchList[int(iteration) % int((periodOfImageSize * len(imageSizeAndBatchList)) / periodOfImageSize)]
        epoch_curr = data_loader_pascal.epoch
        data_start = data_loader_pascal.dataStart
        data_length = data_loader_pascal.dataLength

        batch_data = data_loader_pascal.getNextBatch(batchSizeof3DShape=batch_size, imageSize=(image_col, image_row), augmentation=False)
        inst_list, category_list, sin, cos, input_images, output_images = batch_data
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

latent_dim = 16
config = {
    'encoder_backbone':{
        'name' : 'nolbo_backbone',
        'z_dim':latent_dim,
        'activation' : 'elu',
    },
    'encoder_head':{
        'name' : 'nolbo_head',
        'output_dim' : 2*latent_dim,  # (class+inst) * (mean+logvar)
        'filter_num_list':[],
        'filter_size_list':[],
        # 'filter_num_list':[1024,1024,1024],
        # 'filter_size_list':[3,3,3],
        'activation':'elu',
    },
    'decoder':{
        'name':'decoder',
        'input_dim' : latent_dim,
        'output_shape':[64,64,64,1],
        'filter_num_list':[512,256,128,64,1],
        'filter_size_list':[4,4,4,4,4],
        'strides_list':[1,2,2,2,2],
        'activation':'elu',
        'final_activation':'sigmoid'
    },
}
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# if __name__ == '__main__':
#     sys.exit(train(
#         training_epoch=1000, learning_rate=1e-4,
#         config=config,
#         load_path='./weights/pascal_AE/',
#         missing_pr=0.3,
#         learn='train',
#     ))
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# if __name__ == '__main__':
#     sys.exit(train(
#         training_epoch=1000, learning_rate=1e-4,
#         config=config,
#         load_path='./weights/pascal_AE/',
#         missing_pr=0.5,
#         learn='train',
#     ))
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# if __name__ == '__main__':
#     sys.exit(train(
#         training_epoch=1000, learning_rate=1e-4,
#         config=config,
#         load_path='./weights/pascal_AE/',
#         missing_pr=0.7,
#         learn='train',
#     ))

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
if __name__ == '__main__':
    sys.exit(train(
        learning_rate=1e-4,
        config=config,
        load_path='./weights/pascal_VAE_dr/',
        isVAE=True,
    ))



