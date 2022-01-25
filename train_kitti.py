import numpy as np
import time, sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
        training_epoch = 1000,
        learning_rate = 1e-4,
        config = None,
        save_path = None, load_path = None,
        load_encoder_backbone_path = None, load_encoder_backbone_name = None,
        load_decoder_path = None, load_decoder_name = None,
):

    model = nolbo.nolboSingleObject_instOnly(nolbo_structure=config,
                                    backbone_style=Darknet.Darknet19,
                                    learning_rate=learning_rate)
    data_loader_kitti = KITTI.dataLoaderSingleObject(trainOrVal='train')
    # data_loader_pascal = pascal3D.dataLoaderSingleObject(trainOrVal='train',
    #                                          Pascal3DDataPath='/media/yonsei/4TB_HDD/dataset/PASCAL3D+_release1.1/')


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

    loss = np.zeros(5)
    epoch = 0.
    iteration, run_time = 0., 0.

    print('start training...')
    while epoch < training_epoch:
        start_time = time.time()
        periodOfImageSize = 3
        if int(iteration) % (periodOfImageSize * len(imageSizeAndBatchList)) == 0:
            np.random.shuffle(imageSizeAndBatchList)
        image_col, image_row, batch_size = imageSizeAndBatchList[int(iteration) % int((periodOfImageSize * len(imageSizeAndBatchList)) / periodOfImageSize)]
        epoch_curr = data_loader_kitti.epoch
        data_start = data_loader_kitti.dataStart
        data_length = data_loader_kitti.dataLength

        batch_data = data_loader_kitti.getNextBatch(batchSizeof3DShape=batch_size, imageSize=(image_col, image_row))
        inst_list, sin, cos, input_images, output_images = batch_data
        inputs = input_images, output_images, inst_list

        if epoch!=epoch_curr and iteration!=0:
            print('')
            iteration = 0
            loss = loss * 0.
            run_time = 0.
            if save_path != None:
                print('save model...')
                model.saveModel(save_path=save_path)
        epoch = epoch_curr

        loss_temp = model.fit(inputs=inputs)
        end_time = time.time()

        loss = (loss*iteration + np.array(loss_temp))/(iteration + 1.0)
        run_time = (run_time * iteration + (end_time-start_time)) / (iteration+1.0)

        sys.stdout.write(
            "it:{:04d} rt:{:.2f} Ep_o:{:03d} ".format(int(iteration + 1), run_time, int(epoch + 1)))
        sys.stdout.write("cur_o/tot_o:{:04d}/{:04d} ".format(data_start, data_length))
        sys.stdout.write(
            "kl:{:.4f}, shape:{:.4f}, reg:{:.4f}, pr:{:.4f}, rc:{:.4f}   \r".format(loss[0], loss[1], loss[2], loss[3], loss[4]))
        sys.stdout.flush()

        if np.sum(loss) != np.sum(loss):
            print('')
            print('NaN')
            return
        iteration += 1.0

latent_dim = 16
config = {
    'encoder_backbone':{
        'name' : 'nolbo_backbone',
        'z_inst_dim':latent_dim,
        'activation' : 'elu',
    },
    'encoder_head':{
        'name' : 'nolbo_head',
        'output_dim' : 2 * latent_dim,  # (class+inst) * (mean+logvar)
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
    'prior_inst': {
        'name': 'priornet_inst',
        'input_dim': 10,  # inst_num (one-hot vector)
        'unit_num_list': [32, latent_dim],
        'core_activation': 'elu',
        'const_log_var': 0.0,
    }
}

if __name__ == '__main__':
    sys.exit(train(
        training_epoch=1000, learning_rate=1e-4,
        config=config,
        save_path='./weights/kitti/',
        load_path='./weights/kitti/',
        # load_encoder_backbone_path='./weights/imagenet_and_place365/',
        # load_encoder_backbone_name='imagenet_backbone',
        # load_decoder_path='./weights/AE3D/',
        # load_decoder_name='decoder3D',
    ))



