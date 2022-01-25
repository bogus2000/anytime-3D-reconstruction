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


def train(
        training_epoch = 1000,
        learning_rate = 1e-4, batch_size=32,
        config = None,
        dataset_path = None,
        save_path = None, load_path = None,
        load_decoder_path = None, load_decoder_name = None,
):

    model = nolbo.nolboSingleObject_modelnet_category_AE(nolbo_structure=config,
                                                         dropout=True,
                                    learning_rate=learning_rate)
    # data_loader_kitti = KITTI.dataLoaderSingleObject(trainOrVal='train')
    data_loader_train = dataLoader(data_path=dataset_path, trainortest='train')
    data_loader_test = dataLoader(data_path=dataset_path, trainortest='test')


    if load_path != None:
        print('load weights...')
        model.loadModel(load_path=load_path)
        # model.loadEncoder(load_path=load_path)
        # model.loadDecoder(load_path=load_path)
        # model.loadPriornet(load_path=load_path)
        print('done!')

    if load_decoder_path != None:
        print('load decoder weights...')
        model.loadDecoder(
            load_path=load_decoder_path,
            file_name=load_decoder_name
        )
        print('done!')
    loss = np.zeros(3)
    loss_train, loss_test = np.zeros(3), np.zeros(3)
    epoch = 0.
    iteration, run_time = 0., 0.

    print('start training...')
    while epoch < training_epoch:
        start_time = time.time()
        epoch_curr = data_loader_train.epoch
        data_start = data_loader_train.batchStart
        data_length = data_loader_train.dataLength

        batch_data = data_loader_train.getNextBatch(batchSize=batch_size)
        batch_data_test = data_loader_test.getNextBatch(batchSize=batch_size)
        inst_list, category_list, input_images, output_images = batch_data['inst_list'], batch_data['class_list'], batch_data['input_images'], batch_data['input_images']
        inputs = input_images, output_images
        inst_list_test, category_list_test, input_images_test, output_images_test = batch_data_test['inst_list'], batch_data_test['class_list'], batch_data_test['input_images'], batch_data_test['input_images']
        inputs_test = input_images_test, output_images_test

        if epoch!=epoch_curr and iteration!=0:
            print('')
            iteration = 0
            loss, loss_train, loss_test = loss * 0., loss_train*0., loss_test*0.
            run_time = 0.
            if save_path != None:
                print('save model...')
                model.saveModel(save_path=save_path)
        epoch = epoch_curr

        loss_temp = model.fit(inputs=inputs)
        loss_train_temp = model.getEval(inputs=inputs)[1:]
        loss_test_temp = model.getEval(inputs=inputs_test)[1:]
        end_time = time.time()

        loss = (loss * iteration + np.array(loss_temp)) / (iteration + 1.0)
        loss_train = (loss_train*iteration + np.array(loss_train_temp))/(iteration + 1.0)
        loss_test = (loss_test * iteration + np.array(loss_test_temp)) / (iteration + 1.0)
        run_time = (run_time * iteration + (end_time-start_time)) / (iteration+1.0)

        sys.stdout.write(
            "it:{:04d} rt:{:.2f} Ep_o:{:03d} ".format(int(iteration + 1), run_time, int(epoch + 1)))
        sys.stdout.write("cur_o/tot_o:{:04d}/{:04d} ".format(data_start, data_length))
        sys.stdout.write(
            "shape:{:.4f}, pr:{:.4f}, rc:{:.4f} ".format(
                loss_train[0], loss_train[1], loss_train[2]))
        sys.stdout.write(
            "shape:{:.4f}, pr:{:.4f}, rc:{:.4f}  \r".format(
                loss_test[0], loss_test[1], loss_test[2]))
        sys.stdout.flush()

        if np.sum(loss) != np.sum(loss):
            print('')
            print('NaN')
            return
        iteration += 1.0

latent_dim = 64
config = {
    'z_category_dim': latent_dim,
    'encoder': {
        'name':'encoder3D',
        'input_shape': [64,64,64,1], # or [None,None,None,1]
        'filter_num_list': [64,128,256,512, latent_dim],
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
if __name__ == '__main__':
    sys.exit(train(
        training_epoch=1000, learning_rate=1e-4, batch_size=64,
        config=config,
        dataset_path='/media/yonsei/4TB_HDD/dataset/modelNet/',
        save_path='./weights/modelnet_category_AE_dr/',
        load_path='./weights/modelnet_category_AE_dr/',
        # load_encoder_backbone_path='./weights/imagenet_and_place365/',
        # load_encoder_backbone_name='imagenet_backbone',
        # load_decoder_path='./weights/AE3D/',
        # load_decoder_name='decoder3D',
    ))



