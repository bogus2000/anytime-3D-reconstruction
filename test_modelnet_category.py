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
        training_epoch = 1000,
        learning_rate = 1e-4,
        config = None, dataset_path = None,
        save_path = None, load_path = None,
        load_encoder_backbone_path = None, load_encoder_backbone_name = None,
        load_decoder_path = None, load_decoder_name = None,
        missing_pr = 0.3,
        learn='train',
):

    model = nolbo.nolboSingleObject_modelnet_category_only(nolbo_structure=config,
                                    learning_rate=learning_rate)
    # data_loader_kitti = KITTI.dataLoaderSingleObject(trainOrVal='train')
    # data_loader_pascal_train = pascal3D.dataLoaderSingleObject(trainOrVal='train', Pascal3DDataPath='/media/yonsei/4TB_HDD/dataset/PASCAL3D+_release1.1/')
    data_loader = dataLoader(data_path=dataset_path, trainortest='test')

    if load_path != None:
        print('load weights...')
        model.loadModel(load_path=load_path)
        print('done!')

    loss = np.zeros(8)
    epoch, epoch_curr = 0., 0.
    iteration, run_time = 0., 0.

    output_images_gt = []
    output_images_preds = []
    output_images_pred_correcteds = []
    category_labels = []

    print('start training...')
    while epoch < 1:
        start_time = time.time()
        periodOfImageSize = 3
        if int(iteration) % (periodOfImageSize * len(imageSizeAndBatchList)) == 0:
            np.random.shuffle(imageSizeAndBatchList)
        image_col, image_row, batch_size = imageSizeAndBatchList[int(iteration) % int((periodOfImageSize * len(imageSizeAndBatchList)) / periodOfImageSize)]
        epoch_curr = data_loader.epoch
        data_start = data_loader.batchStart
        data_length = data_loader.dataLength

        batch_data = data_loader.getNextBatch(batchSize=batch_size)
        inst_list, category_list, input_images, output_images = batch_data['inst_list'], batch_data['class_list'], \
                                                                batch_data['input_images'], batch_data['input_images']
        inputs = input_images, output_images, category_list

        if epoch!=epoch_curr and iteration!=0:
            break
        epoch = epoch_curr

        # loss_temp = model.fit(inputs=inputs)

        output_images_pred, loss_shape, pr, rc, acc_cat, \
        output_images_pred_corrected, loss_shape_corrected, pr_corrected, rc_corrected, acc_cat_corrected = model.getEval(inputs=inputs, missing_prob=missing_pr)

        category_labels.append(np.array(category_list))
        output_images_gt.append(np.array(output_images))
        output_images_preds.append(np.array(output_images_pred))
        output_images_pred_correcteds.append(np.array(output_images_pred_corrected))

        # output_images_pred, loss_shape, pr, rc, acc_cat, acc_inst,\
        # output_images_pred_corrected, loss_shape_corrected, pr_corrected, rc_corrected, acc_cat_corrected = model.getEval(inputs=inputs_test)[1:]

        loss_temp = loss_shape, pr, rc, acc_cat, loss_shape_corrected, pr_corrected, rc_corrected, acc_cat_corrected
        end_time = time.time()

        loss = (loss * iteration + np.array(loss_temp)) / (iteration + 1.0)
        run_time = (run_time * iteration + (end_time-start_time)) / (iteration+1.0)

        sys.stdout.write(
            "it:{:04d} rt:{:.2f} Ep_o:{:03d} ".format(int(iteration + 1), run_time, int(epoch + 1)))
        sys.stdout.write("cur_o/tot_o:{:05d}/{:05d} ".format(data_start, data_length))
        sys.stdout.write(
            "loss:{:.4f}, pr:{:.4f}, rc:{:.4f}, c:{:.4f}, ".format(
                loss[0], loss[1], loss[2], loss[3]))
        sys.stdout.write(
            "closs:{:.4f}, cpr:{:.4f}, crc:{:.4f}, cc:{:.4f}  \r".format(
                loss[4], loss[5], loss[6], loss[7]))
        sys.stdout.flush()

        if np.sum(loss) != np.sum(loss):
            print('')
            print('NaN')
            return
        iteration += 1.0
    print('')
    # category_labels = np.concatenate(category_labels, axis=0)
    # np.save('./data/modelnet/data_category/learn_' + learn + '/' + str(missing_pr) + '_cl_label.npy', category_labels)
    # del category_labels
    # output_images_gt = np.concatenate(output_images_gt, axis=0)
    # np.save('./data/modelnet/data_category/learn_' + learn + '/' + str(missing_pr) + '_gt.npy', output_images_gt)
    # del output_images_gt
    # output_images_preds = np.concatenate(output_images_preds, axis=0)
    # np.save('./data/modelnet/data_category/learn_' + learn + '/' + str(missing_pr) + '_pred.npy', output_images_preds)
    # del output_images_preds
    # output_images_pred_correcteds = np.concatenate(output_images_pred_correcteds, axis=0)
    # np.save('./data/modelnet/data_category/learn_'+learn+'/' + str(missing_pr) + '_corrected.npy', output_images_pred_correcteds)
    # del output_images_pred_correcteds


latent_dim = 64
config = {
    'z_category_dim': latent_dim,
    'encoder': {
        'name':'encoder3D',
        'input_shape': [64,64,64,1], # or [None,None,None,1]
        'filter_num_list': [64,128,256,512, 2 * latent_dim],
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
    'prior_class': {
        'name': 'priornet_class',
        'input_dim': 40,  # class num (one-hot vector)
        'unit_num_list': [32, latent_dim],
        'core_activation': 'elu',
        'const_log_var': 0.0,
    },
}
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == '__main__':
    sys.exit(train(
        training_epoch=1000, learning_rate=1e-4,
        config=config, dataset_path='/media/yonsei/4TB_HDD/dataset/modelNet/',
        load_path='./weights/modelnet_category/',
        missing_pr=0.0,
        learn='train',
    ))

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# if __name__ == '__main__':
#     sys.exit(train(
#         training_epoch=1000, learning_rate=1e-4,
#         config=config, dataset_path='/media/yonsei/4TB_HDD/dataset/modelNet/',
#         load_path='./weights/modelnet_category/',
#         missing_pr=0.5,
#         learn='train',
#     ))
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# if __name__ == '__main__':
#     sys.exit(train(
#         training_epoch=1000, learning_rate=1e-4,
#         config=config, dataset_path='/media/yonsei/4TB_HDD/dataset/modelNet/',
#         load_path='./weights/modelnet_category/',
#         missing_pr=0.7,
#         learn='train',
#     ))
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# if __name__ == '__main__':
#     sys.exit(train(
#         training_epoch=1000, learning_rate=1e-4,
#         config=config, dataset_path='/media/yonsei/4TB_HDD/dataset/modelNet/',
#         load_path='./weights/modelnet_category/',
#         missing_pr=0.9,
#         learn='train',
#     ))

