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
        training_epoch = 1000,
        learning_rate = 1e-4,
        config = None,
        save_path = None, load_path = None,
        load_encoder_backbone_path = None, load_encoder_backbone_name = None,
        load_decoder_path = None, load_decoder_name = None,
        missing_pr = 0.3,
        learn='train',
):

    model = nolbo.nolboSingleObject_VAE(nolbo_structure=config,
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
    loss = np.zeros(8)
    epoch, epoch_curr = 0., 0.
    iteration, run_time = 0., 0.

    output_images_gt = []
    output_images_preds = []
    category_labels = []

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
        inputs = input_images, output_images, category_list

        if epoch!=epoch_curr and iteration!=0:
            break
        epoch = epoch_curr

        # loss_temp = model.fit(inputs=inputs)
        category_vectors = np.load(os.path.join(load_path, 'category_vectors.npy')).astype('float32')
        output_images_pred, loss_shape, pr, rc, acc_cat, \
        output_images_pred_corrected, loss_shape_corrected, pr_corrected, rc_corrected, acc_cat_corrected = model.getEval(
            inputs=inputs, category_vectors=category_vectors, missing_prob=missing_pr)

        category_labels.append(np.array(category_list))
        output_images_gt.append(np.array(output_images))
        output_images_preds.append(np.array(output_images_pred))

        # output_images_pred, loss_shape, pr, rc, acc_cat, acc_inst,\
        # output_images_pred_corrected, loss_shape_corrected, pr_corrected, rc_corrected, acc_cat_corrected = model.getEval(inputs=inputs_test)[1:]

        loss_temp = loss_shape, pr, rc, acc_cat, loss_shape_corrected, pr_corrected, rc_corrected, acc_cat_corrected
        end_time = time.time()

        loss = (loss * iteration + np.array(loss_temp)) / (iteration + 1.0)
        run_time = (run_time * iteration + (end_time - start_time)) / (iteration + 1.0)

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
    # np.save('./data/pascal/data_VAE/learn_' + learn + '/' + str(missing_pr) + '_cl_label.npy', category_labels)
    # del category_labels
    # output_images_gt = np.concatenate(output_images_gt, axis=0)
    # np.save('./data/pascal/data_VAE/learn_' + learn + '/' + str(missing_pr) + '_gt.npy', output_images_gt)
    # del output_images_gt
    # output_images_preds = np.concatenate(output_images_preds, axis=0)
    # np.save('./data/pascal/data_VAE/learn_' + learn + '/' + str(missing_pr) + '_pred.npy', output_images_preds)
    # del output_images_preds

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
#         load_path='./weights/pascal_VAE/',
#         missing_pr=0.3,
#         learn='train',
#     ))
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# if __name__ == '__main__':
#     sys.exit(train(
#         training_epoch=1000, learning_rate=1e-4,
#         config=config,
#         load_path='./weights/pascal_VAE/',
#         missing_pr=0.5,
#         learn='train',
#     ))
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# if __name__ == '__main__':
#     sys.exit(train(
#         training_epoch=1000, learning_rate=1e-4,
#         config=config,
#         load_path='./weights/pascal_VAE/',
#         missing_pr=0.7,
#         learn='train',
#     ))
#
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
if __name__ == '__main__':
    sys.exit(train(
        training_epoch=1000, learning_rate=1e-4,
        config=config,
        load_path='./weights/pascal_VAE/',
        missing_pr=0.0,
        learn='train',
    ))



