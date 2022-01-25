import numpy as np
import time, sys, os, cv2
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

latent_dim = 16
config = {
    'encoder_backbone':{
        'name' : 'nolbo_backbone',
        'z_category_dim':latent_dim//2,
        'z_inst_dim':latent_dim//2,
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
    'prior_class': {
        'name': 'priornet_class',
        'input_dim': 12,  # class num (one-hot vector)
        'unit_num_list': [32, latent_dim//2],
        'core_activation': 'elu',
        'const_log_var': 0.0,
    },
    'prior_inst': {
        'name': 'priornet_inst',
        'input_dim': 12 + 10,  # class_num + inst_num (one-hot vector)
        'unit_num_list': [32, latent_dim//2],
        'core_activation': 'elu',
        'const_log_var': 0.0,
    }
}

config_AE = {
    'encoder_backbone':{
        'name' : 'nolbo_backbone',
        'z_dim':latent_dim,
        'activation' : 'elu',
    },
    'encoder_head':{
        'name' : 'nolbo_head',
        'output_dim' : latent_dim,  # (class+inst) * (mean+logvar)
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

config_VAE = {
    'encoder_backbone':{
        'name' : 'nolbo_backbone',
        'z_dim':latent_dim,
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
}

def train():

    model = nolbo.nolboSingleObject(nolbo_structure=config,
                                    backbone_style=Darknet.Darknet19,
                                    learning_rate=None)
    model_AE = nolbo.nolboSingleObject_AE(nolbo_structure=config_AE,
                                    backbone_style=Darknet.Darknet19,
                                    learning_rate=None)
    model_VAE = nolbo.nolboSingleObject_VAE(nolbo_structure=config_VAE,
                                       backbone_style=Darknet.Darknet19,
                                       learning_rate=None)

    data_loader_pascal_train = pascal3D.dataLoaderSingleObject(trainOrVal='train',
                                             Pascal3DDataPath='/media/yonsei/4TB_HDD/dataset/PASCAL3D+_release1.1/')
    data_loader_pascal_test = pascal3D.dataLoaderSingleObject(trainOrVal='val',
                                                         Pascal3DDataPath='/media/yonsei/4TB_HDD/dataset/PASCAL3D+_release1.1/')

    image_col, image_row, batch_size = 256, 256, 72
    batch_data_train = data_loader_pascal_train.getNextBatch(batchSizeof3DShape=batch_size, imageSize=(image_col, image_row), augmentation=False)
    batch_data_test = data_loader_pascal_test.getNextBatch(batchSizeof3DShape=batch_size, imageSize=(image_col, image_row), augmentation=False)
    inst_list, category_list, sin, cos, input_images, output_images = batch_data_train
    inputs_train = input_images, output_images, category_list, inst_list
    inst_list_test, category_list_test, sin_test, cos_test, input_images_test, output_images_test = batch_data_test
    inputs_test = input_images_test, output_images_test, category_list_test, inst_list_test

    for dir, trainortest, data in zip(['train', 'test'], ['', '_test'], [inputs_train, inputs_test]):
        for missing_pr in [0.15, 0.25, 0.35, 0.45]:
            print(dir, missing_pr)
            model.loadModel('weights/pascal'+trainortest)
            model_AE.loadModel('weights/pascal_AE'+trainortest)
            model_VAE.loadModel('weights/pascal_VAE'+trainortest)

            input_images, output_images = data[0], data[1]
            output_images_pred, _, _, _, _, _, output_images_pred_corrected, _, _, _, _ = model.getEval(inputs=data, missing_prob=missing_pr*2)
            output_images_pred_AE, _, _, _ = model_AE.getEval(inputs=data[0:2], missing_prob=missing_pr)
            output_images_pred_VAE, _, _, _ = model_VAE.getEval(inputs=data[0:2], missing_prob=missing_pr)

            save_dir_3D = os.path.join('data_3D', dir, str(missing_pr * 2), '3D')
            save_dir_image = os.path.join('data_3D', dir, str(missing_pr * 2), 'image')
            i = 0
            for input_image, output_image_gt, output_image_mVAE, output_image_mVAE_corrected, output_image_AE, output_image_VAE in zip(
                input_images, output_images, output_images_pred, output_images_pred_corrected, output_images_pred_AE, output_images_pred_VAE
            ):
                file_name = '{:03d}'.format(int(i))
                cv2.imwrite(os.path.join(save_dir_image, file_name+'.jpg'), input_image*255)
                output_image_gt = np.reshape(output_image_gt, (64*64, 64))
                np.savetxt(os.path.join(save_dir_3D, file_name+'_gt.txt'), output_image_gt)
                output_image_mVAE = np.reshape(output_image_mVAE, (64 * 64, 64))
                np.savetxt(os.path.join(save_dir_3D, file_name + '_mVAE.txt'), output_image_mVAE)
                output_image_mVAE_corrected = np.reshape(output_image_mVAE_corrected, (64 * 64, 64))
                np.savetxt(os.path.join(save_dir_3D, file_name + '_mVAE_c.txt'), output_image_mVAE_corrected)
                output_image_AE = np.reshape(output_image_AE, (64 * 64, 64))
                np.savetxt(os.path.join(save_dir_3D, file_name + '_AE.txt'), output_image_AE)
                output_image_VAE = np.reshape(output_image_VAE, (64 * 64, 64))
                np.savetxt(os.path.join(save_dir_3D, file_name + '_VAE.txt'), output_image_VAE)
                i += 1


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == '__main__':
    sys.exit(train())



