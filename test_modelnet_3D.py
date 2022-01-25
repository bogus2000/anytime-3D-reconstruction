import numpy as np
import time, sys, os, cv2
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

config_AE = {
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

config_VAE = {
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
}

def test(
        dataset_path=None,
        batch_size=72,
        save_dir='./data/eval/image/modelnet/'
):

    model = nolbo.nolboSingleObject_modelnet_category_only(nolbo_structure=config)
    model_AE = nolbo.nolboSingleObject_modelnet_category_AE(nolbo_structure=config_AE)
    model_VAE = nolbo.nolboSingleObject_modelnet_category_VAE(nolbo_structure=config_VAE)

    data_loader_test = dataLoader(data_path=dataset_path, trainortest='test')

    batch_data = data_loader_test.getNextBatch(batchSize=batch_size)
    inst_list, category_list, input_images, output_images = batch_data['inst_list'], batch_data['class_list'], \
                                                            batch_data['input_images'], batch_data['input_images']
    data = input_images, output_images, category_list
    for missing_pr in [0.30, 0.50, 0.70, 0.90]:
        print(missing_pr)
        model.loadModel('./weights/modelnet_category/')
        model_AE.loadModel('./weights/modelnet_category_AE')
        model_VAE.loadModel('./weights/modelnet_category_VAE')

        input_images, output_images = data[0], data[1]
        output_images_pred, _, _, _, _, output_images_pred_corrected, _, _, _, _ = model.getEval(inputs=data, missing_prob=missing_pr)
        output_images_pred_AE, _, _, _ = model_AE.getEval(inputs=data[0:2], missing_prob=missing_pr)
        output_images_pred_VAE, _, _, _ = model_VAE.getEval(inputs=data[0:2], missing_prob=missing_pr)

        i = 0
        for input_image, output_image_gt, output_image_mVAE, output_image_mVAE_corrected, output_image_AE, output_image_VAE in zip(
            input_images, output_images, output_images_pred, output_images_pred_corrected, output_images_pred_AE, output_images_pred_VAE
        ):
            file_name = '{:03d}'.format(int(i))
            output_image_gt = np.reshape(output_image_gt, (64*64, 64))
            np.savetxt(os.path.join(save_dir, file_name+'_'+str(missing_pr)+'_gt.txt'), output_image_gt)
            output_image_mVAE = np.reshape(output_image_mVAE, (64 * 64, 64))
            np.savetxt(os.path.join(save_dir, file_name+'_'+str(missing_pr)+'_ca.txt'), output_image_mVAE)
            output_image_mVAE_corrected = np.reshape(output_image_mVAE_corrected, (64 * 64, 64))
            np.savetxt(os.path.join(save_dir, file_name+'_'+str(missing_pr)+'_ca_corrected.txt'), output_image_mVAE_corrected)
            output_image_AE = np.reshape(output_image_AE, (64 * 64, 64))
            np.savetxt(os.path.join(save_dir, file_name+'_'+str(missing_pr)+'_AE.txt'), output_image_AE)
            output_image_VAE = np.reshape(output_image_VAE, (64 * 64, 64))
            np.savetxt(os.path.join(save_dir, file_name+'_'+str(missing_pr)+'_VAE.txt'), output_image_VAE)
            i += 1


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == '__main__':
    sys.exit(test(
        dataset_path='/media/yonsei/4TB_HDD/dataset/modelNet/',
        batch_size=128,
        save_dir='./data/eval/image/modelnet/'
    ))



