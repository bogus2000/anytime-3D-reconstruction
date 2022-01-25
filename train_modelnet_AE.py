import numpy as np
import time, sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 0 = all messages are logged(default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages  arenot printed
# 3 = INFO, WARNING, and ERROR messages  arenot printed

from src.dataset_loader.modelnet_dataset import dataLoader
import src.module.AE3D as AE3D
import tensorflow as tf

tf.get_logger().warning('test')
# WARNING:tensorflow:test
tf.get_logger().setLevel('ERROR')
tf.get_logger().warning('test')

latent_dim=16
#=========== autoencoder architecture example (from 3D GAN) ===============
encoder_structure = {
    'name':'encoder3D',
    'input_shape':[64,64,64,1], # or [None,None,None,1]
    'filter_num_list':[64,128,256,512,latent_dim],
    'filter_size_list':[4,4,4,4,4],
    'strides_list':[2,2,2,2,1],
    'final_pool':'average',
    'activation':'elu',
    'final_activation':'None',
}
decoder_structure = {
    'name':'decoder3D',
    'input_dim' : latent_dim,  # must be same as encoder filter_num_list[-1]
    'output_shape':[64,64,64,1],
    'filter_num_list':[512,256,128,64,1],
    'filter_size_list':[4,4,4,4,4],
    'strides_list':[1,2,2,2,2],
    'activation':'elu',
    'final_activation':'sigmoid'
}

def train(training_epoch=1000,
          learning_rate=1e-4, BATCH_SIZE_PER_REPLICA=32,
          dataset_path=None,
          encoder_structure=None, decoder_structure=None,
          save_path=None, load_path=None):

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = AE3D.AE3D(encoder_structure=encoder_structure,
                          decoder_structure=decoder_structure,
                          BATCH_SIZE_PER_REPLICA=BATCH_SIZE_PER_REPLICA, strategy=strategy,
                          learning_rate=learning_rate)
    data_loader = dataLoader(data_path=dataset_path)

    if load_path != None:
        print('load weights...')
        model.loadModel(load_path=load_path)
        print('done!')

    loss = np.zeros(3)
    precision, recall = 0.0, 0.0
    epoch, iteration, run_time = 0., 0., 0.

    print('start training...')
    while epoch < training_epoch:
        start_time = time.time()
        batch_size = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
        batch_data = data_loader.getNextBatch(batchSize=batch_size)
        input_images = batch_data['input_images']
        output_images = input_images
        epoch_curr = data_loader.epoch
        data_start = data_loader.batchStart
        data_length = data_loader.dataLength

        train_dataset = tf.data.Dataset.from_tensor_slices((input_images, output_images)).batch(batch_size)
        train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

        if epoch_curr != epoch or ((iteration+1)%1000==0 and (iteration+1)!=1):
            print('')
            iteration = 0
            loss = loss * 0.0
            run_time = 0.0
            if save_path != None:
                print('save model...')
                model.saveModel(save_path)
        epoch = epoch_curr
        rloss_temp, ploss_temp, total_loss_temp, pr_temp, rc_temp = model.distributed_fit(
            inputs=next(iter(train_dist_dataset)))
        end_time = time.time()

        loss = (loss * iteration + np.array([rloss_temp, ploss_temp,total_loss_temp])) / (iteration + 1.0)
        precision = (precision*iteration + pr_temp)/(iteration+1.0)
        recall = (recall*iteration + rc_temp)/(iteration+1.0)
        run_time = (run_time * iteration + (end_time - start_time)) / (iteration + 1.0)
        sys.stdout.write(
            "Epoch:{:03d} iter:{:04d} runtime:{:.2f} ".format(int(epoch + 1), int(iteration + 1), run_time))
        sys.stdout.write("batch:{} cur/tot:{:05d}/{:05d} ".format(batch_size, data_start, data_length))
        sys.stdout.write(
            "rloss:{:.4f}, ploss:{:.4f}, tloss:{:.4f} ".format(loss[0], loss[1], loss[2]))
        sys.stdout.write(" pr:{:.4f}, rc:{:.4f}   \r".format(precision, recall))
        sys.stdout.flush()

        if np.sum(loss) != np.sum(loss):
            print('')
            print('NaN')
            return
        iteration = iteration + 1.0

if __name__ == "__main__":
    sys.exit(train(
        training_epoch=1000,
        learning_rate=1e-3, BATCH_SIZE_PER_REPLICA=110,
        dataset_path='/media/yonsei/4TB_HDD/dataset/modelNet/',
        encoder_structure=encoder_structure,
        decoder_structure=decoder_structure,
        save_path='./weights/AE3D/',
        # load_path='./weights/AE3D/',
        # load_path=None,
    ))




















