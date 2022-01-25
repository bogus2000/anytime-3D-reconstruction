import os
import tensorflow as tf
import src.net_core.autoencoder3D as ae
from src.module.function import *

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
    'name':'decoder',
    'input_dim' : 200,
    'output_shape':[64,64,64,1],
    'filter_num_list':[512,256,128,64,1],
    'filter_size_list':[4,4,4,4,4],
    'strides_list':[1,2,2,2,2],
    'activation':'elu',
    'final_activation':'sigmoid'
}

class AE3D(object):
    def __init__(self, encoder_structure, decoder_structure,
                 BATCH_SIZE_PER_REPLICA=64, strategy=None,
                 learning_rate=1e-4):
        self._enc_str = encoder_structure
        self._dec_str = decoder_structure
        self._learning_rate = learning_rate

        self._strategy = strategy
        self._BATCH_SIZE_PER_REPLICA = BATCH_SIZE_PER_REPLICA
        self._GLOBAL_BATCH_SIZE = self._BATCH_SIZE_PER_REPLICA * self._strategy.num_replicas_in_sync

        with self._strategy.scope():
            self._buildModel()
            self._optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)
            # self._optimizer = tf.keras.optimizers.SGD(learning_rate=self._learning_rate)

    # @tf.function
    def _lossObject(self, y_target, y_pred, gamma=0.60):
        loss = binary_loss(xTarget=y_target, xPred=y_pred, gamma=gamma, b_range=False)
        return tf.nn.compute_average_loss(loss, global_batch_size=self._GLOBAL_BATCH_SIZE)

    # @tf.function
    def _evaluation(self, y_target, y_pred):
        TP, FP, FN = voxelPrecisionRecall(xTarget=y_target, xPred=y_pred)
        TP = tf.nn.compute_average_loss(TP, global_batch_size=self._GLOBAL_BATCH_SIZE)
        FP = tf.nn.compute_average_loss(FP, global_batch_size=self._GLOBAL_BATCH_SIZE)
        FN = tf.nn.compute_average_loss(FN, global_batch_size=self._GLOBAL_BATCH_SIZE)
        # p = TP / (TP + FP + 1e-10)
        # r = TP / (TP + FN + 1e-10)
        # return p, r
        return TP, FP, FN

    def _buildModel(self):
        print('build Models...')
        self._encoder = ae.encoder3D(structure=self._enc_str)
        self._decoder = ae.decoder3D(structure=self._dec_str)
        print('done')

    def fit(self, inputs):
        input_images, output_images = inputs
        # ====== input voxel value range : [0,1] -> [-1,2]
        input_images = 2.0 * input_images - 1.0
        with tf.GradientTape() as tape:
            latents = self._encoder(input_images, training=True)
            output_images_pred = self._decoder(latents, training=True)
            pred_loss = self._lossObject(y_target=output_images, y_pred=output_images_pred, gamma=0.6)
            # pred_loss = tf.nn.compute_average_loss(
            #     tf.reduce_sum(
            #         tf.keras.losses.binary_crossentropy(y_true=output_images, y_pred=output_images_pred)
            #         ,axis=[1, 2, 3]
            #     )
            # )
            # pred_loss = lossObject(y_target=output_images, y_pred=output_images_pred, GLOBAL_BATCH_SIZE=self._GLOBAL_BATCH_SIZE)
            reg_loss = tf.reduce_sum(self._encoder.losses + self._decoder.losses)
            # reg_loss = pred_loss
            total_loss = pred_loss + reg_loss
        trainable_variables = self._encoder.trainable_variables + self._decoder.trainable_variables
        grads = tape.gradient(total_loss, trainable_variables)
        self._optimizer.apply_gradients(zip(grads, trainable_variables))

        TP, FP, FN = self._evaluation(y_target=output_images, y_pred=output_images_pred)
        return reg_loss, pred_loss, total_loss, TP, FP, FN

    @tf.function
    def distributed_fit(self, inputs):
        per_rep_rl, per_rep_pl, per_rep_tl, per_rep_TP, per_rep_FP, per_rep_FN = self._strategy.run(self.fit, args=(inputs,))
        rl = self._strategy.reduce(tf.distribute.ReduceOp.SUM, per_rep_rl, axis=None)
        pl = self._strategy.reduce(tf.distribute.ReduceOp.SUM, per_rep_pl, axis=None)
        tl = self._strategy.reduce(tf.distribute.ReduceOp.SUM, per_rep_tl, axis=None)
        TP = self._strategy.reduce(tf.distribute.ReduceOp.SUM, per_rep_TP, axis=None)
        FP = self._strategy.reduce(tf.distribute.ReduceOp.SUM, per_rep_FP, axis=None)
        FN = self._strategy.reduce(tf.distribute.ReduceOp.SUM, per_rep_FN, axis=None)
        # TP, FP, FN = per_rep_TP,per_rep_FP,per_rep_FN
        p = TP / (TP + FP + 1e-10)
        r = TP / (TP + FN + 1e-10)
        return rl, pl, tl, p, r

    def saveEncoder(self, save_path):
        file_name = self._enc_str['name']
        self._encoder.save_weights(os.path.join(save_path, file_name))

    def saveDecoder(self, save_path):
        file_name = self._dec_str['name']
        self._decoder.save_weights(os.path.join(save_path, file_name))

    def saveModel(self, save_path):
        self.saveEncoder(save_path=save_path)
        self.saveDecoder(save_path=save_path)

    def loadEncoder(self, load_path):
        file_name = self._enc_str['name']
        self._encoder.load_weights(os.path.join(load_path, file_name))

    def loadDecoder(self, load_path):
        file_name = self._dec_str['name']
        self._decoder.load_weights(os.path.join(load_path, file_name))

    def loadModel(self, load_path):
        self.loadEncoder(load_path=load_path)
        self.loadDecoder(load_path=load_path)


with tf.distribute.MirroredStrategy().scope():
    def lossObject(y_target, y_pred, GLOBAL_BATCH_SIZE):
        loss = binary_loss(xTarget=y_target, xPred=y_pred, gamma=0.60, b_range=False)
        return tf.nn.compute_average_loss(loss, global_batch_size=GLOBAL_BATCH_SIZE)
