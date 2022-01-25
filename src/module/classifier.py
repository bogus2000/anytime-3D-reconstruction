import numpy as np
import os
import tensorflow as tf
import src.net_core.darknet as darknet


class classifier(object):
    def __init__(self, backbone_style=None, model_backbone=None, name=None, BATCH_SIZE_PER_REPLICA=64, strategy=None,
                 output_dim=1000, is_training=True,
                 backbone_activation='elu', head_activation = 'elu',
                 last_filter_num_list=[],
                 last_filter_size_list=[],
                 last_pooling='average',
                 learning_rate=1e-4):
        if name == None:
            name = 'None'
        self._name = name
        self._backbone_style = backbone_style
        self.model_backbone = model_backbone
        self._output_dim = output_dim
        self._is_training = is_training
        self._b_act, self._h_act = backbone_activation, head_activation
        self._last_fnl = last_filter_num_list
        self._last_fsl = last_filter_size_list
        self._last_pooling = last_pooling
        self._learning_rate = learning_rate

        self._strategy = strategy
        self._BATCH_SIZE_PER_REPLICA = BATCH_SIZE_PER_REPLICA
        self._GLOBAL_BATCH_SIZE = self._BATCH_SIZE_PER_REPLICA * self._strategy.num_replicas_in_sync

        with self._strategy.scope():
            self._buildModel()
            self._optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)
            # self._optimizer = tf.keras.optimizers.SGD(learning_rate=self._learning_rate)

    @tf.function
    def _lossObject(self, y_target, y_pred):
        y_pred = tf.nn.softmax(y_pred)
        loss = -tf.reduce_sum(y_target * tf.math.log(y_pred + 1e-9), axis=-1)
        return tf.nn.compute_average_loss(loss, global_batch_size=self._GLOBAL_BATCH_SIZE)

    @tf.function
    def _evaluation(self, y_target, y_pred):
        gt = tf.argmax(y_target, axis=-1)
        pr = tf.argmax(y_pred, axis=-1)
        equality = tf.equal(pr, gt)
        acc_top1 = tf.cast(equality, tf.float32)
        acc_top5 = tf.cast(
            tf.math.in_top_k(
                predictions=y_pred,
                targets=gt, k=5
            ),
            tf.float32)
        return tf.nn.compute_average_loss(
            acc_top1, global_batch_size=self._GLOBAL_BATCH_SIZE
        ), tf.nn.compute_average_loss(
            acc_top5, global_batch_size=self._GLOBAL_BATCH_SIZE
        )

    def _buildModel(self):
        print('build Models...')
        if self.model_backbone == None:
            self.model_backbone = self._backbone_style(name=self.name+'_backbone', activation=self._b_act)
        # ================================= set model head
        self.model_head = darknet.head2D(name='head_'+self._name,
                                         input_shape=self.model_backbone.output_shape[1:],
                                         output_dim=self._output_dim,
                                         filter_num_list=self._last_fnl, filter_size_list=self._last_fsl,
                                         last_pooling=self._last_pooling, activation=self._h_act)

    # @tf.function
    def fit(self, inputs):
        input_images, output_labels_gt = inputs

        with tf.GradientTape() as tape:
            output_backbone = self.model_backbone(input_images, training=True)
            if isinstance(output_backbone, tuple):
                output_backbone = output_backbone[-1]
            else:
                output_backbone = output_backbone
            output_labels_pred = self.model_head(output_backbone, training=True)
            reg_loss = tf.reduce_sum(self.model_backbone.losses + self.model_head.losses)
            pred_loss = self._lossObject(y_target=output_labels_gt, y_pred=output_labels_pred)
            total_loss = pred_loss  # + reg_loss

        grads = tape.gradient(
            total_loss, self.model_backbone.trainable_variables + self.model_head.trainable_variables
        )

        self._optimizer.apply_gradients(
            zip(grads, self.model_backbone.trainable_variables + self.model_head.trainable_variables)
        )

        acc_top1, acc_top5 = self._evaluation(y_pred=output_labels_pred, y_target=output_labels_gt)
        return reg_loss, pred_loss, total_loss, acc_top1, acc_top5

    @tf.function
    def distributed_fit(self, inputs):
        per_rep_rl, per_rep_pl, per_rep_tl, per_rep_a1, per_rep_a5 = self._strategy.run(self.fit, args=(inputs,))
        rl = self._strategy.reduce(tf.distribute.ReduceOp.SUM, per_rep_rl, axis=None)
        pl = self._strategy.reduce(tf.distribute.ReduceOp.SUM, per_rep_pl, axis=None)
        tl = self._strategy.reduce(tf.distribute.ReduceOp.SUM, per_rep_tl, axis=None)
        a1 = self._strategy.reduce(tf.distribute.ReduceOp.SUM, per_rep_a1, axis=None)
        a5 = self._strategy.reduce(tf.distribute.ReduceOp.SUM, per_rep_a5, axis=None)
        # return rl.numpy(), pl.numpy(), tl.numpy(), a1.numpy(), a5.numpy()
        return rl, pl, tl, a1, a5

    def saveBackbone(self, save_path):
        file_name = self._name + '_backbone'
        self.model_backbone.save_weights(os.path.join(save_path, file_name))

    def saveHead(self, save_path):
        file_name = self._name + '_head'
        self.model_head.save_weights(os.path.join(save_path, file_name))

    def saveModel(self, save_path):
        self.saveBackbone(save_path=save_path)
        self.saveHead(save_path=save_path)

    def loadBackbone(self, load_path):
        file_name = self._name + '_backbone'
        self.model_backbone.load_weights(os.path.join(load_path, file_name))

    def loadHead(self, load_path):
        file_name = self._name + '_head'
        self.model_head.load_weights(os.path.join(load_path, file_name))

    def loadModel(self, load_path):
        self.loadBackbone(load_path=load_path)
        self.loadHead(load_path=load_path)

# a = imagenetClassifier(backbone=darknet.Darknet53)