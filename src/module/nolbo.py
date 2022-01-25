import src.net_core.darknet as darknet
import src.net_core.autoencoder3D as ae3D
import src.net_core.priornet as priornet
import numpy as np

from src.module.function import *

config = {
    'encoder_backbone': {
        'name': 'nolbo_backbone',
        'predictor_num': 3,
        'bbox2D_dim': 4, 'bbox3D_dim': 3, 'orientation_dim': 3,
        'inst_dim': 10, 'z_inst_dim': 16,
        'activation': 'elu',
    },
    'encoder_head': {
        'name': 'nolbo_head',
        'output_dim': 5 * (1 + 4 + 3 + (2 * 3 + 3) + 16),
        'filter_num_list': [1024, 1024, 1024, 1024],
        'filter_size_list': [3, 3, 3, 1],
        'activation': 'elu',
    },
    'decoder': {
        'name': 'decoder',
        'input_dim': 16,
        'output_shape': [64, 64, 64, 1],
        'filter_num_list': [512, 256, 128, 64, 1],
        'filter_size_list': [4, 4, 4, 4, 4],
        'strides_list': [1, 2, 2, 2, 2],
        'activation': 'elu',
        'final_activation': 'sigmoid'
    },
    'prior_class' : {
        'name' : 'priornet_class',
        'input_dim' : 10,  # class num (one-hot vector)
        'unit_num_list' : [64, 32, 16],
        'core_activation' : 'elu',
        'const_log_var' : 0.0,
    },
    'prior_inst' : {
        'name' : 'priornet_inst',
        'input_dim' : 10,  # class num (one-hot vector)
        'unit_num_list' : [64, 32, 16],
        'core_activation' : 'elu',
        'const_log_var' : 0.0,
    }
}

class nolboSingleObject(object):
    def __init__(self, nolbo_structure,
                 backbone_style=None, encoder_backbone=None,
                 # BATCH_SIZE_PER_REPLICA=64,
                 # strategy=None,
                 learning_rate=1e-4):
        self._enc_backbone_str = nolbo_structure['encoder_backbone']
        self._enc_head_str = nolbo_structure['encoder_head']
        self._dec_str = nolbo_structure['decoder']
        self._prior_class_str = nolbo_structure['prior_class']
        self._prior_inst_str = nolbo_structure['prior_inst']

        self._backbone_style = backbone_style
        self._encoder_backbone = encoder_backbone

        # self._strategy = strategy
        # self._BATCH_SIZE_PER_REPLICA = BATCH_SIZE_PER_REPLICA
        # self._GLOBAL_BATCH_SIZE = self._BATCH_SIZE_PER_REPLICA * self._strategy.num_replicas_in_sync

        # with self._strategy.scope():
        self._buildModel()
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def _buildModel(self):
        print('build Models...')
        if self._encoder_backbone == None:
            self._encoder_backbone = self._backbone_style(name=self._enc_backbone_str['name'])
        # ==============set encoder head
        self._encoder_head = darknet.head2D(name=self._enc_head_str['name'],
                                            input_shape=self._encoder_backbone.output_shape[1:],
                                            output_dim=self._enc_head_str['output_dim'],
                                            filter_num_list=self._enc_head_str['filter_num_list'],
                                            filter_size_list=self._enc_head_str['filter_size_list'],
                                            last_pooling='max', activation=self._enc_head_str['activation'])
        # ==============set decoder3D
        self._decoder = ae3D.decoder3D(structure=self._dec_str)
        # ==============set prior network
        self._priornet_class = priornet.priornet(structure=self._prior_class_str)
        self._priornet_inst = priornet.priornet(structure=self._prior_inst_str)
        print('done')

    def fit(self, inputs):
        input_images, output_images, category_list, inst_list = inputs
        input_images, output_images, category_list, inst_list = tf.convert_to_tensor(input_images), tf.convert_to_tensor(output_images), tf.convert_to_tensor(category_list), tf.convert_to_tensor(inst_list)
        with tf.GradientTape() as tape:
            # get priornet output
            self._mean_category_prior, self._log_var_category_prior = self._priornet_class(category_list, training=True)
            inst_vector = tf.concat([category_list, inst_list], axis=-1)
            self._mean_inst_prior, self._log_var_inst_prior = self._priornet_inst(inst_vector, training=True)

            # get encoder output and loss
            enc_output = self._encoder_head(self._encoder_backbone(input_images, training=True), training=True)
            part_start, part_end = 0, self._enc_backbone_str['z_category_dim']
            self._mean_category = enc_output[..., part_start:part_end]
            part_start, part_end = part_end, part_end + self._enc_backbone_str['z_category_dim']
            self._log_var_category = tf.clip_by_value(enc_output[..., part_start:part_end], clip_value_min=-10.0, clip_value_max=10.0)
            part_start, part_end = part_end, part_end + self._enc_backbone_str['z_inst_dim']
            self._mean_inst = enc_output[..., part_start:part_end]
            part_start, part_end = part_end, part_end + self._enc_backbone_str['z_inst_dim']
            self._log_var_inst = tf.clip_by_value(enc_output[..., part_start:part_end], clip_value_min=-10.0, clip_value_max=10.0)

            self._z_category = sampling(mu=self._mean_category, logVar=self._log_var_category)
            self._z_inst = sampling(mu=self._mean_inst, logVar=self._log_var_inst)
            self._z = tf.concat([self._z_category, self._z_inst], axis=-1)

            self._z_category_prior = sampling(mu=self._mean_category_prior, logVar=self._log_var_category_prior)
            self._z_inst_prior = sampling(mu=self._mean_inst_prior, logVar=self._log_var_inst_prior)
            self._z_prior = tf.concat([self._z_category_prior, self._z_inst_prior], axis=-1)

            # get (priornet, decoder) output and loss
            if np.random.rand()>0.5:
                self._output_images_pred = self._decoder(self._z, training=True)
            else:
                noise = tf.convert_to_tensor((np.random.choice(a=[False, True], size=np.array(tf.shape(self._z)), p=[0.5, 0.5])).astype('float32'))
                self._z_input = tf.where(noise==0, self._z, self._z_prior)
                self._output_images_pred = self._decoder(self._z_input, training=True)

            # get loss
            self._loss_category_kl = kl_loss(mean=self._mean_category, logVar=self._log_var_category,
                                             mean_target=self._mean_category_prior, logVar_target=self._log_var_category_prior)
            self._loss_inst_kl = kl_loss(mean=self._mean_inst, logVar=self._log_var_inst,
                                             mean_target=self._mean_inst_prior, logVar_target=self._log_var_inst_prior)
            self._loss_kl =tf.reduce_mean(self._loss_category_kl + self._loss_inst_kl, axis=0)

            self._loss_shape = binary_loss(xPred=self._output_images_pred, xTarget=output_images, gamma=0.60, b_range=False)
            self._loss_shape = tf.reduce_mean(self._loss_shape, axis=0)

            self._loss_reg_category = regulizer_loss(z_mean=self._mean_category_prior, z_logVar=self._log_var_category_prior,
                                                     dist_in_z_space=3.0*self._enc_backbone_str['z_category_dim'])
            self._loss_reg_inst = regulizer_loss(z_mean=self._mean_inst_prior, z_logVar=self._log_var_inst_prior,
                                                 dist_in_z_space=3.0*self._enc_backbone_str['z_inst_dim'], class_input=category_list)
            self._loss_reg = tf.reduce_mean(self._loss_reg_category + self._loss_reg_inst, axis=0)

            loss_net_reg = tf.reduce_sum(self._encoder_head.losses + self._encoder_backbone.losses
                                         + self._decoder.losses
                                         + self._priornet_class.losses + self._priornet_inst.losses)

            # total loss
            total_loss = (self._loss_kl + self._loss_shape + self._loss_reg + loss_net_reg)

        trainable_variables = self._encoder_backbone.trainable_variables + self._encoder_head.trainable_variables \
                            + self._decoder.trainable_variables \
                              + self._priornet_class.trainable_variables + self._priornet_inst.trainable_variables
        grads = tape.gradient(total_loss, trainable_variables)
        self._optimizer.apply_gradients(zip(grads, trainable_variables))

        TP, FP, FN = voxelPrecisionRecall(xTarget=output_images, xPred=self._output_images_pred)
        pr = tf.reduce_mean(TP / (TP + FP + 1e-10), axis=0)
        rc = tf.reduce_mean(TP / (TP + FN + 1e-10), axis=0)

        return self._loss_kl, self._loss_shape, self._loss_reg, pr, rc

    def getEval(self, inputs, category_indices=np.identity(12), inst_indices=np.identity(10), training=False, missing_prob=0.0):
        input_images, output_images, category_list, inst_list = inputs
        batch_num = len(input_images)
        category_num = len(category_indices)
        inst_num = len(inst_indices)
        inst_indices_input = []
        for cat_l in category_list:
            for ins_ind in inst_indices:
                vec = np.concatenate([cat_l, ins_ind], axis=-1)
                inst_indices_input.append(vec)
        inst_indices_input = tf.convert_to_tensor(np.array(inst_indices_input))
        input_images, output_images, category_list, inst_list, category_indices, inst_indices = tf.convert_to_tensor(
            input_images), tf.convert_to_tensor(output_images), tf.convert_to_tensor(
            category_list), tf.convert_to_tensor(inst_list), tf.convert_to_tensor(category_indices), tf.convert_to_tensor(inst_indices)
        # get priornet output
        self._mean_category_prior, self._log_var_category_prior = self._priornet_class(category_indices, training=training)
        self._mean_category_prior_tile = tf.reshape(self._mean_category_prior, (1, category_num, self._enc_backbone_str['z_category_dim']))
        self._mean_category_prior_tile = tf.tile(self._mean_category_prior_tile, (batch_num, 1, 1))
        # inst_vector = tf.concat([category_list, inst_list], axis=-1)
        self._mean_inst_prior, self._log_var_inst_prior = self._priornet_inst(inst_indices_input, training=training)
        self._mean_inst_prior_tile = tf.reshape(self._mean_inst_prior, (batch_num, inst_num, -1))

        # get encoder output and loss
        enc_output = self._encoder_head(self._encoder_backbone(input_images, training=training), training=training)
        part_start, part_end = 0, self._enc_backbone_str['z_category_dim']
        self._mean_category = enc_output[..., part_start:part_end]
        part_start, part_end = part_end, part_end + self._enc_backbone_str['z_category_dim']
        self._log_var_category = tf.clip_by_value(enc_output[..., part_start:part_end], clip_value_min=-10.0,
                                                  clip_value_max=10.0)
        part_start, part_end = part_end, part_end + self._enc_backbone_str['z_inst_dim']
        self._mean_inst = enc_output[..., part_start:part_end]
        part_start, part_end = part_end, part_end + self._enc_backbone_str['z_inst_dim']
        self._log_var_inst = tf.clip_by_value(enc_output[..., part_start:part_end], clip_value_min=-10.0,
                                              clip_value_max=10.0)

        self._z_category = sampling(mu=self._mean_category, logVar=self._log_var_category)
        self._z_inst = sampling(mu=self._mean_inst, logVar=self._log_var_inst)

        if missing_prob>0:
            mean_category_prior_mean = tf.reduce_mean(self._mean_category_prior, axis=0)
            z_dim = tf.shape(self._z_category).numpy()[-1]
            mask = np.random.choice(2, batch_num * z_dim, p=[missing_prob, 1.- missing_prob])
            mask = tf.convert_to_tensor(np.reshape(mask, [batch_num, z_dim]).astype('float32'))
            self._z_category = self._z_category * mask
            # mu, var = 0.0, 1.0 ** 2
            # noise = sampling(mu= mu * tf.ones_like(self._z_category), logVar=tf.math.log( var * tf.ones_like(self._z_category)))
            # self._z_category = tf.where(self._z_category == 0, noise, self._z_category)
            self._z_category = tf.where(self._z_category == 0, mean_category_prior_mean * tf.ones_like(self._z_category), self._z_category)
        else:
            mask = tf.ones_like(self._z_category)
        mask_tile = tf.reshape(mask, (batch_num, 1, self._enc_backbone_str['z_category_dim']))
        mask_tile = tf.tile(mask_tile, (1, category_num, 1))

        # classification
        z_category_tile = tf.reshape(self._z_category, (batch_num, 1, self._enc_backbone_str['z_category_dim']))
        z_category_tile = tf.tile(z_category_tile, (1, category_num, 1))
        mean_category_prior_tile = self._mean_category_prior_tile
        dist_category = tf.reduce_sum(tf.square(z_category_tile - mean_category_prior_tile), axis=-1)
        cat_equal = tf.equal(tf.argmin(dist_category, axis=-1), tf.argmax(category_list, axis=-1))
        acc_cat = tf.reduce_mean(tf.cast(cat_equal, tf.float32), axis=0)

        z_inst_tile = tf.reshape(self._z_inst, (batch_num, 1, self._enc_backbone_str['z_inst_dim']))
        z_inst_tile = tf.tile(z_inst_tile, (1, inst_num, 1))
        mean_inst_prior_tile = self._mean_inst_prior_tile
        dist_inst = tf.reduce_sum(tf.square(z_inst_tile - mean_inst_prior_tile), axis=-1)
        inst_equal = tf.equal(tf.argmin(dist_inst, axis=-1), tf.argmax(inst_list, axis=-1))
        acc_inst = tf.reduce_mean(tf.cast(inst_equal, tf.float32), axis=0)

        self._z = tf.concat([self._z_category, self._z_inst], axis=-1)
        output_images_pred = self._decoder(self._z, training=training)
        loss_shape = binary_loss(xPred=output_images_pred, xTarget=output_images, gamma=0.60, b_range=False)
        loss_shape = tf.reduce_mean(loss_shape, axis=0)
        TP, FP, FN = voxelPrecisionRecall(xTarget=output_images, xPred=output_images_pred)
        pr = tf.reduce_mean(TP / (TP + FP + 1e-10), axis=0)
        rc = tf.reduce_mean(TP / (TP + FN + 1e-10), axis=0)
        if missing_prob == 0.0:
            return output_images_pred, loss_shape, pr, rc, acc_cat, acc_inst
        else:
            dist_category = tf.reduce_sum(mask_tile * tf.square(z_category_tile - mean_category_prior_tile), axis=-1)
            min_idx = tf.reshape(tf.argmin(dist_category, axis=-1), (-1, 1))
            mean_category_prior_distmin = tf.gather_nd(self._mean_category_prior_tile, min_idx, batch_dims=1)
            z_category_prior = sampling(mu=mean_category_prior_distmin, logVar=tf.zeros_like(mean_category_prior_distmin))
            self._z_category_corrected = tf.where(mask == 0, z_category_prior, self._z_category)
            # classification
            z_category_tile = tf.reshape(self._z_category_corrected, (batch_num, 1, self._enc_backbone_str['z_category_dim']))
            z_category_tile = tf.tile(z_category_tile, (1, category_num, 1))
            mean_category_prior_tile = self._mean_category_prior_tile
            dist_category = tf.reduce_sum(tf.square(z_category_tile - mean_category_prior_tile), axis=-1)
            cat_equal = tf.equal(tf.argmin(dist_category, axis=-1), tf.argmax(category_list, axis=-1))
            acc_cat_corrected = tf.reduce_mean(tf.cast(cat_equal, tf.float32), axis=0)

            self._z_corrected = tf.concat([self._z_category_corrected, self._z_inst], axis=-1)
            output_images_pred_corrected = self._decoder(self._z_corrected, training=training)
            loss_shape_corrected = binary_loss(xPred=output_images_pred_corrected, xTarget=output_images, gamma=0.60, b_range=False)
            loss_shape_corrected = tf.reduce_mean(loss_shape_corrected, axis=0)
            TP_corrected, FP_corrected, FN_corrected = voxelPrecisionRecall(xTarget=output_images, xPred=output_images_pred_corrected)
            pr_corrected = tf.reduce_mean(TP_corrected / (TP_corrected + FP_corrected + 1e-10), axis=0)
            rc_corrected = tf.reduce_mean(TP_corrected / (TP_corrected + FN_corrected + 1e-10), axis=0)
            return output_images_pred, loss_shape, pr, rc, acc_cat, acc_inst, output_images_pred_corrected, loss_shape_corrected, pr_corrected, rc_corrected, acc_cat_corrected

    def saveEncoderBackbone(self, save_path):
        file_name = self._enc_backbone_str['name']
        self._encoder_backbone.save_weights(os.path.join(save_path, file_name))

    def saveEncoderHead(self, save_path):
        file_name = self._enc_head_str['name']
        self._encoder_head.save_weights(os.path.join(save_path, file_name))

    def saveEncoder(self, save_path):
        self.saveEncoderBackbone(save_path=save_path)
        self.saveEncoderHead(save_path=save_path)

    def saveDecoder(self, save_path):
        file_name = self._dec_str['name']
        self._decoder.save_weights(os.path.join(save_path, file_name))

    def savePriorCategory(self, save_path):
        file_name = self._prior_class_str['name']
        self._priornet_class.save_weights(os.path.join(save_path, file_name))

    def savePriorInst(self, save_path):
        file_name = self._prior_inst_str['name']
        self._priornet_inst.save_weights(os.path.join(save_path, file_name))

    def saveModel(self, save_path):
        self.saveEncoder(save_path=save_path)
        self.saveDecoder(save_path=save_path)
        self.savePriorCategory(save_path=save_path)
        self.savePriorInst(save_path=save_path)

    def loadEncoderBackbone(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._enc_backbone_str['name']
        self._encoder_backbone.load_weights(os.path.join(load_path, file_name))

    def loadEncoderHead(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._enc_head_str['name']
        self._encoder_head.load_weights(os.path.join(load_path, file_name))

    def loadEncoder(self, load_path):
        self.loadEncoderBackbone(load_path=load_path)
        self.loadEncoderHead(load_path=load_path)

    def loadDecoder(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._dec_str['name']
        self._decoder.load_weights(os.path.join(load_path, file_name))

    def loadPriorCategory(self, load_path, file_name=None):
        if file_name is None:
            file_name = self._prior_class_str['name']
        self._priornet_class.load_weights(os.path.join(load_path, file_name))

    def loadPriorInst(self, load_path, file_name=None):
        if file_name is None:
            file_name = self._prior_inst_str['name']
        self._priornet_inst.load_weights(os.path.join(load_path, file_name))

    def loadModel(self, load_path):
        self.loadEncoder(load_path=load_path)
        self.loadDecoder(load_path=load_path)
        self.loadPriorCategory(load_path=load_path)
        self.loadPriorInst(load_path=load_path)

class nolboSingleObject_instOnly(object):
    def __init__(self, nolbo_structure,
                 backbone_style=None, encoder_backbone=None,
                 # BATCH_SIZE_PER_REPLICA=64,
                 # strategy=None,
                 learning_rate=1e-4):
        self._enc_backbone_str = nolbo_structure['encoder_backbone']
        self._enc_head_str = nolbo_structure['encoder_head']
        self._dec_str = nolbo_structure['decoder']
        self._prior_inst_str = nolbo_structure['prior_inst']

        self._backbone_style = backbone_style
        self._encoder_backbone = encoder_backbone

        # self._strategy = strategy
        # self._BATCH_SIZE_PER_REPLICA = BATCH_SIZE_PER_REPLICA
        # self._GLOBAL_BATCH_SIZE = self._BATCH_SIZE_PER_REPLICA * self._strategy.num_replicas_in_sync

        # with self._strategy.scope():
        self._buildModel()
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def _buildModel(self):
        print('build Models...')
        if self._encoder_backbone == None:
            self._encoder_backbone = self._backbone_style(name=self._enc_backbone_str['name'])
        # ==============set encoder head
        self._encoder_head = darknet.head2D(name=self._enc_head_str['name'],
                                            input_shape=self._encoder_backbone.output_shape[1:],
                                            output_dim=self._enc_head_str['output_dim'],
                                            filter_num_list=self._enc_head_str['filter_num_list'],
                                            filter_size_list=self._enc_head_str['filter_size_list'],
                                            last_pooling='max', activation=self._enc_head_str['activation'])
        # ==============set decoder3D
        self._decoder = ae3D.decoder3D(structure=self._dec_str)
        # ==============set prior network
        self._priornet_inst = priornet.priornet(structure=self._prior_inst_str)
        print('done')

    def fit(self, inputs):
        input_images, output_images, inst_list = inputs
        input_images, output_images, inst_list = tf.convert_to_tensor(input_images), tf.convert_to_tensor(output_images), tf.convert_to_tensor(inst_list)
        with tf.GradientTape() as tape:
            # get priornet output
            self._mean_inst_prior, self._log_var_inst_prior = self._priornet_inst(inst_list, training=True)

            # get encoder output and loss
            enc_output = self._encoder_head(self._encoder_backbone(input_images, training=True), training=True)
            part_start, part_end = 0, self._enc_backbone_str['z_inst_dim']
            self._mean_inst = enc_output[..., part_start:part_end]
            part_start, part_end = part_end, part_end + self._enc_backbone_str['z_inst_dim']
            self._log_var_inst = tf.clip_by_value(enc_output[..., part_start:part_end], clip_value_min=-10.0, clip_value_max=10)

            self._z = sampling(mu=self._mean_inst, logVar=self._log_var_inst)
            self._z_prior = sampling(mu=self._mean_inst_prior, logVar=self._log_var_inst_prior)

            # get (priornet, decoder) output and loss
            if np.random.rand() < 0.5:
                self._output_images_pred = self._decoder(self._z, training=True)
            else:
                self._output_images_pred = self._decoder(self._z_prior, training=True)

            # get loss
            self._loss_inst_kl = kl_loss(mean=self._mean_inst, logVar=self._log_var_inst, mean_target=self._mean_inst_prior, logVar_target=self._log_var_inst_prior)
            self._loss_kl =tf.reduce_mean(self._loss_inst_kl, axis=0)

            self._loss_shape = binary_loss(xPred=self._output_images_pred, xTarget=output_images, gamma=0.60, b_range=False)
            self._loss_shape = tf.reduce_mean(self._loss_shape, axis=0)

            self._loss_reg_inst = regulizer_loss(z_mean=self._mean_inst_prior, z_logVar=self._log_var_inst_prior, dist_in_z_space=10.0*self._enc_backbone_str['z_inst_dim'])
            self._loss_reg = tf.reduce_mean(self._loss_reg_inst, axis=0)

            loss_net_reg = tf.reduce_sum(self._encoder_head.losses + self._encoder_backbone.losses
                                         + self._decoder.losses
                                         + self._priornet_inst.losses)

            # total loss
            total_loss = (self._loss_kl + self._loss_shape + self._loss_reg + loss_net_reg)

        trainable_variables = self._encoder_backbone.trainable_variables + self._encoder_head.trainable_variables \
                              + self._decoder.trainable_variables \
                              + self._priornet_inst.trainable_variables
        grads = tape.gradient(total_loss, trainable_variables)
        self._optimizer.apply_gradients(zip(grads, trainable_variables))

        TP, FP, FN = voxelPrecisionRecall(xTarget=output_images, xPred=self._output_images_pred)
        pr = tf.reduce_mean(TP / (TP + FP + 1e-10), axis=0)
        rc = tf.reduce_mean(TP / (TP + FN + 1e-10), axis=0)

        return self._loss_kl, self._loss_shape, self._loss_reg, pr, rc

    def getEval(self, input_images, output_images, inst_list=np.identity(10), missing_prob = 0.1, training=True):
        batch_num = len(input_images)
        inst_num = len(inst_list)
        input_images, output_images, inst_list = tf.convert_to_tensor(input_images), tf.convert_to_tensor(output_images), tf.convert_to_tensor(inst_list)
        self._mean_inst_prior, self._log_var_inst_prior = self._priornet_inst(inst_list, training=training)
        # print(self._mean_inst_prior)

        enc_output = self._encoder_head(self._encoder_backbone(input_images, training=training), training=training)
        part_start, part_end = 0, self._enc_backbone_str['z_inst_dim']
        self._mean_inst = enc_output[..., part_start:part_end]
        part_start, part_end = part_end, part_end + self._enc_backbone_str['z_inst_dim']
        self._log_var_inst = tf.clip_by_value(enc_output[..., part_start:part_end], clip_value_min=-10.0, clip_value_max=10)
        self._z = sampling(mu=self._mean_inst, logVar=self._log_var_inst)

        if missing_prob>0:
            z_dim = tf.shape(self._z).numpy()[-1]
            mask = np.random.choice(2, batch_num * z_dim, p=[missing_prob, 1.- missing_prob])
            mask = tf.convert_to_tensor(np.reshape(mask, [batch_num, z_dim]).astype('float32'))
            self._z = self._z * mask
            # print(self._z)
            mu, var = 0.0, 1.0 ** 2
            noise = sampling(mu= mu * tf.ones_like(self._z), logVar=tf.math.log( var * tf.ones_like(self._z)))
            self._z = tf.where(self._z == 0, noise, self._z)
        else:
            mask = tf.ones_like(self._z)

        output_images_pred = self._decoder(self._z, training=training)

        loss_shape = binary_loss(xPred=output_images_pred, xTarget=output_images, gamma=0.60, b_range=False)
        loss_shape = tf.reduce_mean(loss_shape, axis=0)
        TP, FP, FN = voxelPrecisionRecall(xTarget=output_images, xPred=output_images_pred)
        pr = tf.reduce_mean(TP / (TP + FP + 1e-10), axis=0)
        rc = tf.reduce_mean(TP / (TP + FN + 1e-10), axis=0)

        mask_tile = tf.reshape(mask, (batch_num, 1, -1))
        mask_tile = tf.tile(mask_tile, (1, inst_num, 1))
        self._mean_inst_prior_tile = tf.reshape(self._mean_inst_prior, (1, inst_num, -1))
        self._mean_inst_prior_tile = tf.tile(self._mean_inst_prior_tile, (batch_num, 1, 1))
        # print(self._mean_inst_prior_tile.shape)
        self._z_tile = tf.reshape(self._z, (batch_num, 1, -1))
        self._z_tile = tf.tile(self._z_tile, (1, inst_num, 1))
        # print(self._z_tile.shape)

        dist = tf.reduce_sum(mask_tile*tf.square(self._z_tile - self._mean_inst_prior_tile), axis=-1)
        # print(dist.shape)
        min_idx = tf.argmin(dist, axis=-1)
        print(min_idx)
        min_idx = tf.reshape(min_idx, (-1, 1))
        # print(min_idx.shape)
        min_inst_prior_distmin = self._mean_inst_prior_tile  # batch, instnum, zdim
        mean_inst_prior_distmin = tf.gather_nd(min_inst_prior_distmin, min_idx, batch_dims=1)  # (batch, zdim)
        # print(mean_inst_prior_distmin.shape)
        # self._z_corrected = tf.where(self._z == -100.0, mean_inst_prior_distmin, self._z)
        z_prior = sampling(mu=mean_inst_prior_distmin, logVar=tf.zeros_like(mean_inst_prior_distmin))
        # self._z_corrected = ((missing_prob)*z_prior + (1.0-missing_prob)*self._z)
        # self._z_corrected = ((missing_prob) * mean_inst_prior_distmin + (1.0 - missing_prob) * self._z)
        # self._z_corrected = z_prior
        self._z_corrected = mean_inst_prior_distmin
        # print(self._z_corrected.shape)

        output_images_pred_corrected = self._decoder(self._z_corrected, training=training)

        loss_shape_corrected = binary_loss(xPred=output_images_pred_corrected, xTarget=output_images, gamma=0.60, b_range=False)
        loss_shape_corrected = tf.reduce_mean(loss_shape_corrected, axis=0)
        TP_corrected, FP_corrected, FN_corrected = voxelPrecisionRecall(xTarget=output_images, xPred=output_images_pred_corrected)
        pr_corrected = tf.reduce_mean(TP_corrected / (TP_corrected + FP_corrected + 1e-10), axis=0)
        rc_corrected = tf.reduce_mean(TP_corrected / (TP_corrected + FN_corrected + 1e-10), axis=0)

        return output_images_pred, loss_shape, pr, rc, output_images_pred_corrected, loss_shape_corrected, pr_corrected, rc_corrected

    def saveEncoderBackbone(self, save_path):
        file_name = self._enc_backbone_str['name']
        self._encoder_backbone.save_weights(os.path.join(save_path, file_name))

    def saveEncoderHead(self, save_path):
        file_name = self._enc_head_str['name']
        self._encoder_head.save_weights(os.path.join(save_path, file_name))

    def saveEncoder(self, save_path):
        self.saveEncoderBackbone(save_path=save_path)
        self.saveEncoderHead(save_path=save_path)

    def saveDecoder(self, save_path):
        file_name = self._dec_str['name']
        self._decoder.save_weights(os.path.join(save_path, file_name))

    def savePriorInst(self, save_path):
        file_name = self._prior_inst_str['name']
        self._priornet_inst.save_weights(os.path.join(save_path, file_name))

    def saveModel(self, save_path):
        self.saveEncoder(save_path=save_path)
        self.saveDecoder(save_path=save_path)
        self.savePriorInst(save_path=save_path)

    def loadEncoderBackbone(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._enc_backbone_str['name']
        self._encoder_backbone.load_weights(os.path.join(load_path, file_name))

    def loadEncoderHead(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._enc_head_str['name']
        self._encoder_head.load_weights(os.path.join(load_path, file_name))

    def loadEncoder(self, load_path):
        self.loadEncoderBackbone(load_path=load_path)
        self.loadEncoderHead(load_path=load_path)

    def loadDecoder(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._dec_str['name']
        self._decoder.load_weights(os.path.join(load_path, file_name))

    def loadPriorInst(self, load_path, file_name=None):
        if file_name is None:
            file_name = self._prior_inst_str['name']
        self._priornet_inst.load_weights(os.path.join(load_path, file_name))

    def loadModel(self, load_path):
        self.loadEncoder(load_path=load_path)
        self.loadDecoder(load_path=load_path)
        self.loadPriorInst(load_path=load_path)

class nolboSingleObject_AE(object):
    def __init__(self, nolbo_structure,
                 backbone_style=None, encoder_backbone=None,
                 dropout=False,
                 # BATCH_SIZE_PER_REPLICA=64,
                 # strategy=None,
                 learning_rate=1e-4):
        self._enc_backbone_str = nolbo_structure['encoder_backbone']
        self._enc_head_str = nolbo_structure['encoder_head']
        self._dec_str = nolbo_structure['decoder']

        self._backbone_style = backbone_style
        self._encoder_backbone = encoder_backbone

        self._dropout = dropout

        # self._strategy = strategy
        # self._BATCH_SIZE_PER_REPLICA = BATCH_SIZE_PER_REPLICA
        # self._GLOBAL_BATCH_SIZE = self._BATCH_SIZE_PER_REPLICA * self._strategy.num_replicas_in_sync

        # with self._strategy.scope():
        self._buildModel()
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def _buildModel(self):
        print('build Models...')
        if self._encoder_backbone == None:
            self._encoder_backbone = self._backbone_style(name=self._enc_backbone_str['name'])
        # ==============set encoder head
        self._encoder_head = darknet.head2D(name=self._enc_head_str['name'],
                                            input_shape=self._encoder_backbone.output_shape[1:],
                                            output_dim=self._enc_head_str['output_dim'],
                                            filter_num_list=self._enc_head_str['filter_num_list'],
                                            filter_size_list=self._enc_head_str['filter_size_list'],
                                            last_pooling='max', activation=self._enc_head_str['activation'])
        # ==============set decoder3D
        self._decoder = ae3D.decoder3D(structure=self._dec_str)
        print('done')

    def fit(self, inputs):
        input_images, output_images = inputs
        input_images, output_images = tf.convert_to_tensor(input_images), tf.convert_to_tensor(output_images)
        with tf.GradientTape() as tape:
            # get encoder output and loss
            self._z = self._encoder_head(self._encoder_backbone(input_images, training=True), training=True)
            if self._dropout:
                dropout_rate = np.random.rand()
                self._z = tf.keras.layers.Dropout(rate=dropout_rate)(self._z, training=True)
            self._output_images_pred = self._decoder(self._z, training=True)

            # get loss
            self._loss_shape = binary_loss(xPred=self._output_images_pred, xTarget=output_images, gamma=0.60, b_range=False)
            self._loss_shape = tf.reduce_mean(self._loss_shape, axis=0)

            loss_net_reg = tf.reduce_sum(self._encoder_head.losses + self._encoder_backbone.losses
                                         + self._decoder.losses)

            # total loss
            total_loss = (self._loss_shape + loss_net_reg)

        trainable_variables = self._encoder_backbone.trainable_variables + self._encoder_head.trainable_variables \
                            + self._decoder.trainable_variables
        grads = tape.gradient(total_loss, trainable_variables)
        self._optimizer.apply_gradients(zip(grads, trainable_variables))

        TP, FP, FN = voxelPrecisionRecall(xTarget=output_images, xPred=self._output_images_pred)
        pr = tf.reduce_mean(TP / (TP + FP + 1e-10), axis=0)
        rc = tf.reduce_mean(TP / (TP + FN + 1e-10), axis=0)

        return self._loss_shape, pr, rc

    # def getEval(self, inputs, training=False, missing_prob=0.0):
    #     input_images, output_images = inputs
    #     batch_num = len(input_images)
    #     input_images, output_images = tf.convert_to_tensor(input_images), tf.convert_to_tensor(output_images)
    #     # get encoder output and loss
    #     self._z = self._encoder_head(self._encoder_backbone(input_images, training=training), training=training)
    #
    #     if missing_prob>0:
    #         z_dim = tf.shape(self._z).numpy()[-1]
    #         mask = np.random.choice(2, batch_num * z_dim, p=[missing_prob, 1.- missing_prob])
    #         mask = tf.convert_to_tensor(np.reshape(mask, [batch_num, z_dim]).astype('float32'))
    #         self._z = tf.where(mask == 0, 0.0 * tf.ones_like(self._z), self._z)
    #
    #     output_images_pred = self._decoder(self._z, training=training)
    #     loss_shape = binary_loss(xPred=output_images_pred, xTarget=output_images, gamma=0.60, b_range=False)
    #     loss_shape = tf.reduce_mean(loss_shape, axis=0)
    #     TP, FP, FN = voxelPrecisionRecall(xTarget=output_images, xPred=output_images_pred)
    #     pr = tf.reduce_mean(TP / (TP + FP + 1e-10), axis=0)
    #     rc = tf.reduce_mean(TP / (TP + FN + 1e-10), axis=0)
    #     return output_images_pred, loss_shape, pr, rc
    def getEval(self, inputs, category_vectors, training=False, missing_prob=0.0):
        input_images, output_images, category_list = inputs
        batch_num = len(input_images)
        category_num = len(category_vectors)
        input_images, output_images, category_list, category_vectors = tf.convert_to_tensor(
            input_images), tf.convert_to_tensor(output_images), tf.convert_to_tensor(
            category_list), tf.convert_to_tensor(category_vectors)
        # get priornet output
        self._mean_category_prior = category_vectors
        self._mean_category_prior_tile = tf.reshape(self._mean_category_prior, (1, category_num, self._enc_backbone_str['z_dim']))
        self._mean_category_prior_tile = tf.tile(self._mean_category_prior_tile, (batch_num, 1, 1))

        # get encoder output and loss
        self._z_category = self._encoder_head(self._encoder_backbone(input_images, training=training), training=training)

        if missing_prob > 0:
            mean_category_prior_mean = tf.reduce_mean(self._mean_category_prior, axis=0)
            z_dim = tf.shape(self._z_category).numpy()[-1]
            mask = np.random.choice(2, batch_num * z_dim, p=[missing_prob, 1. - missing_prob])
            mask = tf.convert_to_tensor(np.reshape(mask, [batch_num, z_dim]).astype('float32'))
            self._z_category = self._z_category * mask
            # mu, var = 0.0, 1.0 ** 2
            # noise = sampling(mu= mu * tf.ones_like(self._z_category), logVar=tf.math.log( var * tf.ones_like(self._z_category)))
            # self._z_category = tf.where(self._z_category == 0, noise, self._z_category)
            self._z_category = tf.where(self._z_category == 0, mean_category_prior_mean * tf.ones_like(self._z_category), self._z_category)
        else:
            mask = tf.ones_like(self._z_category)
        mask_tile = tf.reshape(mask, (batch_num, 1, self._enc_backbone_str['z_dim']))
        mask_tile = tf.tile(mask_tile, (1, category_num, 1))

        # classification
        z_category_tile = tf.reshape(self._z_category, (batch_num, 1, self._enc_backbone_str['z_dim']))
        z_category_tile = tf.tile(z_category_tile, (1, category_num, 1))
        mean_category_prior_tile = self._mean_category_prior_tile
        dist_category = tf.reduce_sum(tf.square(z_category_tile - mean_category_prior_tile), axis=-1)
        cat_equal = tf.equal(tf.argmin(dist_category, axis=-1), tf.argmax(category_list, axis=-1))
        acc_cat = tf.reduce_mean(tf.cast(cat_equal, tf.float32), axis=0)

        output_images_pred = self._decoder(self._z_category, training=training)
        loss_shape = binary_loss(xPred=output_images_pred, xTarget=output_images, gamma=0.60, b_range=False)
        loss_shape = tf.reduce_mean(loss_shape, axis=0)
        TP, FP, FN = voxelPrecisionRecall(xTarget=output_images, xPred=output_images_pred)
        pr = tf.reduce_mean(TP / (TP + FP + 1e-10), axis=0)
        rc = tf.reduce_mean(TP / (TP + FN + 1e-10), axis=0)
        if missing_prob == 0.0:
            return output_images_pred, loss_shape, pr, rc, acc_cat, 0, 0, 0, 0, 0
        else:
            dist_category = tf.reduce_sum(mask_tile * tf.square(z_category_tile - mean_category_prior_tile), axis=-1)
            min_idx = tf.reshape(tf.argmin(dist_category, axis=-1), (-1, 1))
            mean_category_prior_distmin = tf.gather_nd(self._mean_category_prior_tile, min_idx, batch_dims=1)
            z_category_prior = sampling(mu=mean_category_prior_distmin, logVar=tf.zeros_like(mean_category_prior_distmin))
            self._z_category_corrected = tf.where(mask == 0, z_category_prior, self._z_category)
            # classification
            z_category_tile = tf.reshape(self._z_category_corrected, (batch_num, 1, self._enc_backbone_str['z_dim']))
            z_category_tile = tf.tile(z_category_tile, (1, category_num, 1))
            mean_category_prior_tile = self._mean_category_prior_tile
            dist_category = tf.reduce_sum(tf.square(z_category_tile - mean_category_prior_tile), axis=-1)
            cat_equal = tf.equal(tf.argmin(dist_category, axis=-1), tf.argmax(category_list, axis=-1))
            acc_cat_corrected = tf.reduce_mean(tf.cast(cat_equal, tf.float32), axis=0)

            output_images_pred_corrected = self._decoder(self._z_category_corrected, training=training)
            loss_shape_corrected = binary_loss(xPred=output_images_pred_corrected, xTarget=output_images, gamma=0.60, b_range=False)
            loss_shape_corrected = tf.reduce_mean(loss_shape_corrected, axis=0)
            TP_corrected, FP_corrected, FN_corrected = voxelPrecisionRecall(xTarget=output_images, xPred=output_images_pred_corrected)
            pr_corrected = tf.reduce_mean(TP_corrected / (TP_corrected + FP_corrected + 1e-10), axis=0)
            rc_corrected = tf.reduce_mean(TP_corrected / (TP_corrected + FN_corrected + 1e-10), axis=0)
            return output_images_pred, loss_shape, pr, rc, acc_cat, output_images_pred_corrected, loss_shape_corrected, pr_corrected, rc_corrected, acc_cat_corrected


    def getLatent(self, inputs):
        input_images = tf.convert_to_tensor(inputs)
        latents = self._encoder_head(self._encoder_backbone(input_images, training=False), training=False)
        return np.array(latents)


    def saveEncoderBackbone(self, save_path):
        file_name = self._enc_backbone_str['name']
        self._encoder_backbone.save_weights(os.path.join(save_path, file_name))

    def saveEncoderHead(self, save_path):
        file_name = self._enc_head_str['name']
        self._encoder_head.save_weights(os.path.join(save_path, file_name))

    def saveEncoder(self, save_path):
        self.saveEncoderBackbone(save_path=save_path)
        self.saveEncoderHead(save_path=save_path)

    def saveDecoder(self, save_path):
        file_name = self._dec_str['name']
        self._decoder.save_weights(os.path.join(save_path, file_name))

    def saveModel(self, save_path):
        self.saveEncoder(save_path=save_path)
        self.saveDecoder(save_path=save_path)

    def loadEncoderBackbone(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._enc_backbone_str['name']
        self._encoder_backbone.load_weights(os.path.join(load_path, file_name))

    def loadEncoderHead(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._enc_head_str['name']
        self._encoder_head.load_weights(os.path.join(load_path, file_name))

    def loadEncoder(self, load_path):
        self.loadEncoderBackbone(load_path=load_path)
        self.loadEncoderHead(load_path=load_path)

    def loadDecoder(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._dec_str['name']
        self._decoder.load_weights(os.path.join(load_path, file_name))

    def loadModel(self, load_path):
        self.loadEncoder(load_path=load_path)
        self.loadDecoder(load_path=load_path)

class nolboSingleObject_VAE(object):
    def __init__(self, nolbo_structure,
                 backbone_style=None, encoder_backbone=None,
                 dropout=False,
                 # BATCH_SIZE_PER_REPLICA=64,
                 # strategy=None,
                 learning_rate=1e-4):
        self._enc_backbone_str = nolbo_structure['encoder_backbone']
        self._enc_head_str = nolbo_structure['encoder_head']
        self._dec_str = nolbo_structure['decoder']

        self._backbone_style = backbone_style
        self._encoder_backbone = encoder_backbone

        self._dropout = dropout

        # self._strategy = strategy
        # self._BATCH_SIZE_PER_REPLICA = BATCH_SIZE_PER_REPLICA
        # self._GLOBAL_BATCH_SIZE = self._BATCH_SIZE_PER_REPLICA * self._strategy.num_replicas_in_sync

        # with self._strategy.scope():
        self._buildModel()
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def _buildModel(self):
        print('build Models...')
        if self._encoder_backbone == None:
            self._encoder_backbone = self._backbone_style(name=self._enc_backbone_str['name'])
        # ==============set encoder head
        self._encoder_head = darknet.head2D(name=self._enc_head_str['name'],
                                            input_shape=self._encoder_backbone.output_shape[1:],
                                            output_dim=self._enc_head_str['output_dim'],
                                            filter_num_list=self._enc_head_str['filter_num_list'],
                                            filter_size_list=self._enc_head_str['filter_size_list'],
                                            last_pooling='max', activation=self._enc_head_str['activation'])
        # ==============set decoder3D
        self._decoder = ae3D.decoder3D(structure=self._dec_str)
        print('done')

    def fit(self, inputs):
        input_images, output_images = inputs
        input_images, output_images = tf.convert_to_tensor(input_images), tf.convert_to_tensor(output_images)
        with tf.GradientTape() as tape:
            # get encoder output and loss
            enc_output = self._encoder_head(self._encoder_backbone(input_images, training=True), training=True)
            part_start, part_end = 0, self._enc_backbone_str['z_dim']
            self._mean = enc_output[..., part_start:part_end]
            part_start, part_end = part_end, part_end + self._enc_backbone_str['z_dim']
            self._log_var = tf.clip_by_value(enc_output[..., part_start:], clip_value_min=-10.0, clip_value_max=10.0)

            self._z = sampling(mu=self._mean, logVar=self._log_var)
            if self._dropout:
                dropout_rate = np.random.rand()
                self._z = tf.keras.layers.Dropout(rate=dropout_rate)(self._z, training=True)
            self._output_images_pred = self._decoder(self._z, training=True)

            # get loss
            self._loss_kl = kl_loss(mean=self._mean, logVar=self._log_var, mean_target=tf.zeros_like(self._mean), logVar_target=tf.zeros_like(self._log_var))
            self._loss_kl = tf.reduce_mean(self._loss_kl, axis=0)
            self._loss_shape = binary_loss(xPred=self._output_images_pred, xTarget=output_images, gamma=0.60, b_range=False)
            self._loss_shape = tf.reduce_mean(self._loss_shape, axis=0)

            loss_net_reg = tf.reduce_sum(self._encoder_head.losses + self._encoder_backbone.losses
                                         + self._decoder.losses)

            # total loss
            total_loss = (self._loss_kl + self._loss_shape + loss_net_reg)

        trainable_variables = self._encoder_backbone.trainable_variables + self._encoder_head.trainable_variables \
                            + self._decoder.trainable_variables
        grads = tape.gradient(total_loss, trainable_variables)
        self._optimizer.apply_gradients(zip(grads, trainable_variables))

        TP, FP, FN = voxelPrecisionRecall(xTarget=output_images, xPred=self._output_images_pred)
        pr = tf.reduce_mean(TP / (TP + FP + 1e-10), axis=0)
        rc = tf.reduce_mean(TP / (TP + FN + 1e-10), axis=0)

        return self._loss_kl, self._loss_shape, pr, rc

    # def getEval(self, inputs, training=False, missing_prob=0.0):
    #     input_images, output_images = inputs
    #     batch_num = len(input_images)
    #     input_images, output_images = tf.convert_to_tensor(input_images), tf.convert_to_tensor(output_images)
    #     # get encoder output and loss
    #     enc_output = self._encoder_head(self._encoder_backbone(input_images, training=training), training=training)
    #     part_start, part_end = 0, self._enc_backbone_str['z_dim']
    #     self._mean = enc_output[..., part_start:part_end]
    #     part_start, part_end = part_end, part_end + self._enc_backbone_str['z_dim']
    #     self._log_var = tf.clip_by_value(enc_output[..., part_start:], clip_value_min=-10.0, clip_value_max=10.0)
    #
    #     self._z = sampling(mu=self._mean, logVar=self._log_var)
    #
    #     if missing_prob>0:
    #         z_dim = tf.shape(self._z).numpy()[-1]
    #         mask = np.random.choice(2, batch_num * z_dim, p=[missing_prob, 1.- missing_prob])
    #         mask = tf.convert_to_tensor(np.reshape(mask, [batch_num, z_dim]).astype('float32'))
    #         self._z = tf.where(mask == 0, 0.0 * tf.ones_like(self._z), self._z)
    #
    #     output_images_pred = self._decoder(self._z, training=training)
    #     loss_shape = binary_loss(xPred=output_images_pred, xTarget=output_images, gamma=0.60, b_range=False)
    #     loss_shape = tf.reduce_mean(loss_shape, axis=0)
    #     TP, FP, FN = voxelPrecisionRecall(xTarget=output_images, xPred=output_images_pred)
    #     pr = tf.reduce_mean(TP / (TP + FP + 1e-10), axis=0)
    #     rc = tf.reduce_mean(TP / (TP + FN + 1e-10), axis=0)
    #     return output_images_pred, loss_shape, pr, rc

    def getEval(self, inputs, category_vectors, training=False, missing_prob=0.0):
        input_images, output_images, category_list = inputs
        batch_num = len(input_images)
        category_num = len(category_vectors)
        input_images, output_images, category_list, category_vectors = tf.convert_to_tensor(
            input_images), tf.convert_to_tensor(output_images), tf.convert_to_tensor(
            category_list), tf.convert_to_tensor(category_vectors)
        # get priornet output
        self._mean_category_prior = category_vectors
        self._mean_category_prior_tile = tf.reshape(self._mean_category_prior, (1, category_num, self._enc_backbone_str['z_dim']))
        self._mean_category_prior_tile = tf.tile(self._mean_category_prior_tile, (batch_num, 1, 1))

        # get encoder output and loss
        enc_output = self._encoder_head(self._encoder_backbone(input_images, training=training), training=training)
        part_start, part_end = 0, self._enc_backbone_str['z_dim']
        self._mean_category = enc_output[..., part_start:part_end]
        part_start, part_end = part_end, part_end + self._enc_backbone_str['z_dim']
        self._log_var_category = tf.clip_by_value(enc_output[..., part_start:part_end], clip_value_min=-10.0, clip_value_max=10.0)

        self._z_category = sampling(mu=self._mean_category, logVar=self._log_var_category)

        if missing_prob > 0:
            mean_category_prior_mean = tf.reduce_mean(self._mean_category_prior, axis=0)
            z_dim = tf.shape(self._z_category).numpy()[-1]
            mask = np.random.choice(2, batch_num * z_dim, p=[missing_prob, 1. - missing_prob])
            mask = tf.convert_to_tensor(np.reshape(mask, [batch_num, z_dim]).astype('float32'))
            self._z_category = self._z_category * mask
            # mu, var = 0.0, 1.0 ** 2
            # noise = sampling(mu= mu * tf.ones_like(self._z_category), logVar=tf.math.log( var * tf.ones_like(self._z_category)))
            # self._z_category = tf.where(self._z_category == 0, noise, self._z_category)
            self._z_category = tf.where(self._z_category == 0, mean_category_prior_mean * tf.ones_like(self._z_category), self._z_category)
        else:
            mask = tf.ones_like(self._z_category)
        mask_tile = tf.reshape(mask, (batch_num, 1, self._enc_backbone_str['z_dim']))
        mask_tile = tf.tile(mask_tile, (1, category_num, 1))

        # classification
        z_category_tile = tf.reshape(self._z_category, (batch_num, 1, self._enc_backbone_str['z_dim']))
        z_category_tile = tf.tile(z_category_tile, (1, category_num, 1))
        mean_category_prior_tile = self._mean_category_prior_tile
        dist_category = tf.reduce_sum(tf.square(z_category_tile - mean_category_prior_tile), axis=-1)
        cat_equal = tf.equal(tf.argmin(dist_category, axis=-1), tf.argmax(category_list, axis=-1))
        acc_cat = tf.reduce_mean(tf.cast(cat_equal, tf.float32), axis=0)

        output_images_pred = self._decoder(self._z_category, training=training)
        loss_shape = binary_loss(xPred=output_images_pred, xTarget=output_images, gamma=0.60, b_range=False)
        loss_shape = tf.reduce_mean(loss_shape, axis=0)
        TP, FP, FN = voxelPrecisionRecall(xTarget=output_images, xPred=output_images_pred)
        pr = tf.reduce_mean(TP / (TP + FP + 1e-10), axis=0)
        rc = tf.reduce_mean(TP / (TP + FN + 1e-10), axis=0)
        if missing_prob == 0.0:
            return output_images_pred, loss_shape, pr, rc, acc_cat, 0, 0, 0, 0, 0
        else:
            dist_category = tf.reduce_sum(mask_tile * tf.square(z_category_tile - mean_category_prior_tile), axis=-1)
            min_idx = tf.reshape(tf.argmin(dist_category, axis=-1), (-1, 1))
            mean_category_prior_distmin = tf.gather_nd(self._mean_category_prior_tile, min_idx, batch_dims=1)
            z_category_prior = sampling(mu=mean_category_prior_distmin, logVar=tf.zeros_like(mean_category_prior_distmin))
            self._z_category_corrected = tf.where(mask == 0, z_category_prior, self._z_category)
            # classification
            z_category_tile = tf.reshape(self._z_category_corrected, (batch_num, 1, self._enc_backbone_str['z_dim']))
            z_category_tile = tf.tile(z_category_tile, (1, category_num, 1))
            mean_category_prior_tile = self._mean_category_prior_tile
            dist_category = tf.reduce_sum(tf.square(z_category_tile - mean_category_prior_tile), axis=-1)
            cat_equal = tf.equal(tf.argmin(dist_category, axis=-1), tf.argmax(category_list, axis=-1))
            acc_cat_corrected = tf.reduce_mean(tf.cast(cat_equal, tf.float32), axis=0)

            output_images_pred_corrected = self._decoder(self._z_category_corrected, training=training)
            loss_shape_corrected = binary_loss(xPred=output_images_pred_corrected, xTarget=output_images, gamma=0.60, b_range=False)
            loss_shape_corrected = tf.reduce_mean(loss_shape_corrected, axis=0)
            TP_corrected, FP_corrected, FN_corrected = voxelPrecisionRecall(xTarget=output_images, xPred=output_images_pred_corrected)
            pr_corrected = tf.reduce_mean(TP_corrected / (TP_corrected + FP_corrected + 1e-10), axis=0)
            rc_corrected = tf.reduce_mean(TP_corrected / (TP_corrected + FN_corrected + 1e-10), axis=0)
            return output_images_pred, loss_shape, pr, rc, acc_cat, output_images_pred_corrected, loss_shape_corrected, pr_corrected, rc_corrected, acc_cat_corrected


    def getLatent(self, inputs):
        input_images = tf.convert_to_tensor(inputs)
        enc_output = self._encoder_head(self._encoder_backbone(input_images, training=False), training=False)
        part_start, part_end = 0, self._enc_backbone_str['z_dim']
        mean = enc_output[..., part_start:part_end]
        part_start, part_end = part_end, part_end + self._enc_backbone_str['z_dim']
        log_var = tf.clip_by_value(enc_output[..., part_start:], clip_value_min=-10.0, clip_value_max=10.0)
        latents = sampling(mu=mean, logVar=log_var)
        return np.array(latents)

    def saveEncoderBackbone(self, save_path):
        file_name = self._enc_backbone_str['name']
        self._encoder_backbone.save_weights(os.path.join(save_path, file_name))

    def saveEncoderHead(self, save_path):
        file_name = self._enc_head_str['name']
        self._encoder_head.save_weights(os.path.join(save_path, file_name))

    def saveEncoder(self, save_path):
        self.saveEncoderBackbone(save_path=save_path)
        self.saveEncoderHead(save_path=save_path)

    def saveDecoder(self, save_path):
        file_name = self._dec_str['name']
        self._decoder.save_weights(os.path.join(save_path, file_name))

    def saveModel(self, save_path):
        self.saveEncoder(save_path=save_path)
        self.saveDecoder(save_path=save_path)

    def loadEncoderBackbone(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._enc_backbone_str['name']
        self._encoder_backbone.load_weights(os.path.join(load_path, file_name))

    def loadEncoderHead(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._enc_head_str['name']
        self._encoder_head.load_weights(os.path.join(load_path, file_name))

    def loadEncoder(self, load_path):
        self.loadEncoderBackbone(load_path=load_path)
        self.loadEncoderHead(load_path=load_path)

    def loadDecoder(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._dec_str['name']
        self._decoder.load_weights(os.path.join(load_path, file_name))

    def loadModel(self, load_path):
        self.loadEncoder(load_path=load_path)
        self.loadDecoder(load_path=load_path)

class nolboSingleObject_category_only(object):
    def __init__(self, nolbo_structure,
                 backbone_style=None, encoder_backbone=None,
                 # BATCH_SIZE_PER_REPLICA=64,
                 # strategy=None,
                 learning_rate=1e-4):
        self._enc_backbone_str = nolbo_structure['encoder_backbone']
        self._enc_head_str = nolbo_structure['encoder_head']
        self._dec_str = nolbo_structure['decoder']
        self._prior_class_str = nolbo_structure['prior_class']

        self._backbone_style = backbone_style
        self._encoder_backbone = encoder_backbone

        # self._strategy = strategy
        # self._BATCH_SIZE_PER_REPLICA = BATCH_SIZE_PER_REPLICA
        # self._GLOBAL_BATCH_SIZE = self._BATCH_SIZE_PER_REPLICA * self._strategy.num_replicas_in_sync

        # with self._strategy.scope():
        self._buildModel()
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def _buildModel(self):
        print('build Models...')
        if self._encoder_backbone == None:
            self._encoder_backbone = self._backbone_style(name=self._enc_backbone_str['name'])
        # ==============set encoder head
        self._encoder_head = darknet.head2D(name=self._enc_head_str['name'],
                                            input_shape=self._encoder_backbone.output_shape[1:],
                                            output_dim=self._enc_head_str['output_dim'],
                                            filter_num_list=self._enc_head_str['filter_num_list'],
                                            filter_size_list=self._enc_head_str['filter_size_list'],
                                            last_pooling='max', activation=self._enc_head_str['activation'])
        # ==============set decoder3D
        self._decoder = ae3D.decoder3D(structure=self._dec_str)
        # ==============set prior network
        self._priornet_class = priornet.priornet(structure=self._prior_class_str)
        print('done')

    def fit(self, inputs):
        input_images, output_images, category_list = inputs
        input_images, output_images, category_list = tf.convert_to_tensor(input_images), tf.convert_to_tensor(output_images), tf.convert_to_tensor(category_list)
        with tf.GradientTape() as tape:
            # get priornet output
            self._mean_category_prior, self._log_var_category_prior = self._priornet_class(category_list, training=True)

            # get encoder output and loss
            enc_output = self._encoder_head(self._encoder_backbone(input_images, training=True), training=True)
            part_start, part_end = 0, self._enc_backbone_str['z_category_dim']
            self._mean_category = enc_output[..., part_start:part_end]
            part_start, part_end = part_end, part_end + self._enc_backbone_str['z_category_dim']
            self._log_var_category = tf.clip_by_value(enc_output[..., part_start:part_end], clip_value_min=-10.0, clip_value_max=10.0)

            self._z = sampling(mu=self._mean_category, logVar=self._log_var_category)

            self._z_prior = sampling(mu=self._mean_category_prior, logVar=self._log_var_category_prior)

            # get (priornet, decoder) output and loss
            if np.random.rand()>0.5:
                self._output_images_pred = self._decoder(self._z, training=True)
            else:
                missing_pr = 0.3
                noise = tf.convert_to_tensor((np.random.choice(a=[True, False], size=np.array(tf.shape(self._z)), p=[1.-missing_pr, missing_pr])).astype('float32'))
                self._z_input = tf.where(noise==1., self._z, self._z_prior)
                self._output_images_pred = self._decoder(self._z_input, training=True)

            # get loss
            self._loss_kl = kl_loss(mean=self._mean_category, logVar=self._log_var_category,
                                             mean_target=self._mean_category_prior, logVar_target=self._log_var_category_prior)
            self._loss_kl = tf.reduce_mean(self._loss_kl, axis=0)

            self._loss_shape = binary_loss(xPred=self._output_images_pred, xTarget=output_images, gamma=0.60, b_range=False)
            self._loss_shape = tf.reduce_mean(self._loss_shape, axis=0)

            self._loss_reg = regulizer_loss(z_mean=self._mean_category_prior, z_logVar=self._log_var_category_prior,
                                                     dist_in_z_space=3.0*self._enc_backbone_str['z_category_dim'])
            self._loss_reg = tf.reduce_mean(self._loss_reg, axis=0)

            # total loss
            total_loss = (self._loss_kl + self._loss_shape + 0.01 * self._loss_reg)

        trainable_variables = self._encoder_backbone.trainable_variables + self._encoder_head.trainable_variables \
                            + self._decoder.trainable_variables \
                              + self._priornet_class.trainable_variables
        grads = tape.gradient(total_loss, trainable_variables)
        self._optimizer.apply_gradients(zip(grads, trainable_variables))

        TP, FP, FN = voxelPrecisionRecall(xTarget=output_images, xPred=self._output_images_pred)
        pr = tf.reduce_mean(TP / (TP + FP + 1e-10), axis=0)
        rc = tf.reduce_mean(TP / (TP + FN + 1e-10), axis=0)

        return self._loss_kl, self._loss_shape, self._loss_reg, pr, rc

    def getEval(self, inputs, category_indices=np.identity(12), training=False, missing_prob=0.0):
        input_images, output_images, category_list = inputs
        batch_num = len(input_images)
        category_num = len(category_indices)
        input_images, output_images, category_list, category_indices = tf.convert_to_tensor(
            input_images), tf.convert_to_tensor(output_images), tf.convert_to_tensor(
            category_list), tf.convert_to_tensor(category_indices)
        # get priornet output
        self._mean_category_prior, self._log_var_category_prior = self._priornet_class(category_indices, training=training)
        self._mean_category_prior_tile = tf.reshape(self._mean_category_prior, (1, category_num, self._enc_backbone_str['z_category_dim']))
        self._mean_category_prior_tile = tf.tile(self._mean_category_prior_tile, (batch_num, 1, 1))

        # get encoder output and loss
        enc_output = self._encoder_head(self._encoder_backbone(input_images, training=training), training=training)
        part_start, part_end = 0, self._enc_backbone_str['z_category_dim']
        self._mean_category = enc_output[..., part_start:part_end]
        part_start, part_end = part_end, part_end + self._enc_backbone_str['z_category_dim']
        self._log_var_category = tf.clip_by_value(enc_output[..., part_start:part_end], clip_value_min=-10.0, clip_value_max=10.0)

        self._z_category = sampling(mu=self._mean_category, logVar=self._log_var_category)

        if missing_prob > 0:
            mean_category_prior_mean = tf.reduce_mean(self._mean_category_prior, axis=0)
            z_dim = tf.shape(self._z_category).numpy()[-1]
            mask = np.random.choice(2, batch_num * z_dim, p=[missing_prob, 1. - missing_prob])
            mask = tf.convert_to_tensor(np.reshape(mask, [batch_num, z_dim]).astype('float32'))
            self._z_category = self._z_category * mask
            # mu, var = 0.0, 1.0 ** 2
            # noise = sampling(mu= mu * tf.ones_like(self._z_category), logVar=tf.math.log( var * tf.ones_like(self._z_category)))
            # self._z_category = tf.where(self._z_category == 0, noise, self._z_category)
            self._z_category = tf.where(self._z_category == 0, mean_category_prior_mean * tf.ones_like(self._z_category), self._z_category)
        else:
            mask = tf.ones_like(self._z_category)
        mask_tile = tf.reshape(mask, (batch_num, 1, self._enc_backbone_str['z_category_dim']))
        mask_tile = tf.tile(mask_tile, (1, category_num, 1))

        # classification
        z_category_tile = tf.reshape(self._z_category, (batch_num, 1, self._enc_backbone_str['z_category_dim']))
        z_category_tile = tf.tile(z_category_tile, (1, category_num, 1))
        mean_category_prior_tile = self._mean_category_prior_tile
        dist_category = tf.reduce_sum(tf.square(z_category_tile - mean_category_prior_tile), axis=-1)
        cat_equal = tf.equal(tf.argmin(dist_category, axis=-1), tf.argmax(category_list, axis=-1))
        acc_cat = tf.reduce_mean(tf.cast(cat_equal, tf.float32), axis=0)

        output_images_pred = self._decoder(self._z_category, training=training)
        loss_shape = binary_loss(xPred=output_images_pred, xTarget=output_images, gamma=0.60, b_range=False)
        loss_shape = tf.reduce_mean(loss_shape, axis=0)
        TP, FP, FN = voxelPrecisionRecall(xTarget=output_images, xPred=output_images_pred)
        pr = tf.reduce_mean(TP / (TP + FP + 1e-10), axis=0)
        rc = tf.reduce_mean(TP / (TP + FN + 1e-10), axis=0)
        if missing_prob == 0.0:
            return output_images_pred, loss_shape, pr, rc, acc_cat, 0, 0, 0, 0, 0
        else:
            dist_category = tf.reduce_sum(mask_tile * tf.square(z_category_tile - mean_category_prior_tile), axis=-1)
            min_idx = tf.reshape(tf.argmin(dist_category, axis=-1), (-1, 1))
            mean_category_prior_distmin = tf.gather_nd(self._mean_category_prior_tile, min_idx, batch_dims=1)
            z_category_prior = sampling(mu=mean_category_prior_distmin, logVar=tf.zeros_like(mean_category_prior_distmin))
            self._z_category_corrected = tf.where(mask == 0, z_category_prior, self._z_category)
            # classification
            z_category_tile = tf.reshape(self._z_category_corrected, (batch_num, 1, self._enc_backbone_str['z_category_dim']))
            z_category_tile = tf.tile(z_category_tile, (1, category_num, 1))
            mean_category_prior_tile = self._mean_category_prior_tile
            dist_category = tf.reduce_sum(tf.square(z_category_tile - mean_category_prior_tile), axis=-1)
            cat_equal = tf.equal(tf.argmin(dist_category, axis=-1), tf.argmax(category_list, axis=-1))
            acc_cat_corrected = tf.reduce_mean(tf.cast(cat_equal, tf.float32), axis=0)

            output_images_pred_corrected = self._decoder(self._z_category_corrected, training=training)
            loss_shape_corrected = binary_loss(xPred=output_images_pred_corrected, xTarget=output_images, gamma=0.60, b_range=False)
            loss_shape_corrected = tf.reduce_mean(loss_shape_corrected, axis=0)
            TP_corrected, FP_corrected, FN_corrected = voxelPrecisionRecall(xTarget=output_images, xPred=output_images_pred_corrected)
            pr_corrected = tf.reduce_mean(TP_corrected / (TP_corrected + FP_corrected + 1e-10), axis=0)
            rc_corrected = tf.reduce_mean(TP_corrected / (TP_corrected + FN_corrected + 1e-10), axis=0)
            return output_images_pred, loss_shape, pr, rc, acc_cat, output_images_pred_corrected, loss_shape_corrected, pr_corrected, rc_corrected, acc_cat_corrected

    def saveEncoderBackbone(self, save_path):
        file_name = self._enc_backbone_str['name']
        self._encoder_backbone.save_weights(os.path.join(save_path, file_name))

    def saveEncoderHead(self, save_path):
        file_name = self._enc_head_str['name']
        self._encoder_head.save_weights(os.path.join(save_path, file_name))

    def saveEncoder(self, save_path):
        self.saveEncoderBackbone(save_path=save_path)
        self.saveEncoderHead(save_path=save_path)

    def saveDecoder(self, save_path):
        file_name = self._dec_str['name']
        self._decoder.save_weights(os.path.join(save_path, file_name))

    def savePriorCategory(self, save_path):
        file_name = self._prior_class_str['name']
        self._priornet_class.save_weights(os.path.join(save_path, file_name))

    def saveModel(self, save_path):
        self.saveEncoder(save_path=save_path)
        self.saveDecoder(save_path=save_path)
        self.savePriorCategory(save_path=save_path)

    def loadEncoderBackbone(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._enc_backbone_str['name']
        self._encoder_backbone.load_weights(os.path.join(load_path, file_name))

    def loadEncoderHead(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._enc_head_str['name']
        self._encoder_head.load_weights(os.path.join(load_path, file_name))

    def loadEncoder(self, load_path):
        self.loadEncoderBackbone(load_path=load_path)
        self.loadEncoderHead(load_path=load_path)

    def loadDecoder(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._dec_str['name']
        self._decoder.load_weights(os.path.join(load_path, file_name))

    def loadPriorCategory(self, load_path, file_name=None):
        if file_name is None:
            file_name = self._prior_class_str['name']
        self._priornet_class.load_weights(os.path.join(load_path, file_name))

    def loadModel(self, load_path):
        self.loadEncoder(load_path=load_path)
        self.loadDecoder(load_path=load_path)
        self.loadPriorCategory(load_path=load_path)


class nolboSingleObject_modelnet_category_AE(object):
    def __init__(self, nolbo_structure,
                 learning_rate=1e-4,
                 dropout=False):
        self._enc_backbone_str = nolbo_structure
        self._enc_str = nolbo_structure['encoder']
        self._dec_str = nolbo_structure['decoder']
        self._dropout = dropout

        # self._strategy = strategy
        # self._BATCH_SIZE_PER_REPLICA = BATCH_SIZE_PER_REPLICA
        # self._GLOBAL_BATCH_SIZE = self._BATCH_SIZE_PER_REPLICA * self._strategy.num_replicas_in_sync

        # with self._strategy.scope():
        self._buildModel()
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def _buildModel(self):
        print('build Models...')
        self._encoder = ae3D.encoder3D(structure=self._enc_str)
        # ==============set decoder3D
        self._decoder = ae3D.decoder3D(structure=self._dec_str)
        print('done')

    def fit(self, inputs):
        input_images, output_images = inputs
        input_images, output_images = tf.convert_to_tensor(input_images), tf.convert_to_tensor(output_images)
        with tf.GradientTape() as tape:

            # get encoder output and loss
            enc_output = self._encoder(input_images, training=True)
            if self._dropout:
                dropout_rate = np.random.rand()
                enc_output = tf.keras.layers.Dropout(rate=dropout_rate)(enc_output, training=True)
            self._output_images_pred = self._decoder(enc_output, training=True)

            # get loss
            self._loss_shape = binary_loss(xPred=self._output_images_pred, xTarget=output_images, gamma=0.60, b_range=False)
            self._loss_shape = tf.reduce_mean(self._loss_shape, axis=0)

            # total loss
            total_loss = self._loss_shape

        trainable_variables = self._encoder.trainable_variables \
                              + self._decoder.trainable_variables
        grads = tape.gradient(total_loss, trainable_variables)
        self._optimizer.apply_gradients(zip(grads, trainable_variables))

        TP, FP, FN = voxelPrecisionRecall(xTarget=output_images, xPred=self._output_images_pred)
        pr = tf.reduce_mean(TP / (TP + FP + 1e-10), axis=0)
        rc = tf.reduce_mean(TP / (TP + FN + 1e-10), axis=0)

        return self._loss_shape, pr, rc

    def getEval(self, inputs, category_vectors, training=False, missing_prob=0.0):
        input_images, output_images, category_list = inputs
        batch_num = len(input_images)
        category_num = len(category_vectors)
        input_images, output_images, category_list, category_vectors = tf.convert_to_tensor(
            input_images), tf.convert_to_tensor(output_images), tf.convert_to_tensor(
            category_list), tf.convert_to_tensor(category_vectors)
        # get priornet output
        self._mean_category_prior = category_vectors
        self._mean_category_prior_tile = tf.reshape(self._mean_category_prior,
                                                    (1, category_num, self._enc_backbone_str['z_category_dim']))
        self._mean_category_prior_tile = tf.tile(self._mean_category_prior_tile, (batch_num, 1, 1))

        # get encoder output and loss
        self._z_category = self._encoder(input_images, training=training)

        if missing_prob > 0:
            mean_category_prior_mean = tf.reduce_mean(self._mean_category_prior, axis=0)
            z_dim = tf.shape(self._z_category).numpy()[-1]
            mask = np.random.choice(2, batch_num * z_dim, p=[missing_prob, 1. - missing_prob])
            mask = tf.convert_to_tensor(np.reshape(mask, [batch_num, z_dim]).astype('float32'))
            self._z_category = self._z_category * mask
            # mu, var = 0.0, 1.0 ** 2
            # noise = sampling(mu= mu * tf.ones_like(self._z_category), logVar=tf.math.log( var * tf.ones_like(self._z_category)))
            # self._z_category = tf.where(self._z_category == 0, noise, self._z_category)
            self._z_category = tf.where(self._z_category == 0,
                                        mean_category_prior_mean * tf.ones_like(self._z_category), self._z_category)
        else:
            mask = tf.ones_like(self._z_category)
        mask_tile = tf.reshape(mask, (batch_num, 1, self._enc_backbone_str['z_category_dim']))
        mask_tile = tf.tile(mask_tile, (1, category_num, 1))

        # classification
        z_category_tile = tf.reshape(self._z_category, (batch_num, 1, self._enc_backbone_str['z_category_dim']))
        z_category_tile = tf.tile(z_category_tile, (1, category_num, 1))
        mean_category_prior_tile = self._mean_category_prior_tile
        dist_category = tf.reduce_sum(tf.square(z_category_tile - mean_category_prior_tile), axis=-1)
        cat_equal = tf.equal(tf.argmin(dist_category, axis=-1), tf.argmax(category_list, axis=-1))
        acc_cat = tf.reduce_mean(tf.cast(cat_equal, tf.float32), axis=0)

        output_images_pred = self._decoder(self._z_category, training=training)
        loss_shape = binary_loss(xPred=output_images_pred, xTarget=output_images, gamma=0.60, b_range=False)
        loss_shape = tf.reduce_mean(loss_shape, axis=0)
        TP, FP, FN = voxelPrecisionRecall(xTarget=output_images, xPred=output_images_pred)
        pr = tf.reduce_mean(TP / (TP + FP + 1e-10), axis=0)
        rc = tf.reduce_mean(TP / (TP + FN + 1e-10), axis=0)
        if missing_prob == 0.0:
            return output_images_pred, loss_shape, pr, rc, acc_cat, 0, 0, 0, 0, 0
        else:
            dist_category = tf.reduce_sum(mask_tile * tf.square(z_category_tile - mean_category_prior_tile), axis=-1)
            min_idx = tf.reshape(tf.argmin(dist_category, axis=-1), (-1, 1))
            mean_category_prior_distmin = tf.gather_nd(self._mean_category_prior_tile, min_idx, batch_dims=1)
            z_category_prior = sampling(mu=mean_category_prior_distmin,
                                        logVar=tf.zeros_like(mean_category_prior_distmin))
            self._z_category_corrected = tf.where(mask == 0, z_category_prior, self._z_category)
            # classification
            z_category_tile = tf.reshape(self._z_category_corrected,
                                         (batch_num, 1, self._enc_backbone_str['z_category_dim']))
            z_category_tile = tf.tile(z_category_tile, (1, category_num, 1))
            mean_category_prior_tile = self._mean_category_prior_tile
            dist_category = tf.reduce_sum(tf.square(z_category_tile - mean_category_prior_tile), axis=-1)
            cat_equal = tf.equal(tf.argmin(dist_category, axis=-1), tf.argmax(category_list, axis=-1))
            acc_cat_corrected = tf.reduce_mean(tf.cast(cat_equal, tf.float32), axis=0)

            output_images_pred_corrected = self._decoder(self._z_category_corrected, training=training)
            loss_shape_corrected = binary_loss(xPred=output_images_pred_corrected, xTarget=output_images, gamma=0.60,
                                               b_range=False)
            loss_shape_corrected = tf.reduce_mean(loss_shape_corrected, axis=0)
            TP_corrected, FP_corrected, FN_corrected = voxelPrecisionRecall(xTarget=output_images,
                                                                            xPred=output_images_pred_corrected)
            pr_corrected = tf.reduce_mean(TP_corrected / (TP_corrected + FP_corrected + 1e-10), axis=0)
            rc_corrected = tf.reduce_mean(TP_corrected / (TP_corrected + FN_corrected + 1e-10), axis=0)
            return output_images_pred, loss_shape, pr, rc, acc_cat, output_images_pred_corrected, loss_shape_corrected, pr_corrected, rc_corrected, acc_cat_corrected

    # def getEval(self, inputs, training=False, missing_prob=0.0):
    #     input_images, output_images = inputs
    #     batch_num = len(input_images)
    #     input_images, output_images = tf.convert_to_tensor(input_images), tf.convert_to_tensor(output_images)
    #     # get encoder output and loss
    #     self._z = self._encoder(input_images, training=training)
    #
    #     if missing_prob > 0:
    #         z_dim = tf.shape(self._z).numpy()[-1]
    #         mask = np.random.choice(2, batch_num * z_dim, p=[missing_prob, 1. - missing_prob])
    #         mask = tf.convert_to_tensor(np.reshape(mask, [batch_num, z_dim]).astype('float32'))
    #         self._z = tf.where(mask == 0, 0.0 * tf.ones_like(self._z), self._z)
    #
    #     output_images_pred = self._decoder(self._z, training=training)
    #     loss_shape = binary_loss(xPred=output_images_pred, xTarget=output_images, gamma=0.60, b_range=False)
    #     loss_shape = tf.reduce_mean(loss_shape, axis=0)
    #     TP, FP, FN = voxelPrecisionRecall(xTarget=output_images, xPred=output_images_pred)
    #     pr = tf.reduce_mean(TP / (TP + FP + 1e-10), axis=0)
    #     rc = tf.reduce_mean(TP / (TP + FN + 1e-10), axis=0)
    #     return output_images_pred, loss_shape, pr, rc

    def getLatent(self, inputs):
        input_images = tf.convert_to_tensor(inputs)
        latent = self._encoder(input_images, training=False)
        return np.array(latent)

    def saveEncoder(self, save_path):
        file_name = self._enc_str['name']
        self._encoder.save_weights(os.path.join(save_path, file_name))

    def saveDecoder(self, save_path):
        file_name = self._dec_str['name']
        self._decoder.save_weights(os.path.join(save_path, file_name))

    def saveModel(self, save_path):
        self.saveEncoder(save_path=save_path)
        self.saveDecoder(save_path=save_path)

    def loadEncoder(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._enc_str['name']
        self._encoder.load_weights(os.path.join(load_path, file_name))

    def loadDecoder(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._dec_str['name']
        self._decoder.load_weights(os.path.join(load_path, file_name))

    def loadModel(self, load_path):
        self.loadEncoder(load_path=load_path)
        self.loadDecoder(load_path=load_path)


class nolboSingleObject_modelnet_category_VAE(object):
    def __init__(self, nolbo_structure,
                 dropout=False,
                 learning_rate=1e-4):
        self._enc_backbone_str = nolbo_structure
        self._enc_str = nolbo_structure['encoder']
        self._dec_str = nolbo_structure['decoder']
        self._dropout = dropout

        # self._strategy = strategy
        # self._BATCH_SIZE_PER_REPLICA = BATCH_SIZE_PER_REPLICA
        # self._GLOBAL_BATCH_SIZE = self._BATCH_SIZE_PER_REPLICA * self._strategy.num_replicas_in_sync

        # with self._strategy.scope():
        self._buildModel()
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def _buildModel(self):
        print('build Models...')
        self._encoder = ae3D.encoder3D(structure=self._enc_str)
        # ==============set decoder3D
        self._decoder = ae3D.decoder3D(structure=self._dec_str)
        print('done')

    def fit(self, inputs):
        input_images, output_images = inputs
        input_images, output_images = tf.convert_to_tensor(input_images), tf.convert_to_tensor(output_images)
        with tf.GradientTape() as tape:
            # get encoder output and loss
            enc_output = self._encoder(input_images, training=True)
            part_start, part_end = 0, self._enc_backbone_str['z_category_dim']
            self._mean = enc_output[..., part_start:part_end]
            part_start, part_end = part_end, part_end + self._enc_backbone_str['z_category_dim']
            self._log_var = tf.clip_by_value(enc_output[..., part_start:part_end], clip_value_min=-10.0, clip_value_max=10.0)

            self._z = sampling(mu=self._mean, logVar=self._log_var)
            if self._dropout:
                dropout_rate = np.random.rand()
                self._z = tf.keras.layers.Dropout(rate=dropout_rate)(self._z, training=True)
            self._output_images_pred = self._decoder(self._z, training=True)

            # get loss
            self._loss_kl = kl_loss(mean=self._mean, logVar=self._log_var,
                                    mean_target=tf.zeros_like(self._mean), logVar_target=tf.zeros_like(self._log_var))
            self._loss_kl = tf.reduce_mean(self._loss_kl, axis=0)
            self._loss_shape = binary_loss(xPred=self._output_images_pred, xTarget=output_images, gamma=0.60, b_range=False)
            self._loss_shape = tf.reduce_mean(self._loss_shape, axis=0)

            # total loss
            total_loss = self._loss_kl + self._loss_shape

        trainable_variables = self._encoder.trainable_variables \
                              + self._decoder.trainable_variables
        grads = tape.gradient(total_loss, trainable_variables)
        self._optimizer.apply_gradients(zip(grads, trainable_variables))

        TP, FP, FN = voxelPrecisionRecall(xTarget=output_images, xPred=self._output_images_pred)
        pr = tf.reduce_mean(TP / (TP + FP + 1e-10), axis=0)
        rc = tf.reduce_mean(TP / (TP + FN + 1e-10), axis=0)

        return self._loss_kl, self._loss_shape, pr, rc

    def getEval(self, inputs, category_vectors, training=False, missing_prob=0.0):
        input_images, output_images, category_list = inputs
        batch_num = len(input_images)
        category_num = len(category_vectors)
        input_images, output_images, category_list, category_vectors = tf.convert_to_tensor(
            input_images), tf.convert_to_tensor(output_images), tf.convert_to_tensor(
            category_list), tf.convert_to_tensor(category_vectors)
        # get priornet output
        self._mean_category_prior = category_vectors
        self._mean_category_prior_tile = tf.reshape(self._mean_category_prior,
                                                    (1, category_num, self._enc_backbone_str['z_category_dim']))
        self._mean_category_prior_tile = tf.tile(self._mean_category_prior_tile, (batch_num, 1, 1))

        # get encoder output and loss
        enc_output = self._encoder(input_images, training=training)
        part_start, part_end = 0, self._enc_backbone_str['z_category_dim']
        self._mean_category = enc_output[..., part_start:part_end]
        part_start, part_end = part_end, part_end + self._enc_backbone_str['z_category_dim']
        self._log_var_category = tf.clip_by_value(enc_output[..., part_start:part_end], clip_value_min=-10.0,
                                                  clip_value_max=10.0)

        self._z_category = sampling(mu=self._mean_category, logVar=self._log_var_category)

        if missing_prob > 0:
            mean_category_prior_mean = tf.reduce_mean(self._mean_category_prior, axis=0)
            z_dim = tf.shape(self._z_category).numpy()[-1]
            mask = np.random.choice(2, batch_num * z_dim, p=[missing_prob, 1. - missing_prob])
            mask = tf.convert_to_tensor(np.reshape(mask, [batch_num, z_dim]).astype('float32'))
            self._z_category = self._z_category * mask
            # mu, var = 0.0, 1.0 ** 2
            # noise = sampling(mu= mu * tf.ones_like(self._z_category), logVar=tf.math.log( var * tf.ones_like(self._z_category)))
            # self._z_category = tf.where(self._z_category == 0, noise, self._z_category)
            self._z_category = tf.where(self._z_category == 0,
                                        mean_category_prior_mean * tf.ones_like(self._z_category), self._z_category)
        else:
            mask = tf.ones_like(self._z_category)
        mask_tile = tf.reshape(mask, (batch_num, 1, self._enc_backbone_str['z_category_dim']))
        mask_tile = tf.tile(mask_tile, (1, category_num, 1))

        # classification
        z_category_tile = tf.reshape(self._z_category, (batch_num, 1, self._enc_backbone_str['z_category_dim']))
        z_category_tile = tf.tile(z_category_tile, (1, category_num, 1))
        mean_category_prior_tile = self._mean_category_prior_tile
        dist_category = tf.reduce_sum(tf.square(z_category_tile - mean_category_prior_tile), axis=-1)
        cat_equal = tf.equal(tf.argmin(dist_category, axis=-1), tf.argmax(category_list, axis=-1))
        acc_cat = tf.reduce_mean(tf.cast(cat_equal, tf.float32), axis=0)

        output_images_pred = self._decoder(self._z_category, training=training)
        loss_shape = binary_loss(xPred=output_images_pred, xTarget=output_images, gamma=0.60, b_range=False)
        loss_shape = tf.reduce_mean(loss_shape, axis=0)
        TP, FP, FN = voxelPrecisionRecall(xTarget=output_images, xPred=output_images_pred)
        pr = tf.reduce_mean(TP / (TP + FP + 1e-10), axis=0)
        rc = tf.reduce_mean(TP / (TP + FN + 1e-10), axis=0)
        if missing_prob == 0.0:
            return output_images_pred, loss_shape, pr, rc, acc_cat, 0, 0, 0, 0, 0
        else:
            dist_category = tf.reduce_sum(mask_tile * tf.square(z_category_tile - mean_category_prior_tile), axis=-1)
            min_idx = tf.reshape(tf.argmin(dist_category, axis=-1), (-1, 1))
            mean_category_prior_distmin = tf.gather_nd(self._mean_category_prior_tile, min_idx, batch_dims=1)
            z_category_prior = sampling(mu=mean_category_prior_distmin,
                                        logVar=tf.zeros_like(mean_category_prior_distmin))
            self._z_category_corrected = tf.where(mask == 0, z_category_prior, self._z_category)
            # classification
            z_category_tile = tf.reshape(self._z_category_corrected,
                                         (batch_num, 1, self._enc_backbone_str['z_category_dim']))
            z_category_tile = tf.tile(z_category_tile, (1, category_num, 1))
            mean_category_prior_tile = self._mean_category_prior_tile
            dist_category = tf.reduce_sum(tf.square(z_category_tile - mean_category_prior_tile), axis=-1)
            cat_equal = tf.equal(tf.argmin(dist_category, axis=-1), tf.argmax(category_list, axis=-1))
            acc_cat_corrected = tf.reduce_mean(tf.cast(cat_equal, tf.float32), axis=0)

            output_images_pred_corrected = self._decoder(self._z_category_corrected, training=training)
            loss_shape_corrected = binary_loss(xPred=output_images_pred_corrected, xTarget=output_images, gamma=0.60,
                                               b_range=False)
            loss_shape_corrected = tf.reduce_mean(loss_shape_corrected, axis=0)
            TP_corrected, FP_corrected, FN_corrected = voxelPrecisionRecall(xTarget=output_images,
                                                                            xPred=output_images_pred_corrected)
            pr_corrected = tf.reduce_mean(TP_corrected / (TP_corrected + FP_corrected + 1e-10), axis=0)
            rc_corrected = tf.reduce_mean(TP_corrected / (TP_corrected + FN_corrected + 1e-10), axis=0)
            return output_images_pred, loss_shape, pr, rc, acc_cat, output_images_pred_corrected, loss_shape_corrected, pr_corrected, rc_corrected, acc_cat_corrected

    # def getEval(self, inputs, category_indices=np.identity(40), training=False, missing_prob=0.0):
    #     input_images, output_images = inputs
    #     batch_num = len(input_images)
    #     input_images, output_images = tf.convert_to_tensor(input_images), tf.convert_to_tensor(output_images)
    #     # get encoder output and loss
    #     enc_output = self._encoder(input_images, training=training)
    #     part_start, part_end = 0, self._enc_backbone_str['z_category_dim']
    #     self._mean = enc_output[..., part_start:part_end]
    #     part_start, part_end = part_end, part_end + self._enc_backbone_str['z_category_dim']
    #     self._log_var = tf.clip_by_value(enc_output[..., part_start:part_end], clip_value_min=-10.0, clip_value_max=10.0)
    #
    #     self._z = sampling(mu=self._mean, logVar=self._log_var)
    #
    #     if missing_prob > 0:
    #         z_dim = tf.shape(self._z).numpy()[-1]
    #         mask = np.random.choice(2, batch_num * z_dim, p=[missing_prob, 1. - missing_prob])
    #         mask = tf.convert_to_tensor(np.reshape(mask, [batch_num, z_dim]).astype('float32'))
    #         self._z = tf.where(mask == 0, 0.0 * tf.ones_like(self._z), self._z)
    #
    #     output_images_pred = self._decoder(self._z, training=training)
    #     loss_shape = binary_loss(xPred=output_images_pred, xTarget=output_images, gamma=0.60, b_range=False)
    #     loss_shape = tf.reduce_mean(loss_shape, axis=0)
    #     TP, FP, FN = voxelPrecisionRecall(xTarget=output_images, xPred=output_images_pred)
    #     pr = tf.reduce_mean(TP / (TP + FP + 1e-10), axis=0)
    #     rc = tf.reduce_mean(TP / (TP + FN + 1e-10), axis=0)
    #     return output_images_pred, loss_shape, pr, rc

    def getLatent(self, inputs):
        input_images = tf.convert_to_tensor(inputs)
        enc_output = self._encoder(input_images, training=False)
        part_start, part_end = 0, self._enc_backbone_str['z_category_dim']
        self._mean = enc_output[..., part_start:part_end]
        part_start, part_end = part_end, part_end + self._enc_backbone_str['z_category_dim']
        self._log_var = tf.clip_by_value(enc_output[..., part_start:part_end], clip_value_min=-10.0,
                                         clip_value_max=10.0)
        latent = sampling(mu=self._mean, logVar=self._log_var)
        return np.array(latent)

    def saveEncoder(self, save_path):
        file_name = self._enc_str['name']
        self._encoder.save_weights(os.path.join(save_path, file_name))

    def saveDecoder(self, save_path):
        file_name = self._dec_str['name']
        self._decoder.save_weights(os.path.join(save_path, file_name))

    def saveModel(self, save_path):
        self.saveEncoder(save_path=save_path)
        self.saveDecoder(save_path=save_path)

    def loadEncoder(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._enc_str['name']
        self._encoder.load_weights(os.path.join(load_path, file_name))

    def loadDecoder(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._dec_str['name']
        self._decoder.load_weights(os.path.join(load_path, file_name))

    def loadModel(self, load_path):
        self.loadEncoder(load_path=load_path)
        self.loadDecoder(load_path=load_path)

class nolboSingleObject_modelnet_category_only(object):
    def __init__(self, nolbo_structure,
                 learning_rate=1e-4):
        self._enc_backbone_str = nolbo_structure
        self._enc_str = nolbo_structure['encoder']
        self._dec_str = nolbo_structure['decoder']
        self._prior_class_str = nolbo_structure['prior_class']

        # self._strategy = strategy
        # self._BATCH_SIZE_PER_REPLICA = BATCH_SIZE_PER_REPLICA
        # self._GLOBAL_BATCH_SIZE = self._BATCH_SIZE_PER_REPLICA * self._strategy.num_replicas_in_sync

        # with self._strategy.scope():
        self._buildModel()
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def _buildModel(self):
        print('build Models...')
        self._encoder = ae3D.encoder3D(structure=self._enc_str)
        # ==============set decoder3D
        self._decoder = ae3D.decoder3D(structure=self._dec_str)
        # ==============set prior network
        self._priornet_class = priornet.priornet(structure=self._prior_class_str)
        print('done')

    def fit(self, inputs, dropout=False):
        input_images, output_images, category_list = inputs
        input_images, output_images, category_list = tf.convert_to_tensor(input_images), tf.convert_to_tensor(output_images), tf.convert_to_tensor(category_list)
        with tf.GradientTape() as tape:
            # get priornet output
            self._mean_category_prior, self._log_var_category_prior = self._priornet_class(category_list, training=True)

            # get encoder output and loss
            enc_output = self._encoder(input_images, training=True)
            part_start, part_end = 0, self._enc_backbone_str['z_category_dim']
            self._mean_category = enc_output[..., part_start:part_end]
            part_start, part_end = part_end, part_end + self._enc_backbone_str['z_category_dim']
            self._log_var_category = tf.clip_by_value(enc_output[..., part_start:part_end], clip_value_min=-10.0, clip_value_max=10.0)

            self._z = sampling(mu=self._mean_category, logVar=self._log_var_category)

            self._z_prior = sampling(mu=self._mean_category_prior, logVar=self._log_var_category_prior)

            # self._output_images_pred = self._decoder(self._z, training=True)
            # get (priornet, decoder) output and loss
            if np.random.rand() > 0.5:
                self._z_input = self._z
            else:
                missing_pr = 0.3
                noise = tf.convert_to_tensor((np.random.choice(a=[True, False], size=np.array(tf.shape(self._z)), p=[1. - missing_pr, missing_pr])).astype('float32'))
                self._z_input = tf.where(noise == 1., self._z, self._z_prior)

            if dropout:
                dropout_rate = np.random.rand()
                self._z_input = tf.keras.layers.Dropout(rate=dropout_rate)(self._z_input, training=True)
            self._output_images_pred = self._decoder(self._z_input, training=True)

            # get loss
            self._loss_kl = kl_loss(mean=self._mean_category, logVar=self._log_var_category,
                                             mean_target=self._mean_category_prior, logVar_target=self._log_var_category_prior)
            self._loss_kl = tf.reduce_mean(self._loss_kl, axis=0)

            self._loss_shape = binary_loss(xPred=self._output_images_pred, xTarget=output_images, gamma=0.60, b_range=False)
            self._loss_shape = tf.reduce_mean(self._loss_shape, axis=0)

            self._loss_reg = regulizer_loss(z_mean=self._mean_category_prior, z_logVar=self._log_var_category_prior,
                                                     dist_in_z_space=2.0*self._enc_backbone_str['z_category_dim'])
            self._loss_reg = tf.reduce_mean(self._loss_reg, axis=0)

            # total loss
            total_loss = (self._loss_kl + self._loss_shape + 0.01 * self._loss_reg)

        trainable_variables = self._encoder.trainable_variables \
                              + self._decoder.trainable_variables \
                              + self._priornet_class.trainable_variables
        grads = tape.gradient(total_loss, trainable_variables)
        self._optimizer.apply_gradients(zip(grads, trainable_variables))

        TP, FP, FN = voxelPrecisionRecall(xTarget=output_images, xPred=self._output_images_pred)
        pr = tf.reduce_mean(TP / (TP + FP + 1e-10), axis=0)
        rc = tf.reduce_mean(TP / (TP + FN + 1e-10), axis=0)

        return self._loss_kl, self._loss_shape, self._loss_reg, pr, rc

    def getEval(self, inputs, category_indices=np.identity(40), training=False, missing_prob=0.0):
        input_images, output_images, category_list = inputs
        batch_num = len(input_images)
        category_num = len(category_indices)
        input_images, output_images, category_list, category_indices = tf.convert_to_tensor(
            input_images), tf.convert_to_tensor(output_images), tf.convert_to_tensor(
            category_list), tf.convert_to_tensor(category_indices)
        # get priornet output
        self._mean_category_prior, self._log_var_category_prior = self._priornet_class(category_indices, training=training)
        self._mean_category_prior_tile = tf.reshape(self._mean_category_prior, (1, category_num, self._enc_backbone_str['z_category_dim']))
        self._mean_category_prior_tile = tf.tile(self._mean_category_prior_tile, (batch_num, 1, 1))

        # get encoder output and loss
        enc_output = self._encoder(input_images, training=training)
        part_start, part_end = 0, self._enc_backbone_str['z_category_dim']
        self._mean_category = enc_output[..., part_start:part_end]
        part_start, part_end = part_end, part_end + self._enc_backbone_str['z_category_dim']
        self._log_var_category = tf.clip_by_value(enc_output[..., part_start:part_end], clip_value_min=-10.0, clip_value_max=10.0)

        self._z_category = sampling(mu=self._mean_category, logVar=self._log_var_category)

        if missing_prob > 0:
            mean_category_prior_mean = tf.reduce_mean(self._mean_category_prior, axis=0)
            z_dim = tf.shape(self._z_category).numpy()[-1]
            mask = np.random.choice(2, batch_num * z_dim, p=[missing_prob, 1. - missing_prob])
            mask = tf.convert_to_tensor(np.reshape(mask, [batch_num, z_dim]).astype('float32'))
            self._z_category = self._z_category * mask
            # mu, var = 0.0, 1.0 ** 2
            # noise = sampling(mu= mu * tf.ones_like(self._z_category), logVar=tf.math.log( var * tf.ones_like(self._z_category)))
            # self._z_category = tf.where(self._z_category == 0, noise, self._z_category)
            self._z_category = tf.where(self._z_category == 0, mean_category_prior_mean * tf.ones_like(self._z_category), self._z_category)
        else:
            mask = tf.ones_like(self._z_category)
        mask_tile = tf.reshape(mask, (batch_num, 1, self._enc_backbone_str['z_category_dim']))
        mask_tile = tf.tile(mask_tile, (1, category_num, 1))

        # classification
        z_category_tile = tf.reshape(self._z_category, (batch_num, 1, self._enc_backbone_str['z_category_dim']))
        z_category_tile = tf.tile(z_category_tile, (1, category_num, 1))
        mean_category_prior_tile = self._mean_category_prior_tile
        dist_category = tf.reduce_sum(tf.square(z_category_tile - mean_category_prior_tile), axis=-1)
        cat_equal = tf.equal(tf.argmin(dist_category, axis=-1), tf.argmax(category_list, axis=-1))
        acc_cat = tf.reduce_mean(tf.cast(cat_equal, tf.float32), axis=0)

        output_images_pred = self._decoder(self._z_category, training=training)
        loss_shape = binary_loss(xPred=output_images_pred, xTarget=output_images, gamma=0.60, b_range=False)
        loss_shape = tf.reduce_mean(loss_shape, axis=0)
        TP, FP, FN = voxelPrecisionRecall(xTarget=output_images, xPred=output_images_pred)
        pr = tf.reduce_mean(TP / (TP + FP + 1e-10), axis=0)
        rc = tf.reduce_mean(TP / (TP + FN + 1e-10), axis=0)
        if missing_prob == 0.0:
            return output_images_pred, loss_shape, pr, rc, acc_cat, 0, 0, 0, 0, 0
        else:
            dist_category = tf.reduce_sum(mask_tile * tf.square(z_category_tile - mean_category_prior_tile), axis=-1)
            min_idx = tf.reshape(tf.argmin(dist_category, axis=-1), (-1, 1))
            mean_category_prior_distmin = tf.gather_nd(self._mean_category_prior_tile, min_idx, batch_dims=1)
            z_category_prior = sampling(mu=mean_category_prior_distmin, logVar=tf.zeros_like(mean_category_prior_distmin))
            self._z_category_corrected = tf.where(mask == 0, z_category_prior, self._z_category)
            # classification
            z_category_tile = tf.reshape(self._z_category_corrected, (batch_num, 1, self._enc_backbone_str['z_category_dim']))
            z_category_tile = tf.tile(z_category_tile, (1, category_num, 1))
            mean_category_prior_tile = self._mean_category_prior_tile
            dist_category = tf.reduce_sum(tf.square(z_category_tile - mean_category_prior_tile), axis=-1)
            cat_equal = tf.equal(tf.argmin(dist_category, axis=-1), tf.argmax(category_list, axis=-1))
            acc_cat_corrected = tf.reduce_mean(tf.cast(cat_equal, tf.float32), axis=0)

            output_images_pred_corrected = self._decoder(self._z_category_corrected, training=training)
            loss_shape_corrected = binary_loss(xPred=output_images_pred_corrected, xTarget=output_images, gamma=0.60, b_range=False)
            loss_shape_corrected = tf.reduce_mean(loss_shape_corrected, axis=0)
            TP_corrected, FP_corrected, FN_corrected = voxelPrecisionRecall(xTarget=output_images, xPred=output_images_pred_corrected)
            pr_corrected = tf.reduce_mean(TP_corrected / (TP_corrected + FP_corrected + 1e-10), axis=0)
            rc_corrected = tf.reduce_mean(TP_corrected / (TP_corrected + FN_corrected + 1e-10), axis=0)
            return output_images_pred, loss_shape, pr, rc, acc_cat, output_images_pred_corrected, loss_shape_corrected, pr_corrected, rc_corrected, acc_cat_corrected

    def saveEncoder(self, save_path):
        file_name = self._enc_str['name']
        self._encoder.save_weights(os.path.join(save_path, file_name))

    def saveDecoder(self, save_path):
        file_name = self._dec_str['name']
        self._decoder.save_weights(os.path.join(save_path, file_name))

    def savePriorCategory(self, save_path):
        file_name = self._prior_class_str['name']
        self._priornet_class.save_weights(os.path.join(save_path, file_name))

    def saveModel(self, save_path):
        self.saveEncoder(save_path=save_path)
        self.saveDecoder(save_path=save_path)
        self.savePriorCategory(save_path=save_path)

    def loadEncoder(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._enc_str['name']
        self._encoder.load_weights(os.path.join(load_path, file_name))

    def loadDecoder(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._dec_str['name']
        self._decoder.load_weights(os.path.join(load_path, file_name))

    def loadPriorCategory(self, load_path, file_name=None):
        if file_name is None:
            file_name = self._prior_class_str['name']
        self._priornet_class.load_weights(os.path.join(load_path, file_name))

    def loadModel(self, load_path):
        self.loadEncoder(load_path=load_path)
        self.loadDecoder(load_path=load_path)
        self.loadPriorCategory(load_path=load_path)