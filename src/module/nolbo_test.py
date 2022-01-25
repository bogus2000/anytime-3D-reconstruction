import src.net_core.darknet as darknet
import src.net_core.autoencoder3D as ae3D
# import src.net_core.priornet as priornet
from src.module.function import *
import cv2

config = {
    'encoder_backbone':{
        'name' : 'nolbo_backbone',
        'predictor_num':5,
        'bbox2D_dim':4, 'bbox3D_dim':3, 'orientation_dim':3,
        'inst_dim':10, 'z_inst_dim':16,
        'activation' : 'elu',
    },
    'encoder_head':{
        'name' : 'nolbo_head',
        'output_dim' : 5*(1+4+3+(2*3+3)+2*16),
        'filter_num_list':[1024,1024,1024,1024],
        'filter_size_list':[3,3,3,1],
        'activation':'elu',
    },
    'decoder':{
        'name':'docoder',
        'input_dim' : 16,
        'output_shape':[64,64,64,1],
        'filter_num_list':[512,256,128,64,1],
        'filter_size_list':[4,4,4,4,4],
        'strides_list':[1,2,2,2,2],
        'activation':'elu',
        'final_activation':'sigmoid'
    },
    # 'prior' : {
    #     'name' : 'priornet',
    #     'input_dim' : 10,  # class num (one-hot vector)
    #     'unit_num_list' : [64, 32, 16],
    #     'core_activation' : 'elu',
    #     'const_log_var' : 0.0,
    # }
}

class nolbo_test(object):
    def __init__(self,
                 nolbo_structure,
                 backbone_style=None, encoder_backbone=None,
                 ):
        self._enc_backbone_str = nolbo_structure['encoder_backbone']
        self._enc_head_str = nolbo_structure['encoder_head']
        self._dec_str = nolbo_structure['decoder']
        # self._prior_str = nolbo_structure['prior']

        self._backbone_style = backbone_style
        self._encoder_backbone = encoder_backbone

        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # if gpus:
        #     try:
        #         tf.config.experimental.set_virtual_device_configuration(gpus[0], [
        #             tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
        #     except RuntimeError as e:
        #         print(e)

        self._buildModel()


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
                                            last_pooling=None, activation=self._enc_head_str['activation'])
        # ==============set decoder3D
        self._decoder = ae3D.decoder3D(structure=self._dec_str)
        # self._priornet = priornet.priornet(structure=self._prior_str)
        print('done')

    def getPred(self,
                input_image,
                obj_thresh=0.5, IOU_thresh=0.5,
                top_1_pred=True, get_3D_shape=True, is_sampling=True,
                image_reduced=32):
        if input_image.shape[-1] !=3:
            # if gray image
            input_image = np.stack([input_image,input_image,input_image], axis=-1)
        if input_image.ndim != 4:
            input_image = np.stack([input_image], axis=0)
        _, inputImgRow, inputImgCol, _ = input_image.shape
        self._gridSize = [int(inputImgCol/image_reduced), int(inputImgRow/image_reduced)]

        grid_col, grid_row, predictor_num = self._gridSize[0], self._gridSize[1], self._enc_backbone_str['predictor_num']

        self._enc_output = self._encoder_head(self._encoder_backbone(input_image, training=False), training=False)
        self._encOutPartitioning()

        bbox2D_list, bbox3D_list = [], []
        inst_mean_list, inst_log_var_list = [], []
        sin_mean_list, cos_mean_list, rad_log_var_list = [], [], []
        self._objness = np.array(self._objness)
        self._bbox2D = np.array(self._bbox2D)
        self._bbox3D = np.array(self._bbox3D)
        self._ori_sin_mean = np.array(self._ori_sin_mean)
        self._ori_cos_mean = np.array(self._ori_cos_mean)
        self._rad_log_var = np.array(self._rad_log_var)
        for gr in range(grid_row):
            for gc in range(grid_col):
                objness_pr_list = np.argsort(-self._objness[0, gr, gc, :, 0])
                for prn in objness_pr_list:
                    obj_pr = self._objness[0, gr, gc, prn, 0]
                    if obj_pr > obj_thresh:
                        b2h, b2w, b2x, b2y = self._bbox2D[0, gr, gc, prn, :]
                        row_min = (float(gr + b2y) / float(grid_row) - b2h / 2.0)
                        row_max = (float(gr + b2y) / float(grid_row) + b2h / 2.0)
                        col_min = (float(gc + b2x) / float(grid_col) - b2w / 2.0)
                        col_max = (float(gc + b2x) / float(grid_col) + b2w / 2.0)
                        b3w, b3h, b3l = self._bbox3D[0, gr, gc, prn, :] #hwl
                        inst_mean = self._inst_mean[0, gr, gc, prn, :]
                        inst_log_var = self._inst_log_var[0, gr, gc, prn, :]
                        sin_mean = self._ori_sin_mean[0, gr, gc, prn, :]
                        cos_mean = self._ori_cos_mean[0, gr, gc, prn, :]
                        rad_log_var = self._rad_log_var[0, gr, gc, prn, :]

                        bbox2D_list.append([col_min, row_min, col_max, row_max, obj_pr])
                        bbox3D_list.append([b3h, b3w, b3l])
                        inst_mean_list.append(inst_mean)
                        inst_log_var_list.append(inst_log_var)
                        sin_mean_list.append(sin_mean)
                        cos_mean_list.append(cos_mean)
                        rad_log_var_list.append(rad_log_var)
                        if top_1_pred:
                            break
        bbox2D_list = np.array(bbox2D_list)
        bbox3D_list = np.array(bbox3D_list)
        inst_mean_list = np.array(inst_mean_list)
        inst_log_var_list = np.array(inst_log_var_list)
        sin_mean_list = np.array(sin_mean_list)
        cos_mean_list = np.array(cos_mean_list)
        rad_log_var_list = np.array(rad_log_var_list)

        if len(bbox2D_list) > 0:
            selected_box_indices = nonMaximumSuppresion(bbox2D_list, IOU_thresh)
        else:
            selected_box_indices = []
        bbox2D_selected = bbox2D_list[selected_box_indices]
        bbox3D_selected = bbox3D_list[selected_box_indices]
        inst_mean_selected = inst_mean_list[selected_box_indices]
        inst_log_var_selected = inst_log_var_list[selected_box_indices]
        sin_mean_selected = sin_mean_list[selected_box_indices]
        cos_mean_selected = cos_mean_list[selected_box_indices]
        rad_log_var_selected = rad_log_var_list[selected_box_indices]

        image_bbox2D = input_image[0].copy()
        imrow, imcol, _ = image_bbox2D.shape
        for bbox2D in bbox2D_selected:
            color = (0, 255, 0)
            thickness = 2
            p0 = (int(bbox2D[0] * imcol), int(bbox2D[1] * imrow))
            p1 = (int(bbox2D[2] * imcol), int(bbox2D[3] * imrow))
            # print(p0)
            # print(p1)
            # print(image_bbox2D.shape)
            cv2.rectangle(img=image_bbox2D, pt1=p0, pt2=p1, color=color, thickness=thickness)

        if get_3D_shape:
            if len(inst_mean_selected) > 0:
                if is_sampling:
                    outputs_3D_shape = []
                    sampling_num = 32
                    for inst_mean, inst_log_var in zip(inst_mean_selected, inst_log_var_selected):
                        inst_mean_samples = tf.stack([inst_mean] * sampling_num)
                        inst_log_var_samples = tf.stack([inst_log_var] * sampling_num)
                        latents = sampling(inst_mean_samples, inst_log_var_samples)
                        outputs = tf.reshape(tf.reduce_mean(self._decoder(latents, training=False), axis=0), (64,64,64))
                        outputs_3D_shape.append(outputs)
                    outputs_3D_shape = np.array(outputs_3D_shape)
                else:
                    outputs_3D_shape = np.array(tf.reshape(self._decoder(inst_mean_selected), (-1, 64,64,64)))
            else:
                outputs_3D_shape = np.array([])
            return image_bbox2D, bbox2D_selected, bbox3D_selected,\
            sin_mean_selected, cos_mean_selected, rad_log_var_selected,\
            outputs_3D_shape
        else:
            return image_bbox2D, bbox2D_selected, bbox3D_selected,\
            sin_mean_selected, cos_mean_selected, rad_log_var_selected

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
    # def loadPriornet(self, load_path, file_name=None):
    #     if file_name == None:
    #         file_name = self._prior_str['name']
    #     self._priornet.load_weights(os.path.join(load_path, file_name))
    def loadModel(self, load_path):
        self.loadEncoder(load_path=load_path)
        self.loadDecoder(load_path=load_path)
        # self.loadPriornet(load_path=load_path)

    def _encOutPartitioning(self):
        pr_num = self._enc_backbone_str['predictor_num']
        self._objness, self._bbox2D, self._bbox3D = [], [], []
        self._inst_mean, self._inst_log_var = [], []
        self._ori_sin_mean, self._ori_cos_mean, self._rad_log_var = [], [], []
        part_start = 0
        part_end = part_start
        for predIndex in range(pr_num):
            # objectness
            part_end += 1
            self._objness.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['bbox2D_dim']
            self._bbox2D.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['bbox3D_dim']
            self._bbox3D.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['z_inst_dim']
            self._inst_mean.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['z_inst_dim']
            self._inst_log_var.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['orientation_dim']
            self._ori_sin_mean.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['orientation_dim']
            self._ori_cos_mean.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['orientation_dim']
            self._rad_log_var.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
        self._objness = tf.sigmoid(tf.transpose(tf.stack(self._objness), [1, 2, 3, 0, 4]))
        self._bbox2D = tf.transpose(tf.stack(self._bbox2D), [1, 2, 3, 0, 4])
        self._bbox2D = tf.concat([tf.exp(self._bbox2D[..., :2]), tf.sigmoid(self._bbox2D[..., 2:])], axis=-1)
        self._bbox3D = tf.nn.relu(tf.transpose(tf.stack(self._bbox3D), [1,2,3,0,4]))
        self._inst_mean = tf.transpose(tf.stack(self._inst_mean), [1, 2, 3, 0, 4])
        self._inst_log_var = tf.transpose(tf.stack(self._inst_log_var), [1, 2, 3, 0, 4])
        self._ori_sin_mean = tf.tanh(tf.transpose(tf.stack(self._ori_sin_mean), [1, 2, 3, 0, 4]))
        self._ori_cos_mean = tf.tanh(tf.transpose(tf.stack(self._ori_cos_mean), [1, 2, 3, 0, 4]))
        self._rad_log_var = tf.transpose(tf.stack(self._rad_log_var), [1, 2, 3, 0, 4])



