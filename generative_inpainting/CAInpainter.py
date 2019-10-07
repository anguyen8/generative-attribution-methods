import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import ipdb
import numpy as np
import tensorflow as tf
import torch
from inpaint_model import InpaintCAModel
from torch.autograd import Variable


class CAInpainter(object):
    def __init__(self, batch_size, checkpoint_dir):
        self.model = InpaintCAModel()
        self.images_ph = tf.placeholder(tf.float32,
                                        shape=[batch_size, 256, 512, 3])

        # with tf.device('/gpu:0'):
        # with tf.device('/cpu:0'):
        # for i, d in enumerate(['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3']):
        #     with tf.device(d):
        output = self.model.build_server_graph(self.images_ph)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        self.output = output

        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(checkpoint_dir, from_name)
            self.assign_ops.append(tf.assign(var, var_value))
        # print('Model loaded.')

        self.pth_mean = np.ones((1, 3, 1, 1), dtype='float32')
        self.pth_mean[0, :, 0, 0] = np.array([0.485, 0.456, 0.406])
        self.pth_std = np.ones((1, 3, 1, 1), dtype='float32')
        self.pth_std[0, :, 0, 0] = np.array([0.229, 0.224, 0.225])
        self.upsample = torch.nn.Upsample(size=(256, 256), mode='bilinear')
        self.downsample = torch.nn.Upsample(size=(224, 224), mode='bilinear')

        # Create a session
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        # sess_config.allow_soft_placement = True
        # sess_config.log_device_placement = True
        self.sess = tf.Session(config=sess_config)

        self.sess.run(self.assign_ops)

    def impute_missing_imgs(self, pytorch_image, pytorch_mask):
        '''
        :param pytorch_image: 1 x 3 x 224 x 224
        :param pytorch_mask: 1 x 3 x 224 x 224. Mask
        :return:
        '''
        pth_img = self.generate_background(pytorch_image, pytorch_mask)

        return pytorch_image * pytorch_mask + pth_img * (1. - pytorch_mask)

    def generate_background(self, pytorch_image, pytorch_mask, batch_process=False):
        '''
        Use to generate whole blurry images with pytorch normalization.
        '''

        mask = pytorch_mask.expand(pytorch_mask.shape[0], 3, 224, 224)
        mask = self.upsample(Variable(mask)).data  # .round()
        mask = mask.cpu().numpy()
        thresh = max(0.5, 0.5 * (np.max(mask) + np.min(mask)))
        mask = (mask < thresh).astype(float)

        # Make it into tensorflow input ordering, then resizing then normalization
        # Do 3 things:
        # - Move from NCHW to NHWC, and from RGB to BGR input
        # - Normalize to 0 - 255 with integer round up
        # - Resize the image size to be 256 x 256
        mask = np.moveaxis(mask, 1, -1)*255
        # mask = (1. - mask) * 255

        image = self.upsample(Variable(pytorch_image)).data.cpu().numpy()
        image = np.round((image * self.pth_std + self.pth_mean) * 255)
        image = np.moveaxis(image, 1, -1)
        image = image[:, :, :, ::-1]

        # t1 = time.time()
        if batch_process:
            image = np.stack((image[0, :], )*mask.shape[0], axis=0)

        input_image = np.concatenate([image, mask], axis=2)
        # print(time.time() - t1)

        # DEBUG
        # import cv2
        # cv2.imwrite('./test_input.jpg', input_image[0])

        # t1 = time.time()
        tf_images = self.sess.run(self.output, {self.images_ph: input_image})
        # print(time.time() - t1)
        # print('#'*25)

        # it's RGB back. So just change back to pytorch normalization
        pth_img = np.moveaxis(tf_images, 3, 1)
        pth_img = ((pth_img / 255.) - self.pth_mean) / self.pth_std

        pth_img = pytorch_image.new(pth_img)
        pth_img = self.downsample(Variable(pth_img)).data

        return pth_img, mask

    def time_impute_missing_imgs(self, pytorch_image, pytorch_mask):
        start_time = time.time()
        result = self.impute_missing_imgs(pytorch_image, pytorch_mask)
        print('Total time:', time.time() - start_time)
        return result

    def reset(self):
        pass

    def eval(self):
        pass

    def cuda(self):
        self.upsample.cuda()
        self.downsample.cuda()

    def __del__(self):
        self.sess.close()
