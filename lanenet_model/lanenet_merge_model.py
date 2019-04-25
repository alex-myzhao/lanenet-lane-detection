# -*- coding: utf-8 -*-
# @Time    : 18-5-11 下午5:28
# @Author  : Luo Yao
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_merge_model.py
# @IDE: PyCharm Community Edition
"""
实现LaneNet模型
"""
import tensorflow as tf

from encoder_decoder_model import vgg_encoder
from encoder_decoder_model import fcn_decoder
from encoder_decoder_model import dense_encoder
from encoder_decoder_model import cnn_basenet
from lanenet_model import lanenet_discriminative_loss


class LaneNet(cnn_basenet.CNNBaseModel):
    def __init__(self, phase, net_flag='vgg'):
        super(LaneNet, self).__init__()
        self._net_flag = net_flag
        self._phase = phase
        self._encoder = vgg_encoder.VGG16Encoder(phase=phase)
        self._decoder = fcn_decoder.FCNDecoder(phase=phase)

    def __str__(self):
        info = 'Semantic Segmentation use {:s} as basenet to encode'.format(self._net_flag)
        return info

    def _build_model(self, input_tensor, name):
        with tf.variable_scope(name):
            # first encode
            encode_ret = self._encoder.encode(input_tensor=input_tensor,
                                              name='encode')
            decode_ret = self._decoder.decode(input_tensor_dict=encode_ret,
                                              name='decode',
                                              decode_layer_list=['pool5', 'pool4', 'pool3'])
            return decode_ret


    def compute_loss(self, input_tensor, binary_label, instance_label, name):
        with tf.variable_scope(name):
            # build model
            inference_ret = self._build_model(input_tensor=input_tensor, name='inference')
            decode_logits = inference_ret['logits']
            binary_label_plain = tf.reshape(
                binary_label,
                shape=[binary_label.get_shape().as_list()[0] *
                       binary_label.get_shape().as_list()[1] *
                       binary_label.get_shape().as_list()[2]])
            _, _, counts = tf.unique_with_counts(binary_label_plain)
            counts = tf.cast(counts, tf.float32)
            inverse_weights = tf.divide(1.0, tf.log(tf.add(tf.divide(tf.constant(1.0), counts), tf.constant(1.02))))
            inverse_weights = tf.gather(inverse_weights, binary_label)
            binary_segmenatation_loss = tf.losses.sparse_softmax_cross_entropy(
                labels=binary_label, logits=decode_logits, weights=inverse_weights)
            binary_segmenatation_loss = tf.reduce_mean(binary_segmenatation_loss)

            # regularization
            l2_reg_loss = tf.constant(0.0, tf.float32)
            for vv in tf.trainable_variables():
                if 'bn' in vv.name:
                    continue
                else:
                    l2_reg_loss = tf.add(l2_reg_loss, tf.nn.l2_loss(vv))
            l2_reg_loss *= 0.001
            total_loss = binary_segmenatation_loss + l2_reg_loss
            ret = {
                'total_loss': total_loss,
                'binary_seg_logits': decode_logits,
                'binary_seg_loss': binary_segmenatation_loss
            }
            return ret


    def inference(self, input_tensor, name):
        with tf.variable_scope(name):
            inference_ret = self._build_model(input_tensor=input_tensor, name='inference')
            decode_logits = inference_ret['logits']
            binary_seg_ret = tf.nn.softmax(logits=decode_logits)
            binary_seg_ret = tf.argmax(binary_seg_ret, axis=-1)
            return binary_seg_ret


if __name__ == '__main__':
    model = LaneNet(tf.constant('train', dtype=tf.string))
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input')
    binary_label = tf.placeholder(dtype=tf.int64, shape=[1, 256, 512, 1], name='label')
    instance_label = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 1], name='label')
    ret = model.compute_loss(input_tensor=input_tensor, binary_label=binary_label,
                             instance_label=instance_label, name='loss')
    for vv in tf.trainable_variables():
        if 'bn' in vv.name:
            continue
        print(vv.name)
