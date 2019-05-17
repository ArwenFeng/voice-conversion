#!/usr/bin/env python
# encoding: utf-8
'''
@author: ArwenFeng
@file: model.py
@time: 2019/3/25 15:34
@desc:
'''
import tensorflow as tf
from util.audio import phns
from util.infolog import log
from modules import prenet, cbhg

class M():
    def __init__(self, hparams):
        self._hparams = hparams

    def initialize(self, inputs, input_lengths, targets=None, is_training=True):
        '''Initializes the model for inference.
        Args:
          inputs: float32 Tensor with shape [N, T, M] where N is batch size, T_out is number
            of steps in the output time series, M is num_mels, and values are entries in the mel
            spectrogram. Only needed for training.
          targets: int32 Tensor with shape [N, T] where N is batch size, T_in is number of
            steps in the input time series, and values are character IDs.
          input_lengths: int32 Tensor with shape [N]
        '''
        with tf.variable_scope('inference') as scope:
            batch_size = tf.shape(inputs)[0]
            hp = self._hparams

            prenet_out = prenet(inputs,
                                num_units=[hp.hidden_units, hp.hidden_units // 2],
                                dropout_rate=hp.dropout_rate,
                                is_training=is_training)  # (N, T, E/2)

            # CBHG
            cbhg_out = cbhg(prenet_out, input_lengths, hp.num_banks, hp.hidden_units // 2,
                       hp.num_highway_blocks, hp.norm_type, is_training)

            # Final linear projection
            logits = tf.layers.dense(cbhg_out, len(phns))  # (N, T, V)
            self.ppgs = tf.nn.softmax(logits / hp.t, name='ppgs')  # (N, T, V)
            self.preds = tf.to_int32(tf.argmax(logits, axis=-1))  # (N, T)

            self.inputs = inputs
            self.input_lengths = input_lengths
            self.prenet_outputs = prenet_out
            self.cbhg_outputs = cbhg_out
            self.logits = logits
            self.targets = targets

            log('Initialized the model. Dimensions: ')
            log('  input:               %d' % inputs.shape[-1])
            log('  prenet out:              %d' % prenet_out.shape[-1])
            log('  cbhg out:             %d' % cbhg_out.shape[-1])
            log('  linear out:              %d' % logits.shape[-1])

    def add_loss(self):
        '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
        with tf.variable_scope('loss') as scope:
            hp = self._hparams
            #检测padding，padding的值不算loss
            istarget = tf.sign(tf.abs(tf.reduce_sum(self.inputs, -1)))  # indicator: (N, T)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits / hp.t,
                                                                  labels=self.targets)
            loss *= istarget
            self.loss = tf.reduce_mean(loss)

    def add_acc(self):
        istarget = tf.sign(tf.abs(tf.reduce_sum(self.inputs, -1)))  # indicator: (N, T)
        num_hits = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.targets)) * istarget)
        num_targets = tf.reduce_sum(istarget)
        self.acc = num_hits / num_targets

    def add_optimizer(self, global_step):
        '''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.

        Args:
          global_step: int32 scalar Tensor representing current global step in training
        '''
        with tf.variable_scope('optimizer') as scope:
            hp = self._hparams
            lr = tf.get_variable('learning_rate', initializer=hp.lr, trainable=False)
            self.learning_rate = lr
            optimizer = tf.train.AdamOptimizer(lr)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            self.gradients = gradients
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
            # https://github.com/tensorflow/tensorflow/issues/1122
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
                                                          global_step=global_step)


class M2():
    def __init__(self, hparams):
        self._hparams = hparams

    def initialize(self, inputs, input_lengths, mel_targets=None, linear_targets=None, is_training=True):
        '''Initializes the model for inference.
        Args:
          inputs: float32 Tensor with shape [N, T, V] where N is batch size, T is number
            of steps in the output time series, V is the length of phonemes, and values are posterior
            probability of each phoneme for each time step.
          mel_targets: float32 Tensor with shape [N, T_out, M] where N is batch size, T_out is number
            of steps in the output time series, M is num_mels, and values are entries in the mel
            spectrogram. Only needed for training.
          linear_targets: float32 Tensor with shape [N, T_out, F] where N is batch_size, T_out is number
            of steps in the output time series, F is num_freq, and values are entries in the linear
            spectrogram. Only needed for training.
        '''
        with tf.variable_scope('inference') as scope:
            hp = self._hparams

            prenet_out = prenet(inputs,
                                num_units=[hp.hidden_units, hp.hidden_units // 2],
                                dropout_rate=hp.dropout_rate,
                                is_training=is_training)  # (N, T, E/2)

            # CBHG: linear-scale
            pred_spec = cbhg(prenet_out, input_lengths, hp.num_banks, hp.hidden_units // 2,
                             hp.num_highway_blocks, hp.norm_type, is_training, scope="cbhg_linear")
            pred_spec = tf.layers.dense(pred_spec, hp.num_freq, name='pred_linear')

            # self.pred_mel = pred_mel
            self.pred_spec = pred_spec

            self.inputs = inputs
            self.input_lengths = input_lengths
            self.prenet_outputs = prenet_out
            self.mel_targets = mel_targets
            self.linear_targets = linear_targets
            log('Initialized the model. Dimensions: ')
            log('  input:               %d' % inputs.shape[-1])
            log('  prenet out:              %d' % prenet_out.shape[-1])
            log('  linear out:              %d' % pred_spec.shape[-1])

    def add_loss(self):
        '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
        with tf.variable_scope('loss') as scope:
            hp = self._hparams
            l1 = tf.abs(self.linear_targets - self.pred_spec)
            # Prioritize loss for frequencies under 3000 Hz.
            n_priority_freq = int(3000 / (hp.sample_rate * 0.5) * hp.num_freq)
            self.linear_loss = 0.5 * tf.reduce_mean(l1) + 0.5 * tf.reduce_mean(l1[:, :, 0:n_priority_freq])
            self.loss = self.linear_loss

    def add_optimizer(self, global_step):
        '''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.

        Args:
          global_step: int32 scalar Tensor representing current global step in training
        '''
        with tf.variable_scope('optimizer') as scope:
            hp = self._hparams
            lr = tf.get_variable('learning_rate', initializer=hp.lr, trainable=False)
            self.learning_rate = lr
            optimizer = tf.train.AdamOptimizer(lr)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            self.gradients = gradients
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
            # https://github.com/tensorflow/tensorflow/issues/1122
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
                                                          global_step=global_step)