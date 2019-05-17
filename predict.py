#!/usr/bin/env python
# encoding: utf-8
'''
@author: ArwenFeng
@file: predict.py
@time: 2019/3/26 18:58
@desc:
'''
import io
import os
import glob
import numpy as np
import tensorflow as tf
from hparams import hp
from model import M,M2
from util import infolog,audio
log = infolog.log
from util.audio import get_mfccs_and_spec, get_train2_mfccs

class Convert():
    def load(self, checkpoint_path="logdir2_one_cbhg_dousen/model.ckpt-60000"):
        print('Loading checkpoint: %s' % checkpoint_path)
        inputs = tf.placeholder(tf.float32, [None, None, 61], 'inputs')
        input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')

        with tf.variable_scope('model') as scope:
            self.model = M2(hp)
            self.model.initialize(inputs, input_lengths, is_training=False)
            self.wav_output = self.model.pred_spec[0]

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        tf.reset_default_graph()
        saver.restore(self.session, checkpoint_path)

    def predict(self, pggs,path):
        feed_dict = {
            self.model.inputs: [np.asarray(pggs, dtype=np.float32)],
            self.model.input_lengths: [pggs.shape[0]],
        }

        spec = self.session.run(self.wav_output, feed_dict=feed_dict)
        wav = spec.T
        waveform = audio.inv_spectrogram(wav)
        audio.save_wav(waveform, path)

class Recognizer():
    def load(self,checkpoint_path= "logdir/model.ckpt-6000"):
        # checkpoint_path = tf.train.latest_checkpoint(hp.logdir)
        print('Loading checkpoint: %s' % checkpoint_path)
        inputs = tf.placeholder(tf.float32, [1, None, hp.num_mels], 'inputs')
        input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')

        with tf.variable_scope('model') as scope:
            self.model = M(hp)
            self.model.initialize(inputs, input_lengths, is_training=False)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        tf.reset_default_graph()
        saver.restore(self.session, checkpoint_path)

    def predict(self, wav):
        feed_dict = {
            self.model.inputs: [np.asarray(wav, dtype=np.float32)],
            self.model.input_lengths: [wav.shape[0]],
        }

        ppgs,seq = self.session.run([self.model.ppgs,self.model.preds], feed_dict=feed_dict)
        ppgs = ppgs.reshape(ppgs.shape[1],ppgs.shape[2])

        return ppgs
        # print(seq)
        # phn2idx, idx2phn = load_vocab()
        # phns = [idx2phn[i] for i in seq[0]]
        # print(phns)

def form_pggs():
    wp = Recognizer()
    wp.load()
    files = glob.glob(hp.train2_data_path)

    for file in files:
        # find target_path
        _, name = os.path.split(file)
        name = name.replace(".wav", "_pggs.npy")
        target_path = os.path.join(hp.train2_feature_path, name)
        if not os.path.exists(target_path):
            # predict and save pggs
            wav,_ = get_train2_mfccs(file)
            logits = wp.predict(wav)
            np.save(target_path, logits, allow_pickle=False)
            print(logits.shape)
            print(target_path)
    pattern = os.path.join(hp.train2_feature_path,"*_pggs.npy")
    print(pattern)
    npy = glob.glob(pattern)
    print(len(files))
    print(len(npy))

def voice_change(file, path = "result.wav",checkpoint_path="logdir2_one_cbhg_dousen/model.ckpt-51000"):
    wp = Recognizer()
    wp.load()
    wav, spec = get_mfccs_and_spec(file)
    logits = wp.predict(wav)
    conv = Convert()
    conv.load(checkpoint_path=checkpoint_path)
    print('Synthesizing: %s' % path)
    conv.predict(logits,path)

