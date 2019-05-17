#!/usr/bin/env python
# encoding: utf-8
'''
@author: ArwenFeng
@file: hparams.py
@time: 2019/3/18 14:59
@desc:
'''
import tensorflow as tf


# Default hyperparameters:
hp = tf.contrib.training.HParams(
    # Path:
    logdir = 'logdir',
    logdir2 = 'logdir2_one_cbhg_dousen',
    train_data_path = 'TIMIT/TRAIN/*/*/*.npy',
    test_data_path = 'TIMIT/TEST/*/*/*.npy',
    train2_data_path = "dousen/*.wav",
    train2_feature_path = 'dousen/train2_data',
    
    # Audio:
    num_mels=80,
    num_freq=1025,
    sample_rate=16000,
    frame_length_ms=25, # 10-30
    frame_shift_ms=5, # frame_shift_ms / frame_length = 0 ~ 1/2
    hop_length = 80,
    preemphasis=0.97,
    min_level_db=-100,
    ref_level_db=20,
    num_class = 2,
    # Model:

    hidden_units = 256,  # alias: E
    num_banks = 8,
    num_highway_blocks = 4,
    norm_type = 'ins',  # a normalizer function. value: bn, ln, ins, or None
    t = 1.0,  # temperature
    dropout_rate = 0.2,

    # train
    batch_size = 32,
    lr = 0.0003,
    num_epochs = 1000,
    steps_per_epoch = 100,
    save_per_epoch = 2,
    num_gpu = 1,

    # Eval:
    max_iters=200,
    griffin_lim_iters=60,
    power=1.5,              # Power to raise magnitudes to prior to Griffin-Lim
)


def hparams_debug_string():
  values = hp.values()
  p = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
  return 'Hyperparameters:\n' + '\n'.join(p)
