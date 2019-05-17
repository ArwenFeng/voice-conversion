#!/usr/bin/env python
# encoding: utf-8
'''
@author: ArwenFeng
@file: eval.py
@time: 2019/3/25 20:06
@desc:
'''
from predict import voice_change
origin_file = "test_files/arctic_a0031.wav"
target_file = "test_files/arctic_a0031_dousen.wav"
checkpoint_path = "logdir2_one_cbhg_dousen/model.ckpt-60000"
voice_change(origin_file,target_file,checkpoint_path)