#!/usr/bin/env python
# encoding: utf-8
'''
@author: ArwenFeng
@file: predict.py
@time: 2019/3/26 18:58
@desc:
'''
import glob
import random
import tensorflow as tf
import threading
import time
import traceback
import numpy as np
from hparams import hp
from util.infolog import log
from util.audio import get_train2_mfccs_and_phones, get_mfccs_and_phones


_batches_per_group = 32
_pad = 0

class DataFeeder(threading.Thread):
  '''Feeds batches of data into a queue on a background thread.'''

  def __init__(self, coordinator, data_path):
    super(DataFeeder, self).__init__()
    self._coord = coordinator
    self._hparams = hp
    self._datadir = data_path

    # Load data:
    self.feature_files = glob.glob(self._datadir)
    log('Loaded %d audio files'%len(self.feature_files))

    # Create placeholders for inputs and targets. Don't specify batch size because we want to
    # be able to feed different sized batches at eval time.
    self._placeholders = [
        tf.placeholder(tf.float32, [None, None, hp.num_mels], 'inputs'),
        tf.placeholder(tf.int32, [None, None], 'targets'),
        tf.placeholder(tf.int32, [None], 'input_lengths'),
    ]

    # Create queue for buffering data:
    queue = tf.FIFOQueue(8, [tf.float32, tf.int32, tf.int32], name='input_queue')
    self._enqueue_op = queue.enqueue(self._placeholders)
    self.inputs, self.targets, self.input_lengths= queue.dequeue()
    self.inputs.set_shape(self._placeholders[0].shape)
    self.targets.set_shape(self._placeholders[1].shape)
    self.input_lengths.set_shape(self._placeholders[2].shape)


  def start_in_session(self, session):
    self._session = session
    self.start()


  def run(self):
    try:
      while not self._coord.should_stop():
        self._enqueue_next_group()
    except Exception as e:
      traceback.print_exc()
      self._coord.request_stop(e)


  def _enqueue_next_group(self):
    start = time.time()

    # Read a group of examples:
    n = self._hparams.batch_size

    examples = [self._get_next_example() for i in range(n * _batches_per_group)]

    # Bucket examples based on similar output sequence length for efficiency:
    examples.sort(key=lambda x: x[-1])
    batches = [examples[i:i + n] for i in range(0, len(examples), n)]
    random.shuffle(batches)

    log('Generated %d batches of size %d in %.03f sec' % (len(batches), n, time.time() - start))
    for batch in batches:
      feed_dict = dict(zip(self._placeholders, _prepare_batch(batch)))
      self._session.run(self._enqueue_op, feed_dict=feed_dict)


  def _get_next_example(self):
    '''Loads a single example (mfccs, phns, length) from disk'''
    feature_file = random.choice(self.feature_files)
    return get_mfccs_and_phones(feature_file)


class DataFeeder2(threading.Thread):
  '''Feeds batches of data into a queue on a background thread.'''

  def __init__(self, coordinator, data_path):
    super(DataFeeder2, self).__init__()
    self._coord = coordinator
    self._hparams = hp
    self._datadir = data_path

    # Load data:
    self.feature_files = glob.glob(self._datadir)
    log('Loaded %d audio files'%len(self.feature_files))

    # Create placeholders for inputs and targets. Don't specify batch size because we want to
    # be able to feed different sized batches at eval time.
    self._placeholders = [
        tf.placeholder(tf.float32, [None, None, 61], 'inputs'),
        tf.placeholder(tf.float32, [None, None, hp.num_mels], 'mel_targets'),
        tf.placeholder(tf.float32, [None, None, hp.num_freq], 'linear_targets'),
        tf.placeholder(tf.int32, [None], 'input_lengths'),
    ]

    # Create queue for buffering data:
    queue = tf.FIFOQueue(8, [tf.float32, tf.float32, tf.float32, tf.int32], name='input_queue')
    self._enqueue_op = queue.enqueue(self._placeholders)
    self.inputs, self.mel_targets, self.linear_targets, self.input_lengths= queue.dequeue()
    self.inputs.set_shape(self._placeholders[0].shape)
    self.mel_targets.set_shape(self._placeholders[1].shape)
    self.linear_targets.set_shape(self._placeholders[2].shape)
    self.input_lengths.set_shape(self._placeholders[3].shape)


  def start_in_session(self, session):
    self._session = session
    self.start()


  def run(self):
    try:
      while not self._coord.should_stop():
        self._enqueue_next_group()
    except Exception as e:
      traceback.print_exc()
      self._coord.request_stop(e)


  def _enqueue_next_group(self):
    start = time.time()

    # Read a group of examples:
    n = self._hparams.batch_size

    examples = [self._get_next_example() for i in range(n * _batches_per_group)]

    # Bucket examples based on similar output sequence length for efficiency:
    examples.sort(key=lambda x: x[-1])
    batches = [examples[i:i + n] for i in range(0, len(examples), n)]
    random.shuffle(batches)

    log('Generated %d batches of size %d in %.03f sec' % (len(batches), n, time.time() - start))
    for batch in batches:
      feed_dict = dict(zip(self._placeholders, _prepare_batch2(batch)))
      self._session.run(self._enqueue_op, feed_dict=feed_dict)


  def _get_next_example(self):
    '''Loads a single example (mfccs, phns, length) from disk'''
    feature_file = random.choice(self.feature_files)
    mel,spec,pggs = get_train2_mfccs_and_phones(feature_file)
    length = mel.shape[0]
    return pggs, mel, spec, length

def _prepare_batch(batch):
  random.shuffle(batch)
  inputs = _prepare_inputs([x[0] for x in batch])
  targets = _prepare_targets([x[1] for x in batch])
  input_lengths = np.asarray([len(x[1]) for x in batch], dtype=np.int32)
  return (inputs,targets, input_lengths)

def _prepare_batch2(batch):
  random.shuffle(batch)
  inputs = _prepare_inputs([x[0] for x in batch])
  mel_targets = _prepare_inputs([x[1] for x in batch])
  linear_targets = _prepare_inputs([x[2] for x in batch])
  input_lengths = np.asarray([x[3] for x in batch], dtype=np.int32)
  return (inputs, mel_targets, linear_targets, input_lengths)

def _prepare_targets(targets):
  max_len = max((len(x) for x in targets))
  return np.stack([_pad_1(x, max_len) for x in targets])


def _prepare_inputs(inputs):
  max_len = max((len(t) for t in inputs))
  return np.stack([_pad_2(t, max_len) for t in inputs])


def _pad_1(x, length):
  return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)


def _pad_2(t, length):
  return np.pad(t, [(0, length - t.shape[0]), (0,0)], mode='constant', constant_values=_pad)


