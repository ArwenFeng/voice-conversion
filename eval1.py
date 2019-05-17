#!/usr/bin/env python
# encoding: utf-8
'''
@author: ArwenFeng
@file: eval.py
@time: 2019/3/25 20:06
@desc:
'''
import argparse
import numpy as np
import time
import os
import tensorflow as tf
import traceback
from datafeeder import DataFeeder
from hparams import hp, hparams_debug_string
from model import M
from util import infolog
log = infolog.log


def eval(log_dir, args):
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = tf.train.latest_checkpoint(hp.logdir)

    log('Loading checkpoint: %s' % checkpoint_path)
    input_path = hp.test_data_path
    log('Loading evaluate data from: %s' % input_path)
    log(hparams_debug_string())

    # Set up DataFeeder:
    coord = tf.train.Coordinator()
    with tf.variable_scope('datafeeder') as scope:
        feeder = DataFeeder(coord, input_path)



    # Set up model:
    with tf.variable_scope('model') as scope:
        model = M(hp)
        model.initialize(feeder.inputs,feeder.input_lengths, feeder.targets, is_training=False)
        model.add_acc()

    # Eval!
    step = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        try:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)
            writer = tf.summary.FileWriter(log_dir)

            feeder.start_in_session(sess)
            i = 0
            sum_acc = 0
            while not coord.should_stop():
                i = i+ 1
                output_seq, target_seq = sess.run([model.preds,model.targets])
                log('Origin: %s' % target_seq[0][:50])
                log('Output: %s' % output_seq[0][:50])
                acc = sess.run(model.acc)
                sum_acc = sum_acc + acc
                avg_acc = sum_acc/i
                log("acc: %f" % acc)
                log("avg_acc: %f" % avg_acc)
            writer.close()

        except Exception as e:
            log('Exiting due to exception: %s' % e, slack=True)
            traceback.print_exc()
            coord.request_stop(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='')
    parser.add_argument('--checkpoint', default='logdir/model.ckpt-7000', help='Path to model checkpoint')
    parser.add_argument('--name', default='test', help='Name of the run. Used for logging. Defaults to model name.')
    parser.add_argument('--hp', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--checkpoint_interval', type=int, default=1000,
    help='Steps between writing checkpoints.')
    parser.add_argument('--slack_url', help='Slack webhook URL to get periodic reports.')
    parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
    run_name = args.name
    os.makedirs(hp.logdir, exist_ok=True)
    infolog.init(os.path.join(hp.logdir, 'eval-7000.log'), run_name, args.slack_url)
    hp.parse(args.hp)
    eval(hp.logdir, args)


if __name__ == '__main__':
  main()
