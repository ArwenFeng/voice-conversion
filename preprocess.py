#!/usr/bin/env python
# encoding: utf-8
'''
@author: ArwenFeng
@file: preprocess.py
@time: 2019/4/25 20:12
@desc:
'''
import glob
import os
import numpy as np

from util import audio
# obtain mel and linear scale spectrograms from audio files
def preprocess_timit(pattern = "TIMIT/*/*/*/*.wav"):
    '''
      This writes the mel scale spectrograms to disk

      Args:
        pattern: The pattern to glob
        target_dir: The directory to save mel features
      Returns:
        print the written feature file and frames
        write mel-feature file to target_dir
      '''

    # Load the audio to a numpy array:
    wav_list = glob.glob(pattern)

    for wav_path in wav_list:
        mel_path = wav_path.replace('.wav', '.npy')
        if not os.path.exists(mel_path):
            wav = audio.load_wav(wav_path)

            # Compute a mel-scale spectrogram from the wav:
            mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
            n_frames = mel_spectrogram.shape[1]

            # Write the spectrograms to disk:
            np.save(mel_path, mel_spectrogram.T, allow_pickle=False)
        # Print a tuple describing this training example:
            print (mel_path, n_frames)

def preprocess_LJS(pattern = "LJSpeech/wavs/*.wav",target_path = "LJSpeech/train2_data"):
    '''
      This writes the mel scale spectrograms to disk

      Args:
        pattern: The pattern to glob
        target_dir: The directory to save mel features
      Returns:
        print the written feature file and frames
        write mel-feature file to target_dir
      '''

    # Load the audio to a numpy array:
    wav_list = glob.glob(pattern)
    for wav_path in wav_list:
        target_name = os.path.split(wav_path)[1]
        mel_path = target_name.replace('.wav', '_mel.npy')
        mel_path = os.path.join(target_path,mel_path)
        spec_path = target_name.replace(".wav","_spec.npy")
        spec_path = os.path.join(target_path,spec_path)

        if not os.path.exists(mel_path):
            wav = audio.load_wav(wav_path)

            # Compute a mel-scale spectrogram from the wav:
            spectrogram = audio.spectrogram(wav).astype(np.float32)
            mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
            n_frames = mel_spectrogram.shape[1]

            # Write the spectrograms to disk:
            np.save(spec_path, spectrogram.T, allow_pickle=False)
            np.save(mel_path, mel_spectrogram.T, allow_pickle=False)
        # Print a tuple describing this training example:
            print (mel_path, spec_path, n_frames)

wav_list = glob.glob("LJSpeech/wavs/*.wav")
mel_list = glob.glob("LJSpeech/train2_data/*_mel.npy")
spec_list = glob.glob("LJSpeech/train2_data/*_spec.npy")
print(len(wav_list))
print(len(mel_list))
print(len(spec_list))