import os
import glob
import librosa
import librosa.filters
import numpy as np
import tensorflow as tf
import scipy
import random
from hparams import hp


def load_wav(path):
  return librosa.core.load(path, sr=hp.sample_rate)[0]

def save_wav(wav, path):
  wav *= 32767 / max(0.01, np.max(np.abs(wav)))
  scipy.io.wavfile.write(path, hp.sample_rate, wav.astype(np.int16))


def _normalize(S):
  return np.clip((S - hp.min_level_db) / -hp.min_level_db, 0, 1)

def _denormalize(S):
  return (np.clip(S, 0, 1) * -hp.min_level_db) + hp.min_level_db

def preemphasis(x):
  return scipy.signal.lfilter([1, -hp.preemphasis], [1], x)


def inv_preemphasis(x):
  return scipy.signal.lfilter([1], [1, -hp.preemphasis], x)


def spectrogram(y):
  D = _stft(preemphasis(y))
  S = _amp_to_db(np.abs(D)) - hp.ref_level_db
  return _normalize(S)


def inv_spectrogram(spectrogram):
  '''Converts spectrogram to waveform using librosa'''
  S = _db_to_amp(_denormalize(spectrogram) + hp.ref_level_db)  # Convert back to linear
  return inv_preemphasis(_griffin_lim(S ** hp.power))          # Reconstruct phase


def inv_spectrogram_tensorflow(spectrogram):
  '''Builds computational graph to convert spectrogram to waveform using TensorFlow.

  Unlike inv_spectrogram, this does NOT invert the preemphasis. The caller should call
  inv_preemphasis on the output after running the graph.
  '''
  S = _db_to_amp_tensorflow(_denormalize_tensorflow(spectrogram) + hp.ref_level_db)
  return _griffin_lim_tensorflow(tf.pow(S, hp.power))


def melspectrogram(y):
  D = _stft(preemphasis(y))
  S = _amp_to_db(_linear_to_mel(np.abs(D))) - hp.ref_level_db
  return _normalize(S)


def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
  window_length = int(hp.sample_rate * min_silence_sec)
  hop_length = int(window_length / 4)
  threshold = _db_to_amp(threshold_db)
  for x in range(hop_length, len(wav) - window_length, hop_length):
    if np.max(wav[x:x+window_length]) < threshold:
      return x + hop_length
  return len(wav)


def _griffin_lim(S):
  '''librosa implementation of Griffin-Lim
  Based on https://github.com/librosa/librosa/issues/434
  '''
  angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
  S_complex = np.abs(S).astype(np.complex)
  y = _istft(S_complex * angles)
  for i in range(hp.griffin_lim_iters):
    angles = np.exp(1j * np.angle(_stft(y)))
    y = _istft(S_complex * angles)
  return y


def _griffin_lim_tensorflow(S):
  '''TensorFlow implementation of Griffin-Lim
  Based on https://github.com/Kyubyong/tensorflow-exercises/blob/master/Audio_Processing.ipynb
  '''
  with tf.variable_scope('griffinlim'):
    # TensorFlow's stft and istft operate on a batch of spectrograms; create batch of size 1
    S = tf.expand_dims(S, 0)
    S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
    y = _istft_tensorflow(S_complex)
    for i in range(hp.griffin_lim_iters):
      est = _stft_tensorflow(y)
      angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
      y = _istft_tensorflow(S_complex * angles)
    return tf.squeeze(y, 0)


def _stft(y):
  n_fft, hop_length, win_length = _stft_parameters()
  return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y):
  _, hop_length, win_length = _stft_parameters()
  return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def _stft_tensorflow(signals):
  n_fft, hop_length, win_length = _stft_parameters()
  return tf.contrib.signal.stft(signals, win_length, hop_length, n_fft, pad_end=False)


def _istft_tensorflow(stfts):
  n_fft, hop_length, win_length = _stft_parameters()
  return tf.contrib.signal.inverse_stft(stfts, win_length, hop_length, n_fft)


def _stft_parameters():
  n_fft = (hp.num_freq - 1) * 2
  hop_length = int(hp.frame_shift_ms / 1000 * hp.sample_rate)
  win_length = int(hp.frame_length_ms / 1000 * hp.sample_rate)
  return n_fft, hop_length, win_length


# Conversions:

_mel_basis = None

def _linear_to_mel(spectrogram):
  global _mel_basis
  if _mel_basis is None:
    _mel_basis = _build_mel_basis()
  return np.dot(_mel_basis, spectrogram)

def _build_mel_basis():
  n_fft = (hp.num_freq - 1) * 2
  return librosa.filters.mel(hp.sample_rate, n_fft, n_mels=hp.num_mels)

def _amp_to_db(x):
  return 20 * np.log10(np.maximum(1e-5, x))

def _db_to_amp(x):
  return np.power(10.0, x * 0.05)

def _db_to_amp_tensorflow(x):
  return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)

def normalize_0_1(values, max, min):
    normalized = np.clip((values - min) / (max - min), 0, 1)
    return normalized

def denormalize_0_1(normalized, max, min):
    values =  np.clip(normalized, 0, 1) * (max - min) + min
    return values

def _denormalize_tensorflow(S):
  return (tf.clip_by_value(S, 0, 1) * -hp.min_db) + hp.min_db

def get_mfccs_and_spec(wav_file):
  wav = load_wav(wav_file)
  spe = spectrogram(wav).astype(np.float32)
  mel_spectrogram = melspectrogram(wav).astype(np.float32)
  mel = mel_spectrogram.T
  spec = spe.T
  return mel, spec

def get_mfccs(wav_file):
  mel = np.load(wav_file)
  return mel

def get_train2_mfccs(wav_file):
    name = os.path.split(wav_file)[1]
    mel_path = name.replace('.wav', '_mel.npy')
    mel_path = os.path.join(hp.train2_feature_path, mel_path)
    spec_path = name.replace(".wav", "_spec.npy")
    spec_path = os.path.join(hp.train2_feature_path, spec_path)
    mel = np.load(mel_path)
    spec = np.load(spec_path)
    return mel, spec

def get_train2_mfccs_and_phones(wav_file):
    name = os.path.split(wav_file)[1]
    mel_path = name.replace('.wav', '_mel.npy')
    mel_path = os.path.join(hp.train2_feature_path, mel_path)
    spec_path = name.replace(".wav", "_spec.npy")
    spec_path = os.path.join(hp.train2_feature_path, spec_path)
    phone_path = name.replace(".wav","_pggs.npy")
    phone_path = os.path.join(hp.train2_feature_path, phone_path)
    mel = np.load(mel_path)
    spec = np.load(spec_path)
    phones = np.load(phone_path)
    return mel, spec, phones

def get_mfccs_and_phones(wav_file):

    '''This is applied in `train1` or `test1` phase.
    '''
    #n_fft, hop_length, win_length = _stft_parameters()
    # Load
    mfccs = get_mfccs(wav_file)

    # timesteps
    num_timesteps = mfccs.shape[0]

    # phones (targets)
    (rpath, temp) = os.path.split(wav_file)
    (name,_) = os.path.splitext(temp)
    phn_file = name + ".PHN"
    phn_file = os.path.join(rpath,phn_file)

    phn2idx, idx2phn = load_vocab()
    phns = np.zeros(shape=(num_timesteps,))
    bnd_list = []
    for line in open(phn_file, 'r').read().splitlines():
        start_point, end_point , phn = line.split()
        bnd = int(start_point) // hp.hop_length
        phns[bnd:] = phn2idx[phn]
        bnd_list.append(bnd)
    return mfccs, phns, len(phns)


phns = ['h#', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl',
        'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi',
        'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh',
        'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl',
        'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']


def load_vocab():
    phn2idx = {phn: idx for idx, phn in enumerate(phns)}
    idx2phn = {idx: phn for idx, phn in enumerate(phns)}

    return phn2idx, idx2phn