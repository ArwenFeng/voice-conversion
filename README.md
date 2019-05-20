# Voice Conversion

This simple project started with a goal to convert someone's voice to a specific target voice. Traditionally, people deal with this kind of work through two steps -- speech recognition and speech sythesis. My implementation contains these two modules, too. However, the intermediate results are phoneme sequence instead of real words.
To simplify the work, I only used CBHG module mentioned in [Tacotron](https://arxiv.org/pdf/1703.10135.pdf) to build those two models mentioned above.


## Phoneme Sequence Labeling Model

The input is mel-scale spectrogram. The output is phoneme squence. I trained the model on TIMIT. According to my evaluation, with only one CBHG module, it could achieve about 75% accuracy on the test set.

## Speech Sythesis Model
This model is like an inverse process of phoneme sequence model. The input is phoneme squence and the output is spectrogram.

First, I made a few test on an anonymous woman's voice in LJSpeech. These results sound ok. Two examples can be found in test_files with the suffix 'ljs'.

Then I obtained a corpus spoken by Tom Hiddleston. Since the size of the corpus too small (<1h), the results did not turn out to be good enough. Two examples can be found in test_files with the suffix 'dousen'.

## Quick Start
### Installing dependencies

1. Install Python 3.

2. Install the latest version of [TensorFlow](https://www.tensorflow.org/install/) for your platform. For better
   performance, install with GPU support if it's available. This code works with TensorFlow 1.4 and later.

3. Install requirements:
The packages in requirements.txt may not be all necessary, I just freezed my workspace and got tired of cleaning it. 
   ```
   pip install -r requirements.txt
   ```


### Using a pre-trained model

1. **Download and unpack the models**:

   After unpacking, your tree should look like this:
   ```
   voice-conversion
     |- logdir
         |- model.ckpt-6000.data-00000-of-00001
         |- model.ckpt-6000.index
         |- model.ckpt-6000.meta
    |- logdir_dousen
         |- model.ckpt-60000.data-00000-of-00001
         |- model.ckpt-60000.index
         |- model.ckpt-60000.meta
    |- logdir_ljs
         |- model.ckpt-50000.data-00000-of-00001
         |- model.ckpt-50000.index
         |- model.ckpt-50000.meta
   ```


2. **Change the parameters in eval2.py and run it**:
   
   **origin_file** is the origin wav file path you want to convert.
   
   **target_file** is the targetwav file path you want to convert.
   
   **checkpoint_path** is the path you may want to change to convert voice into ljspeech annoymous woman or Tom Hiddleston.
