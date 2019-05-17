# voice-conversion

This project started with a goal to convert someone's voice to a specific target voice. So called, it's voice style transfer. Traditionally, people deal with this kind of work through two steps -- speech recognition and speech sythesis. My implementation contains these two modules, too. However, the intermediate results are phoneme sequence instead of real words.
To simplify the work, I only used CBHG mentioned in [Tacotron](https://arxiv.org/pdf/1703.10135.pdf) to build those two models mentioned above.


## Phoneme Sequence Labeling Model

The input is mel-scale spectrogram. The output is phoneme squence. I trained the model on TIMIT. According to my evaluation, with only one CBHG module, it could achieve about 75% accuracy on the test set.

## Speech Sythesis Model
This model is like an inverse process of phoneme sequence model. The input is phoneme squence and the output is spectrogram.

First, I made a few test on an anonymous woman's voice in LJSpeech. These results sound ok.

Then I obtained a corpus spoken by Tom Hiddleston. Since the size of the corpus is less than an hour, the results did not turn out to be good enough. But, I really like my work.