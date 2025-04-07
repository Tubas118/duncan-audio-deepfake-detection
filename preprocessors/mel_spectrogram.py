import numpy as np
import librosa

from preprocessors.abstract_preprocessor import AbstractPreprocessor

# Derived from https://github.com/sksmta/audio-deepfake-detection/blob/main/main.ipynb
class MelSpectrogramPreprocessor(AbstractPreprocessor):

    # -------------------------------------------------------------------------
    def __init__(self, silent=False):
        super().__init__(silent)

    # -------------------------------------------------------------------------
    def __extract_features_singleSource_worker__(self, job, fullDataPath, filename):
        audioSourceFilename = job.fullJoinFilePath(fullDataPath, filename + job.dataExtension)
        
        audio, _ = librosa.load(audioSourceFilename, sr = job.sampleRate, duration = job.duration)

        mel_spectrogram = librosa.feature.melspectrogram(y = audio, sr = job.sampleRate, n_mels = job.numMels)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        if (mel_spectrogram.shape[1] < job.maxTimeSteps):
            padWidth = ((0, 0), (0, job.maxTimeSteps - mel_spectrogram.shape[1]))
            mel_spectrogram = np.pad(array=mel_spectrogram, pad_width=padWidth, mode='constant')
        else:
            mel_spectrogram = mel_spectrogram[:, :job.maxTimeSteps]

        return mel_spectrogram