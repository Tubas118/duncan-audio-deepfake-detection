import numpy as np
import librosa

from preprocessors.abstract_preprocessor import AbstractPreprocessor

# Derived from https://github.com/sksmta/audio-deepfake-detection/blob/main/main.ipynb
class MelSpectrogramPreprocessor(AbstractPreprocessor):

    # -------------------------------------------------------------------------
    def __init__(self, silent=False, exec_power_to_db=True):
        super().__init__(silent, exec_power_to_db)

    # -------------------------------------------------------------------------
    def __extract_features_singleSource_worker__(self, job, fullDataPath, filename):
        audioSourceFilename = job.fullJoinFilePath(fullDataPath, filename + job.dataExtension)
        
        audio, _ = librosa.load(audioSourceFilename, sr = job.sampleRate, duration = job.duration)

        mel_spectrogram = librosa.feature.melspectrogram(y = audio, sr = job.sampleRate, n_mels = job.numMels)

        if (self.exec_power_to_db):
            mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        mel_spectrogram = self.__pad_data__(mel_spectrogram, job)

        return mel_spectrogram