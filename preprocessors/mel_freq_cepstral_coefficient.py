import numpy as np
import librosa

from preprocessors.abstract_preprocessor import AbstractPreprocessor

# Derived from:
# Koul, N. (2024). Ultimate Deepfake detection using Python [Kindle edition]. Orange Education Pvt Ltd. 
class MelFrequencyCepstralCoeffiecient(AbstractPreprocessor):

    # -------------------------------------------------------------------------
    def __init__(self, silent=False):
        super().__init__(silent)

    # -------------------------------------------------------------------------
    def __extract_features_singleSource_worker__(self, job, fullDataPath, filename):
        audioSourceFilename = job.fullJoinFilePath(fullDataPath, filename + job.dataExtension)

        audio, _ = librosa.load(audioSourceFilename, sr = job.sampleRate, duration = job.duration)

        mfccs = librosa.feature.mfcc(y = audio, sr = job.sampleRate, n_mfcc = job.numMels)
        mfccs = self.__pad_data__(mfccs, job)

        return mfccs