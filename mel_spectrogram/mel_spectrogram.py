import sys
import numpy as np
import librosa

from configuration.configuration import Job
from readers.label_reader import readLabels

class MelSpectrogramGenerator:

    def __init__(self):
        pass

    def generateMelSpectrograms(self, job: Job, _p_: str):
        X = []
        y = []

        return X, y

    def generateMelSpectrogram(self, job: Job, fullDataPath, filename, label):
        return np.array([]), []
