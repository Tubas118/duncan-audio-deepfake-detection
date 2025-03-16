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
        labels = readLabels(job)
        fullDataPath = job.fullJoinFilePath(job.dataPath, _p_)
        print(f"fullDataPath: {fullDataPath}")

        for filename, label in labels.items():
            _X, _y = self.generateMelSpectrogram(job, fullDataPath, filename, label)
            X.append(_X)
            y.append(_y)

        return X, y

    def generateMelSpectrogram(self, job: Job, fullDataPath, filename, label):
        audioSourceFilename = job.fullJoinFilePath(fullDataPath, filename + job.trainingDataExtension)
        print(f"audio file path: {audioSourceFilename}")
        
        audio, _ = librosa.load(audioSourceFilename, sr = job.sampleRate, duration = job.duration)

        mel_spectrogram = librosa.feature.melspectrogram(y = audio, sr = job.sampleRate, n_mels = job.numMels)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        if (mel_spectrogram.shape[1] < job.maxTimeSteps):
            padWidth = ((0, 0), (0, job.maxTimeSteps - mel_spectrogram.shape[1]))
            mel_spectrogram = np.pad(array=mel_spectrogram, pad_width=padWidth, mode='constant')
        else:
            mel_spectrogram = mel_spectrogram[:, :job.maxTimeSteps]

        return mel_spectrogram, label
