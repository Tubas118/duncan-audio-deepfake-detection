import numpy as np
import librosa

# from configuration.configuration import Job
from preprocessors.abstract_preprocessor import AbstractPreprocessor
from readers.label_reader import readTrainingLabelsWithJob

class MelSpectrogramPreprocessor(AbstractPreprocessor):

    # -------------------------------------------------------------------------
    def __init__(self):
        pass

    # # -------------------------------------------------------------------------
    # def extract_features_multipleSource(self, job: Job, dataPathSuffix: str):
    #     X = []
    #     y = []
    #     labels = readTrainingLabelsWithJob(job)
    #     segmentLength = int(len(labels) / 20)
    #     fullDataPath = job.fullJoinFilePath(job.dataPathRoot, dataPathSuffix)
    #     print(f"fullDataPath: {fullDataPath}")

    #     for filename, label in labels.items():
    #         _X, _y = self.__extract_features_singleSource_worker__(job, fullDataPath, filename, label)
    #         X.append(_X)
    #         y.append(_y)
    #         if (segmentLength == 0 or (len(X) % segmentLength) == 0):
    #             print(f"Loading audio files: {len(X)}")

    #     X = np.array(X)
    #     y = np.array(y)

    #     if (job.executeToCategoricalForLabels):
    #         y = to_categorical(y, job.numClasses)

    #     print(f"Number of audio files load: {len(X)}")

    #     return X, y

    # # -------------------------------------------------------------------------
    # def extract_features_singleSource(self, job: Job, fullDataPath, filename, label):
    #     X, y = self.__extract_features_singleSource_worker__(job, fullDataPath, filename, label)

    #     if (job.executeToCategoricalForLabels):
    #         y = to_categorical(y, job.numClasses)

    #     return X, y

    # -------------------------------------------------------------------------
    def __extract_features_singleSource_worker__(self, job, fullDataPath, filename, label):
        audioSourceFilename = job.fullJoinFilePath(fullDataPath, filename + job.dataExtension)
        
        audio, _ = librosa.load(audioSourceFilename, sr = job.sampleRate, duration = job.duration)

        mel_spectrogram = librosa.feature.melspectrogram(y = audio, sr = job.sampleRate, n_mels = job.numMels)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        if (mel_spectrogram.shape[1] < job.maxTimeSteps):
            padWidth = ((0, 0), (0, job.maxTimeSteps - mel_spectrogram.shape[1]))
            mel_spectrogram = np.pad(array=mel_spectrogram, pad_width=padWidth, mode='constant')
        else:
            mel_spectrogram = mel_spectrogram[:, :job.maxTimeSteps]

        return mel_spectrogram, label