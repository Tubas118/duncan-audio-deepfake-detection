import numpy as np
from abc import ABC, abstractmethod
from tensorflow.keras.utils import to_categorical

from config.configuration import Job
from readers.label_reader import readTrainingLabelsWithJob

class AbstractPreprocessor(ABC):

    def __init__(self, silent, exec_power_to_db=True):
        if (silent == False):
            print(f'{self.__class__.__name__}')

        self.exec_power_to_db = exec_power_to_db

    # -------------------------------------------------------------------------
    def extract_features_singleSource(self, job: Job, fullDataPath, filename, label):
        X, y = self.__extract_features_singleSource_worker__(job, fullDataPath, filename, label)

        # TODO: running "to_categorical(...)" is necessary to reformat "y" to something
        #   usable for training the model. Consider removing it as configurable.
        if (job.executeToCategoricalForLabels):
            y = to_categorical(y, job.numClasses)

        return X, y
    
    # -------------------------------------------------------------------------
    def extract_features_multipleSource(self, job: Job, dataPathSuffix: str):
        X = []
        y = []
        labels = readTrainingLabelsWithJob(job)
        segmentLength = int(len(labels) / 20)
        fullDataPath = job.fullJoinFilePath(job.dataPathRoot, dataPathSuffix)
        print(f"fullDataPath: {fullDataPath}")

        for filename, label in labels.items():
            _X, _y = self.__extract_features_singleSource_worker__(job, fullDataPath, filename, label)
            X.append(_X)
            y.append(_y)
            if (segmentLength == 0 or (len(X) % segmentLength) == 0):
                print(f"Loading audio files: {len(X)}")

        X = np.array(X)
        y = np.array(y)

        # TODO: running "to_categorical(...)" is necessary to reformat "y" to something
        #   usable for training the model. Consider removing it as configurable.
        if (job.executeToCategoricalForLabels):
            y = to_categorical(y, job.numClasses)

        print(f"Number of audio files load: {len(X)}")

        return X, y

    # -------------------------------------------------------------------------
    @abstractmethod
    def __extract_features_singleSource_worker__(self, job: Job, fullDataPath, filename, label):
        pass

    # -------------------------------------------------------------------------
    def __pad_data__(self, source, job: Job):
        if (source.shape[1] < job.maxTimeSteps):
            padWidth = ((0, 0), (0, job.maxTimeSteps - source.shape[1]))
            source = np.pad(array=source, pad_width=padWidth, mode='constant')
        else:
            source = source[:, :job.maxTimeSteps]

        return source
    