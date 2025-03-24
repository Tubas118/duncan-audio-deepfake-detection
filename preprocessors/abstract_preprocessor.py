import numpy as np
from abc import ABC, abstractmethod
from tensorflow.keras.utils import to_categorical

from configuration.configuration import Job
from readers.label_reader import readTrainingLabelsWithJob

class AbstractPreprocessor(ABC):

    # -------------------------------------------------------------------------
    def extract_features_singleSource(self, job: Job, fullDataPath, filename, label):
        X, y = self.__extract_features_singleSource_worker__(job, fullDataPath, filename, label)

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

        if (job.executeToCategoricalForLabels):
            y = to_categorical(y, job.numClasses)

        print(f"Number of audio files load: {len(X)}")

        return X, y

    @abstractmethod
    def __extract_features_singleSource_worker__(self, job: Job, fullDataPath, filename, label):
        pass
