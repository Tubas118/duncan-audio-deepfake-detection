import numpy as np
from abc import ABC, abstractmethod
from tensorflow.keras.utils import to_categorical

from config.configuration import Job
from readers.label_reader import readLabelsWithJob

class AbstractPreprocessor(ABC):

    def __init__(self, silent):
        if (silent == False):
            print(f'{self.__class__.__name__}')
        
    # -------------------------------------------------------------------------
    def extract_features_singleSource(self, job: Job, fullDataPath, filename):
        return self.__extract_features_singleSource_worker__(job, fullDataPath, filename)
    
    # -------------------------------------------------------------------------
    def extract_features_jobSource(self, job: Job, dataPathSuffix: str, includeTrueLabels = False):
        X = []
        y = []
        source = readLabelsWithJob(job)
        segmentLength = int(len(source) / 20)
        fullDataPath = job.fullJoinFilePath(job.dataPathRoot, dataPathSuffix)
        print(f"fullDataPath: {fullDataPath}")

        true_labels = self.__init_true_labels__(includeTrueLabels)

        for filename, label in source.items():
            _X = self.__extract_features_singleSource_worker__(job, fullDataPath, filename)
            X.append(_X)
            y.append(label)

            if (includeTrueLabels):
                true_labels[filename] = label

            if (segmentLength == 0 or (len(X) % segmentLength) == 0):
                print(f"Loading audio files: {len(X)}")

        X = np.array(X)

        if (job.executeToCategoricalForLabels):
            y = to_categorical(y, job.numClasses)

        print(f"Number of audio files loaded: {len(X)}")

        return X, y, true_labels

    # -------------------------------------------------------------------------
    def __init_true_labels__(self, includeTrueLabels = False):
        if (includeTrueLabels == True):
            return {}

        return None

    # -------------------------------------------------------------------------
    @abstractmethod
    def __extract_features_singleSource_worker__(self, job: Job, fullDataPath, filename):
        pass

    # -------------------------------------------------------------------------
    def __pad_data__(self, source, job: Job):
        if (source.shape[1] < job.maxTimeSteps):
            padWidth = ((0, 0), (0, job.maxTimeSteps - source.shape[1]))
            source = np.pad(array=source, pad_width=padWidth, mode='constant')
        else:
            source = source[:, :job.maxTimeSteps]

        return source
    