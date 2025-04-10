import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from config.configuration import Job
from processors.model_evaluation_result import ModelEvaluationResult


# References:
#   1. Chan, J. (2021). Machine learning with Python for beginners: A step-by-step guide to hands-on projects [Kindle edition].
#   2. Derived from https://github.com/sksmta/audio-deepfake-detection/blob/main/main.ipynb
class ConfusionMatrixPlot:

    # -------------------------------------------------------------------------
    def __init__(self):
        pass


    # -------------------------------------------------------------------------
    def plotFromResults(self, results: ModelEvaluationResult, job: Job, title = 'Confusion Matrix'):
        self.plotFromMatrix(results.confusion_matrix, job.classes, title)


    # -------------------------------------------------------------------------
    def plot(self, testAry: np.array, predAry: np.array, classes, title = 'Confusion Matrix'):
        cm = confusion_matrix(testAry, predAry)
        self.plotFromMatrix(cm, classes, title)


    # -------------------------------------------------------------------------
    def plotFromMatrix(self, cm, classes, title = 'Confusion Matrix'):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(title)
        plt.show(block=False)

