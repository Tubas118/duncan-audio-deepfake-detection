from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

from config.configuration import Job
from processors.model_evaluation_result import ModelEvaluationResult


class RocCurveDetails:

    def __init__(self, fpr, tpr, roc_auc):
        self.fpr = fpr
        self.tpr = tpr
        self.roc_auc = roc_auc


# References:
#   1. https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
#   2. Derived from https://github.com/sksmta/audio-deepfake-detection/blob/main/main.ipynb
class RocCurvePlot:

    # -------------------------------------------------------------------------
    def __init__(self, debugOn = False):
        self.debugOn = debugOn


    # -------------------------------------------------------------------------
    def plotFromResults(self, results: ModelEvaluationResult, title = 'Receiver Operating Characteristic'):
        self.plot(results.roc_fpr, results.roc_tpr, results.roc_auc, title)


    # -------------------------------------------------------------------------
    def plot(self, fpr: np.array, tpr: np.array, roc_auc: float, title = 'Receiver Operating Characteristic'):
        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.show()
