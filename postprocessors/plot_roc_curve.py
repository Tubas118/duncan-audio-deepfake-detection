import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import RocCurveDisplay

from processors.model_evaluation_result import ModelEvaluationResult


# References:
#   1. https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
#   2. Derived from https://github.com/sksmta/audio-deepfake-detection/blob/main/main.ipynb
class PlotRocCurve:

    DEFAULT_TITLE = 'Receiver Operating Characteristic'

    # -------------------------------------------------------------------------
    def __init__(self, version = None):
        self.version = version


    # -------------------------------------------------------------------------
    def plotFromResults(self, results: ModelEvaluationResult, title = DEFAULT_TITLE):
        self.plot(results.roc_fpr, results.roc_tpr, results.roc_auc, title)


    # -------------------------------------------------------------------------
    def plot(self, fpr: np.array, tpr: np.array, roc_auc: float, title = DEFAULT_TITLE):
        match self.version:
            case 1:
                self.__plot_v1__(fpr, tpr, roc_auc, title)
            case _:
                self.__plot__(fpr, tpr, roc_auc, title)


    # -------------------------------------------------------------------------
    def __plot__(self, fpr: np.array, tpr: np.array, roc_auc: float, title = DEFAULT_TITLE):
        # Plot ROC curve
        label = f'ROC curve'    # (area = {roc_auc:,.2f})'
        disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=label)
        disp.plot(color='darkorange')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title(title)
        plt.show(block=False)

    # -------------------------------------------------------------------------
    def __plot_v1__(self, fpr: np.array, tpr: np.array, roc_auc: float, title = DEFAULT_TITLE):
        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:,.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.show(block=False)
