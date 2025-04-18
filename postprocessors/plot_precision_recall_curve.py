import numpy as np
from matplotlib import pyplot as plt

from config.configuration import Job
from processors.model_evaluation_result import ModelEvaluationResult

# Derived from https://www.datacamp.com/tutorial/precision-recall-curve-tutorial
class PlotPrecisionRecallCurve:

    DEFAULT_TITLE = 'Precision-Recall curve'

    # -------------------------------------------------------------------------
    def __init__(self):
        pass

    # -------------------------------------------------------------------------
    def plotFromResults(self, results: ModelEvaluationResult, job: Job, title = DEFAULT_TITLE):
        precision = results.precision_recall_curve[0]
        recall = results.precision_recall_curve[1]
        self.plot(precision, recall, title)

    # -------------------------------------------------------------------------
    def plot(self, precision: np.ndarray, recall: np.ndarray, title = DEFAULT_TITLE):
        plt.plot(recall, precision, color='darkorange', lw=2, label='Avg. Precision = %0.2f' % precision[1])
        plt.ylabel("Precision")
        plt.xlabel("Recall")
        plt.title(title)
        plt.legend(loc="lower left")
        plt.show