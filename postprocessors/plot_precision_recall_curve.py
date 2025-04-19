import numpy as np
from matplotlib import pyplot as plt
# from sklearn.metrics import PrecisionRecallDisplay

from config.configuration import Job
from processors.model_evaluation_result import ModelEvaluationResult

# Derived from https://www.datacamp.com/tutorial/precision-recall-curve-tutorial
class PlotPrecisionRecallCurve:

    DEFAULT_TITLE = 'Precision-Recall curve'

    # -------------------------------------------------------------------------
    def __init__(self):
        pass

    # -------------------------------------------------------------------------
    def plotFromResults(self, results: ModelEvaluationResult, title = DEFAULT_TITLE):
        precision = results.precision_recall_curve[0]
        recall = results.precision_recall_curve[1]
        self.plot(precision, recall, title)

    # # -------------------------------------------------------------------------
    # def plotFromEstimator(self, results: ModelEvaluationResult, title = DEFAULT_TITLE):
    #     disp = PrecisionRecallDisplay.from_predictions(results.testAry, results.predAry, name="Best")
    #     disp.ax_.legend()
    #     disp.plot()

    # -------------------------------------------------------------------------
    def plot(self, precision: np.ndarray, recall: np.ndarray, title = DEFAULT_TITLE):
        # disp = PrecisionRecallDisplay(precision=precision, recall=recall)   #, name="Avg. Precision")
        # disp.ax_.legend()
        # disp.plot()
        # plt.plot(recall, precision, color='darkorange', lw=2, label='Avg. Precision = %0.2f' % precision[1])
        plt.fill_between(recall, precision, alpha=0.2)
        plt.ylabel("Precision")
        plt.xlabel("Recall")
        plt.title(title)

        legend = [ f"Precision = {precision[1]:,.2f}\nRecall = {recall[1]:,.2f}" ]
        plt.legend(legend, loc="lower left")
        plt.show(block=False)