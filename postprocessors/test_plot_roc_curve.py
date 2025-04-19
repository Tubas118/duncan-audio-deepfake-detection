import unittest
import numpy as np
import path
import sys
from matplotlib import pyplot as plt


# -- from parent directory
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)

from postprocessors.plot_roc_curve import PlotRocCurve
from processors.model_evaluation_result import ModelEvaluationResult
from testvalues.test_data_confusion_matrix_plot import ADC_Y_PRED_2
from testvalues.test_data_precision_recall_curve import PREC_RECALL_CURVE_1


class TestPlotPrecisionRecallCurve(unittest.TestCase):


    # -------------------------------------------------------------------------
    def setUp(self):
        pass


    # -------------------------------------------------------------------------
    def test_PlotRocCurve_v1(self):
        # given
        plot_title_suffix = 'unit test'
        PP_TITLE = f"{PlotRocCurve.DEFAULT_TITLE} {plot_title_suffix}"
        y_true = [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ]
        y_pred = np.argmax(ADC_Y_PRED_2, axis=1)
        results = ModelEvaluationResult(y_true, y_pred)
        results.roc_fpr = [0., 0.09719483, 1. ]
        results.roc_tpr = [0., 0.93827328, 1. ]
        results.roc_auc = 0.9205392277582833

        # when
        roc_plot = PlotRocCurve(version=1)
        roc_plot.plotFromResults(results, PP_TITLE)

        plt.pause(0.001)    # Brief delay to allow plot to display


    # -------------------------------------------------------------------------
    def test_PlotRocCurve_latest(self):
        # given
        plot_title_suffix = 'unit test'
        PP_TITLE = f"{PlotRocCurve.DEFAULT_TITLE} {plot_title_suffix}"
        y_true = [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ]
        y_pred = np.argmax(ADC_Y_PRED_2, axis=1)
        results = ModelEvaluationResult(y_true, y_pred)
        results.roc_fpr = [0., 0.09719483, 1. ]
        results.roc_tpr = [0., 0.93827328, 1. ]
        results.roc_auc = 0.9205392277582833

        # when
        roc_plot = PlotRocCurve()
        roc_plot.plotFromResults(results, PP_TITLE)

        plt.pause(0.001)    # Brief delay to allow plot to display



if __name__ == '__main__':
    unittest.main()
