import unittest
import numpy as np
import path
import sys
from matplotlib import pyplot as plt


# -- from parent directory
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)

from postprocessors.plot_precision_recall_curve import PlotPrecisionRecallCurve
from processors.model_evaluation_result import ModelEvaluationResult
from testvalues.test_data_confusion_matrix_plot import ADC_Y_PRED_2
from testvalues.test_data_precision_recall_curve import PREC_RECALL_CURVE_1


class TestPlotPrecisionRecallCurve(unittest.TestCase):


    # -------------------------------------------------------------------------
    def setUp(self):
        pass


    # -------------------------------------------------------------------------
    def test_PlotPrecisionRecallCurve(self):
        # given
        plot_title_suffix = 'unit test'
        PP_TITLE = f"{PlotPrecisionRecallCurve.DEFAULT_TITLE} {plot_title_suffix}"
        y_true = [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ]
        y_pred = np.argmax(ADC_Y_PRED_2, axis=1)
        results = ModelEvaluationResult(y_true, y_pred)
        results.precision_recall_curve = PREC_RECALL_CURVE_1

        # when
        roc_plot = PlotPrecisionRecallCurve()
        roc_plot.plotFromResults(results, PP_TITLE)

        plt.pause(15.001)    # Brief delay to allow plot to display



if __name__ == '__main__':
    unittest.main()
