import unittest
import numpy as np
import path
import sys
from matplotlib import pyplot as plt
from numpy.testing import assert_array_equal
from sklearn.metrics import accuracy_score


# -- from parent directory
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)

from postprocessors.plot_confusion_matrix import PlotConfusionMatrix
from testvalues.test_data_confusion_matrix_plot import ADC_Y_TRUE_DICT, AUDIO_DEEPFAKE_CLASSES, ADC_Y_PRED_1, ADC_Y_PRED_2
from utils.common_test_utils import CALCULATE_EXPECTED_SCORE, CONFUSION_MATRIX_CHART_TITLE


CAT_DOG_CLASSES = ['Cat', 'Dog']
CAT_DOG_FISH_CLASSES = ['Cat', 'Dog', 'Fish']


class TestPlotConfusionMatrix(unittest.TestCase):

    def setUp(self):
        pass

    # -------------------------------------------------------------------------
    def test_concept_catsDogs1(self):
        # given
        classes = CAT_DOG_CLASSES
        y_true = np.array(['Cat', 'Cat', 'Dog', 'Dog', 'Cat', 'Dog'])
        y_pred = np.array(['Cat', 'Cat', 'Cat', 'Dog', 'Cat', 'Cat'])
        expected_cm = [[3, 0], [2, 1]]
        title = CONFUSION_MATRIX_CHART_TITLE()

        # when
        confusionMatrixPlot = PlotConfusionMatrix()
        cm = confusionMatrixPlot.plot(y_true, y_pred, classes, title)
        plt.pause(0.001)    # Brief delay to allow plot to display

        # then
        assert_array_equal(cm, expected_cm)

    # -------------------------------------------------------------------------
    def test_concept_catsDogs2(self):
        # given
        classes = CAT_DOG_CLASSES
        y_true = np.array(['Cat', 'Cat', 'Dog', 'Dog', 'Cat', 'Dog'])
        y_pred = np.array(['Cat', 'Cat', 'Cat', 'Cat', 'Cat', 'Cat'])
        expected_cm = [[3, 0], [3, 0]]
        title = CONFUSION_MATRIX_CHART_TITLE()

        # when
        confusionMatrixPlot = PlotConfusionMatrix()
        cm = confusionMatrixPlot.plot(y_true, y_pred, classes, title)
        plt.pause(0.001)    # Brief delay to allow plot to display

        # then
        assert_array_equal(cm, expected_cm)

    # -------------------------------------------------------------------------
    def test_concept_catsDogsFish1(self):
        # given
        classes = CAT_DOG_FISH_CLASSES
        y_true = np.array(['Cat', 'Cat', 'Fish', 'Dog', 'Cat', 'Fish'])
        y_pred = np.array(['Cat', 'Cat', 'Fish', 'Dog', 'Cat', 'Cat'])
        expected_cm = [[3, 0, 0], [0, 1, 0], [1, 0, 1]]
        title = CONFUSION_MATRIX_CHART_TITLE()

        # when
        confusionMatrixPlot = PlotConfusionMatrix()
        cm = confusionMatrixPlot.plot(y_true, y_pred, classes, title)
        plt.pause(0.001)    # Brief delay to allow plot to display

        # then
        assert_array_equal(cm, expected_cm)

    # -------------------------------------------------------------------------
    @unittest.skip  # confusion_matrix generates [[10]] when expecting [[0,0],[0,10]]
    def test_plotPred1(self):
        # given
        classes = AUDIO_DEEPFAKE_CLASSES
        
        y_true = np.array([ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ])
        y_pred = np.argmax(ADC_Y_PRED_1, axis=1)

        expected_cm = np.array([[0, 0], [0, 10]])
        title = CONFUSION_MATRIX_CHART_TITLE()        

        # when
        confusionMatrixPlot = PlotConfusionMatrix(True)
        cm = confusionMatrixPlot.plot(y_true, y_pred, classes, title)
        plt.pause(0.001)    # Brief delay to allow plot to display

        # then
        assert_array_equal(cm, expected_cm)

    # -------------------------------------------------------------------------
    def test_plotPred2(self):
        # given
        classes = AUDIO_DEEPFAKE_CLASSES
        idx_classes = np.array(range(0, len(classes)))
        y_true = [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ]
        y_pred = np.argmax(ADC_Y_PRED_2, axis=1)
        expected_cm = np.array([[0, 0], [2, 8]])
        title = CONFUSION_MATRIX_CHART_TITLE()        

        # when
        confusionMatrixPlot = PlotConfusionMatrix()
        cm = confusionMatrixPlot.plot(y_true, y_pred, idx_classes, title)
        plt.pause(0.001)    # Brief delay to allow plot to display

        # then
        assert_array_equal(cm, expected_cm)


    # -------------------------------------------------------------------------
    def test_small_eval_data1(self):
        # given
        classes = AUDIO_DEEPFAKE_CLASSES
        idx_classes = np.array(range(0, len(classes)))
        y_true = [0., 1.]
        y_pred = np.argmax([[0., 1.], [1., 0.0001]], axis=1)
        expected_cm = np.array([[0, 1], [1, 0]])
        title = CONFUSION_MATRIX_CHART_TITLE()        

        # when
        confusionMatrixPlot = PlotConfusionMatrix()
        cm = confusionMatrixPlot.plot(y_true, y_pred, idx_classes, title)
        plt.pause(0.001)    # Brief delay to allow plot to display

        # then
        assert_array_equal(cm, expected_cm)



if __name__ == '__main__':
    unittest.main()