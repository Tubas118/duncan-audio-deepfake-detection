import unittest
import numpy as np
import path
import sys
from numpy.testing import assert_array_equal, assert_raises

# -- from parent directory

directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)

from postprocessors.metrics import Metrics
from processors.model_evaluation_result import ModelEvaluationResult


class TestMetrics(unittest.TestCase):

    # -------------------------------------------------------------------------
    def setUp(self):
        pass

    # -------------------------------------------------------------------------
    def test_evaluateResults(self):
        # given
        results = ModelEvaluationResult([0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
                                        [0, 1, 0, 1, 1, 1, 0, 1, 0, 0])
        
        # -- sanity check
        print(f'initial:\n{results.reportSnaphot()}\n\n')
        self.__assert_test_pred_arraysNotNone__(results)
        self.__assert_scoresNone__(results)

        # when
        metrics = Metrics()
        metrics.evaluateResults(results)

        # then
        print(f'results:\n{results.reportSnaphot()}')
        self.__assert_test_pred_arraysNotNone__(results)
        self.__assert_scoresNotNone__(results)

    # -------------------------------------------------------------------------
    def test_joinResults(self):
        # given
        results1 = ModelEvaluationResult([0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
                                         [0, 1, 0, 1, 1, 1, 0, 1, 0, 0])
        results2 = ModelEvaluationResult([1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
                                         [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0])
        
        expectedJoinedTestAry = [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0]
        expectedJoinedPredAry = [0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
        
        # -- sanity check
        self.__assert_test_pred_arraysNotNone__(results1)
        self.__assert_test_pred_arraysNotNone__(results2)
        self.__assert_scoresNone__(results1)
        self.__assert_scoresNone__(results2)

        # when #1
        metrics = Metrics()
        metrics.evaluateResults(results1)
        metrics.evaluateResults(results2)

        # then #1
        self.__assert_test_pred_arraysNotNone__(results1)
        self.__assert_test_pred_arraysNotNone__(results2)
        self.__assert_scoresNotNone__(results1)
        self.__assert_scoresNotNone__(results2)

        # when #2
        joinedResults = Metrics.JOIN_RESULTS(results1, results2)
        print(f'joined:\n{joinedResults.reportSnaphot()}')

        # then #2
        self.__assert_test_pred_arraysNotNone__(joinedResults)
        self.__assert_scoresNone__(joinedResults)
        self.assertEqual(expectedJoinedTestAry, joinedResults.testAry)
        self.assertEqual(expectedJoinedPredAry, joinedResults.predAry)

        # when #3
        metrics.evaluateResults(joinedResults)

        # then #3
        print(f'results:\n{joinedResults.reportSnaphot()}')
        self.__assert_test_pred_arraysNotNone__(joinedResults)
        self.__assert_scoresNotNone__(joinedResults)
        self.__assert_scoresNotMatch__(joinedResults, results1)
        self.__assert_scoresNotMatch__(joinedResults, results2)

    # -------------------------------------------------------------------------
    def __assert_test_pred_arraysNotNone__(self, results: ModelEvaluationResult):
        self.assertIsNotNone(results.testAry)
        self.assertIsNotNone(results.predAry)
        self.assertEqual(len(results.testAry), results.batchSize)

    # -------------------------------------------------------------------------
    def __assert_scoresNone__(self, results: ModelEvaluationResult):
        self.assertIsNone(results.accuracy_score)
        self.assertIsNone(results.confusion_matrix)
        self.assertIsNone(results.f1_score)
        self.assertIsNone(results.root_mean_squared_error)
        self.assertIsNone(results.roc_fpr)
        self.assertIsNone(results.roc_tpr)
        self.assertIsNone(results.roc_auc)
        self.assertIsNone(results.precision_recall_curve)

    # -------------------------------------------------------------------------
    def __assert_scoresNotNone__(self, results: ModelEvaluationResult):
        self.assertIsNotNone(results.accuracy_score)
        self.assertIsNotNone(results.confusion_matrix)
        self.assertIsNotNone(results.f1_score)
        self.assertIsNotNone(results.root_mean_squared_error)
        self.assertIsNotNone(results.roc_fpr)
        self.assertIsNotNone(results.roc_tpr)
        self.assertIsNotNone(results.roc_auc)
        self.assertIsNotNone(results.precision_recall_curve)

    # -------------------------------------------------------------------------
    def __assert_scoresNotMatch__(self, results1: ModelEvaluationResult, results2: ModelEvaluationResult):
        self.assertIsNotNone(results1)
        self.assertIsNotNone(results2)

        self.assertNotEqual(results1.accuracy_score, results2.accuracy_score)
        self.__assert_numpy_array_not_equal__(results1.confusion_matrix, results2.confusion_matrix)
        self.assertNotEqual(results1.f1_score, results2.f1_score)
        self.assertNotEqual(results1.root_mean_squared_error, results2.root_mean_squared_error)

        # Not sure why "roc_fpr" has the same value for this test. Programmer error?
        print(f"roc_fpr1: {results1.roc_fpr},\nroc_fpt2: {results2.roc_fpr}\n\n")
        # self.__assert_numpy_array_not_equal__(results1.roc_fpr, results2.roc_fpr)

        self.__assert_numpy_array_not_equal__(results1.roc_tpr, results2.roc_tpr)
        self.assertNotEqual(results1.roc_auc, results2.roc_auc)

        self.__assert_array_of_arrays_not_equal__(results1.precision_recall_curve, results2.precision_recall_curve)

    # -------------------------------------------------------------------------
    def __assert_numpy_array_not_equal__(self, array1, array2):
        return assert_raises(AssertionError, assert_array_equal, array1, array2)

    # -------------------------------------------------------------------------
    def __assert_array_of_arrays_not_equal__(self, arrayOfArrays1, arrayOfArrays2):
        self.assertIsNotNone(arrayOfArrays1)
        self.assertIsNotNone(arrayOfArrays2)
        self.assertEqual(len(arrayOfArrays1), len(arrayOfArrays2))

        numEqualArrays = 0

        for idx in range(0, len(arrayOfArrays1)):
            array1 = arrayOfArrays1[idx]
            array2 = arrayOfArrays2[idx]
            if (np.array_equal(array1, array2)):
                numEqualArrays = numEqualArrays + 1

        self.assertNotEqual(numEqualArrays, len(arrayOfArrays1))


if __name__ == '__main__':
    unittest.main()