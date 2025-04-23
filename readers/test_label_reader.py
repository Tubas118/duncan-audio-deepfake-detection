import unittest
import path
import sys


# -- from parent directory
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)

import config.configuration as configuration
from readers.label_reader import readTrainingLabelsWithJob


class TestPlotPrecisionRecallCurve(unittest.TestCase):

    # -------------------------------------------------------------------------
    def setUp(self):
        config = configuration.ConfigLoader('testvalues/config-for-unit-test.yml')
        self.job = config.getJobConfig(config.activeJobId)

    # -------------------------------------------------------------------------
    def test_labelReader_bonafidePositive(self):
        # given
        expectedLabels = {'LA_E_1000147': 0, 'LA_E_2267312': 1, 'LA_E_1007069': 0, 'LA_E_9521934': 0, 'LA_E_5169845': 1, 'LA_E_2601971': 0, 'LA_E_4785445': 1, 'LA_E_4065507': 0, 'LA_E_1644479': 0, 'LA_E_4453325': 0}
        self.job.positive_class = "bonafide"
        self.job.labelFilename = "testvalues/sksmta.train.trn.txt"

        # when
        labels = readTrainingLabelsWithJob(self.job)

        # then
        self.assertEqual(labels, expectedLabels)

    # -------------------------------------------------------------------------
    def test_labelReader_spoofPositive(self):
        # given
        expectedLabels = {'LA_E_1000147': 1, 'LA_E_2267312': 0, 'LA_E_1007069': 1, 'LA_E_9521934': 1, 'LA_E_5169845': 0, 'LA_E_2601971': 1, 'LA_E_4785445': 0, 'LA_E_4065507': 1, 'LA_E_1644479': 1, 'LA_E_4453325': 1}
        self.job.positive_class = "spoof"
        self.job.labelFilename = "testvalues/sksmta.train.trn.txt"

        # when
        labels = readTrainingLabelsWithJob(self.job)

        # then
        self.assertEqual(labels, expectedLabels)

    # -------------------------------------------------------------------------
    def test_labelReader_nonePositive(self):
        # given
        expectedLabels = {'LA_E_1000147': 0, 'LA_E_2267312': 1, 'LA_E_1007069': 0, 'LA_E_9521934': 0, 'LA_E_5169845': 1, 'LA_E_2601971': 0, 'LA_E_4785445': 1, 'LA_E_4065507': 0, 'LA_E_1644479': 0, 'LA_E_4453325': 0}
        self.job.positive_class = None
        self.job.labelFilename = "testvalues/sksmta.train.trn.txt"

        # when
        labels = readTrainingLabelsWithJob(self.job)

        # then
        self.assertEqual(labels, expectedLabels)

    # -------------------------------------------------------------------------
    def test_labelReader_nonePositiveFromConfigJob(self):
        # given
        expectedLabels = {'LA_E_1000147': 0, 'LA_E_2267312': 1, 'LA_E_1007069': 0, 'LA_E_9521934': 0, 'LA_E_5169845': 1, 'LA_E_2601971': 0, 'LA_E_4785445': 1, 'LA_E_4065507': 0, 'LA_E_1644479': 0, 'LA_E_4453325': 0}
        config = configuration.ConfigLoader('testvalues/config-for-unit-test.yml')
        self.job = config.getJobConfig('ASVspoof-2019-3_positive-class-none-1')

        # when
        labels = readTrainingLabelsWithJob(self.job)

        # then
        self.assertEqual(labels, expectedLabels)



if __name__ == '__main__':
    unittest.main()
