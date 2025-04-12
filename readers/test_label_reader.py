import copy
import unittest
from unittest.mock import patch
import path
import sys

# -- from parent directory
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
from config.configuration import ConfigLoader
from readers.label_reader import readLabelsWithFilename, readLabelsWithJob


# =============================================================================
class TestLabelReader(unittest.TestCase):

    # -------------------------------------------------------------------------
    def setUp(self):
        pass

    # -------------------------------------------------------------------------
    def test_labelReader(self):
        # given
        classes = ["spoof", "bonafide"]
        expectedLabels = {
            "LA_E_1000147": 0,
            "LA_E_2267312": 1,
            "LA_E_1007069": 0,
            "LA_E_9521934": 0,
            "LA_E_5169845": 1,
            "LA_E_2601971": 0,
            "LA_E_4785445": 1,
            "LA_E_4065507": 0,
            "LA_E_1644479": 0,
            "LA_E_4453325": 0
        }
        
        # when
        labels = readLabelsWithFilename("testvalues/sksmta.train.trn.txt", classes)

        # then
        self.assertEqual(labels, expectedLabels)

    # -------------------------------------------------------------------------
    @patch('readers.label_reader.readLabelsWithFilename')
    def test_mockedReadLabelsWithFilename(self, mockedReadLabelsWithFilename):
        # given
        config = ConfigLoader('testvalues/config-for-unit-test.yml')
        job = config.getJobConfig("ASVspoof-2019-1")

        mockedLabels = {
            "LA_T_1272637": 2,
            "LA_T_1000137": 2
        }
        mockedReadLabelsWithFilename.return_value = copy.copy(mockedLabels)

        # when
        labels = readLabelsWithJob(job)

        # then
        self.assertEqual(labels, mockedLabels)



if __name__ == '__main__':
    unittest.main()