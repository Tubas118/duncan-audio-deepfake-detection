import copy
import unittest
import numpy as np
import path
import sys
from numpy.testing import assert_array_equal


# -- from parent directory
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
from config.configuration import ConfigLoader
from preprocessors.abstract_preprocessor import AbstractPreprocessor

from readers.label_reader import readLabelsWithJob


# =============================================================================
ARY_1 = np.ndarray([0, 1, 1, 0, 1, 0, 0, 1, 1, 1])
ARY_2 = np.ndarray([1, 1, 0, 1, 0, 0, 1, 1, 0, 0])
ARY_3 = np.ndarray([1, 0, 1, 0, 0, 1, 1, 1, 0, 1])
ARY_4 = np.ndarray([0, 1, 0, 0, 1, 1, 1, 1, 0, 0])
ARY_ANY = np.ndarray([0, 0, 1, 0, 1, 1, 0, 0, 1, 0])

DATA_1 = [
    -73.02896,  -79.63663,  -80.,       -80.,       -80.,       -80.,
    -80.,       -80.,       -80.,       -80.,       -80.,       -80.,
    -80.,       -80.,       -80.,       -80.,       -80.,       -80.,
    -80.,       -80.,       -80.,       -80.,       -77.08545,  -76.94783,
    -80.,       -80.,       -80.,       -80.,       -80.,       -80.,
    -80.,       -80.,       -80.,       -80.,       -80.,       -80.,
    -80.,       -80.,       -80.,       -80.,       -80.,       -80.,
    -80.,       -80.,       -80.,       -80.,       -80.,       -80.,
    -80.,       -80.,       -80.,       -80.,       -80.,       -80.,
    -78.79813,  -76.95351,  -80.,       -80.,       -80.,       -76.065125,
    -77.116516, -80.,       -76.67018,  -80.,       -80.,       -80.,
    -80.,       -80.,       -80.,       -80.,       -80.,       -80.,
    -80.,       -80.,       -80.,       -80.,       -80.,       -80.,
    -80.,         0.,         0.,         0.,         0.,         0.,
      0.,         0.,         0.,         0.,         0.,         0.,
      0.,         0.,         0.,         0.,         0.,         0.,
      0.,         0.,         0.,         0.,         0.,         0.,
      0.,         0.,         0.,         0.,         0.,         0.,
      0. ]

DATA_2 = [
    -73.02896,  -79.63663,  -80.,       -80.,       -80.,       -80.,
    -80.,       -80.,       -80.,       -80.,       -80.,       -80.,
    -80.,       -80.,       -80.,       -80.,       -77.08545,  -76.94783,
    -80.,       -80.,       -80.,       -80.,       -80.,       -80.,
    -80.,       -80.,       -80.,       -80.,       -80.,       -80.,
    -78.79813,  -76.95351,  -80.,       -80.,       -80.,       -76.065125,
    -77.116516, -80.,       -76.67018,  -80.,       -80.,       -80.,
    -80.,       -80.,       -80.,       -80.,       -80.,       -80.,
    -80.,       -80.,       -80.,       -80.,       -80.,       -80.,
    -80.,       -80.,       -80.,       -80.,       -80.,       -80.,
    -80.,       -80.,       -80.,       -80.,       -80.,       -80.,
    -80.,       -80.,       -80.,       -80.,       -80.,       -80.,
    -80.,       -80.,       -80.,       -80.,       -80.,       -80.,
    -80.,         0.,         0.,         0.,         0.,         0.,
      0.,         0.,         0.,         0.,         0.,         0.,
      0.,         0.,         0.,         0.,         0.,         0.,
      0.,         0.,         0.,         0.,         0.,         0.,
      0.,         0.,         0.,         0.,         0.,         0.,
      0. ]


# =============================================================================
class ImplAbstractPreprocessorForTestingOnly(AbstractPreprocessor):

    # -------------------------------------------------------------------------
    def __init__(self, silent=False, exec_power_to_db=True):
        super().__init__(silent, exec_power_to_db)
        self.featureMap = {
            "LA_T_1272637": copy.deepcopy(DATA_1),
            "LA_T_1000137": copy.deepcopy(DATA_2)
        }

    def __extract_features_singleSource_worker__(self, job, fullDataPath, filename):
        return self.featureMap.get(filename)


# =============================================================================
class TestAbstractPreprocessor(unittest.TestCase):

    # -------------------------------------------------------------------------
    def setUp(self):
        config = ConfigLoader('testvalues/config-for-unit-test.yml')
        self.job = config.getJobConfig("ASVspoof-2019-1")

        self.fullDataPath = "some-path"


    # -------------------------------------------------------------------------
    def test_extract_features_singleSource(self):
        # given
        preprocessor = ImplAbstractPreprocessorForTestingOnly()
        expectedFile1 = copy.deepcopy(ARY_1)
        expectedFileAny = copy.deepcopy(ARY_ANY)
        
        # when #1
        features = preprocessor.extract_features_singleSource(self.job, self.fullDataPath, "1")

        # then #1
        assert_array_equal(features, expectedFile1)

        # when #2
        features = preprocessor.extract_features_singleSource(self.job, self.fullDataPath, "50")

        # then #2
        assert_array_equal(features, expectedFileAny)


    # -------------------------------------------------------------------------
    def test_extract_features_jobSource(self):
        # given
        expectedLabels = {'LA_T_1272637': 1, 'LA_T_1000137': 0}
        expectedFilenames = ['LA_T_1272637', 'LA_T_1000137']
        preprocessor = ImplAbstractPreprocessorForTestingOnly()

        # when
        features = preprocessor.extract_features_jobSource(self.job, self.job.dataPathSuffix)

        # then
        assert_array_equal(DATA_1, features[0][0])
        assert_array_equal(DATA_2, features[0][1])
        self.assertEqual(expectedLabels, features[2])
        self.assertEqual(expectedFilenames, features[3])
        



if __name__ == '__main__':
    unittest.main()