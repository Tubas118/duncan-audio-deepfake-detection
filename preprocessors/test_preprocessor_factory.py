import unittest
import numpy as np
import path
import sys


# -- from parent directory
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
from config.configuration import ConfigLoader
from preprocessors.abstract_preprocessor import AbstractPreprocessor
from preprocessors.mel_spectrogram import MelSpectrogramPreprocessor
from preprocessors.preprocessor_factory import PreprocessorFactory


class TestGenerateMelSpectrogram(unittest.TestCase):

    # -------------------------------------------------------------------------
    def setUp(self):
        pass

    # -------------------------------------------------------------------------
    def test_generate_preprocessors_from_factory(self):
        availablePreprocessorsFactory = PreprocessorFactory()

        for preprocessorId in availablePreprocessorsFactory.availablePreprocessors:
            preprocessorType = availablePreprocessorsFactory.availablePreprocessors.get(preprocessorId)
            print(f'---- Testing: {preprocessorId} - name: {preprocessorType.get("typeName")}')
            self.__basic_preprocessor_testing__(preprocessorId)

    # -------------------------------------------------------------------------
    def __basic_preprocessor_testing__(self, preprocessorTypeId):
        # given
        config = ConfigLoader('testvalues/config-for-unit-test.yml')
        job = config.getJobConfig(config.activeJobId)
        print(f"path: {directory.parent.parent}")
        filename = "testaudio/LA_T_1272637"

        # when #1
        preprocessorFactory = PreprocessorFactory()
        generator = preprocessorFactory.newPreprocessor(preprocessorTypeId)
        
        # then #1
        self.assertIsInstance(generator, AbstractPreprocessor)

        # when #2
        X = generator.extract_features_singleSource(job, directory.parent.parent, filename)

        # then
        self.assertIsNotNone(X)
        self.assertGreater(len(X), 0)


if __name__ == '__main__':
    unittest.main()