import unittest
from unittest import mock
import numpy as np
import path
import sys

# -- from parent directory
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
from configuration.configuration import ConfigLoader
from mel_spectrogram import MelSpectrogramGenerator

class TestGenerateMelSpectrogram(unittest.TestCase):

    def setUp(self):
        pass

    def test_generate_mel_spectrogram(self):
        # given
        config = ConfigLoader('testvalues/config-for-unit-test.yml')
        job = config.getJobConfig(config.activeJobId)
        print(f"path: {directory.parent.parent}")
        filename = "testaudio/LA_T_1272637"
        label = 1
        generator = MelSpectrogramGenerator()

        # when
        X, y = generator.generateMelSpectrogram(job, directory.parent.parent, filename, label)

        # then
        self.assertIsNotNone(X)
        self.assertGreater(len(X), 0)
        self.assertGreater(y, 0)
        

    def test_generate_mel_spectrograms(self):
        # given
        config = ConfigLoader('testvalues/config-for-unit-test.yml')
        job = config.getJobConfig(config.activeJobId)
        rootDir = directory.parent.parent
        job.dataPathRoot = rootDir
        job.dataPathSuffix = "testaudio"
        job.labelFilename = "testvalues/LA.cm.train.trn.txt"
        generator = MelSpectrogramGenerator()

        # when
        xAry, yAry = generator.generateMelSpectrograms(job, job.dataPathSuffix)

        # then
        self.assertEqual(len(xAry), 2)
        self.assertEqual(len(yAry), 2)
        np.testing.assert_array_equal(yAry[0], np.array([0., 1.]))
        np.testing.assert_array_equal(yAry[1], np.array([1., 0.]))

    # @mock.patch("readTrainingLabelsWithJob")
    # def mocked_readTrainingLabelsWithJob(self, mock_method):
    #     mock_method.return_value = [ "1", "2" ]


if __name__ == '__main__':
    unittest.main()