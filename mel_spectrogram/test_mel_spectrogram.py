import unittest
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
        config = ConfigLoader('config.yml')
        job = config.getJobConfig(config.activeJobId)
        print(f"path: {directory.parent.parent}")
        filename = "testdata/LA_T_1272637"
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
        config = ConfigLoader('config.yml')
        job = config.getJobConfig(config.activeJobId)
        rootDir = directory.parent.parent
        job.dataPath = rootDir
        job.trainingDataPath = "testdata"
        job.trainingLabelFilename = "testlabels/LA.cm.train.trn.txt"
        generator = MelSpectrogramGenerator()

        # when
        xAry, yAry = generator.generateMelSpectrograms(job, job.trainingDataPath)

        # then
        self.assertEqual(len(xAry), 2)
        self.assertEqual(len(yAry), 2)
        self.assertEqual(yAry[0], 1)
        self.assertEqual(yAry[1], 0)


if __name__ == '__main__':
    unittest.main()