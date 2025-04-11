from typing import Tuple
from parameterized import parameterized
import unittest
import path
import sys

# -- from parent directory
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
# from configuration.configuration import ConfigLoader, Job
import config.configuration as configuration


# =============================================================================
class TestConfiguration(unittest.TestCase):

    # -------------------------------------------------------------------------
    def setUp(self):
        pass


    # -------------------------------------------------------------------------
    def test_checkDataPathRoot(self):
        # given
        config = configuration.ConfigLoader('testvalues/config-for-unit-test.yml')

        # when
        job: configuration.Job = config.getJobConfig("Check-data-path-root")

        # then
        assert "/workspace/Deepfake/data/ASVspoof-2019" in job.dataPathRoot
        assert "$HOMEDRIVE$HOMEPATH" in job.dataPathRootRaw
        assert "$HOMEDRIVE$HOMEPATH" not in job.dataPathRoot
        self.assertIsNotNone(job.persistedModel)
        self.assertIsNotNone(job.persistedModelResults)
        self.assertTrue(job.newModelGenerated)
        print(f"kernel size: {job.kernelSize} - type: {type(job.kernelSize)}")


    # -------------------------------------------------------------------------
    def test_loadConfig_noPersistedModelProvided(self):
        # given
        config = configuration.ConfigLoader('testvalues/config-for-unit-test.yml')

        # when
        jobId = "ASVspoof-2019-1"
        job: configuration.Job = config.getJobConfig(jobId)

        # then
        self.assertIsNotNone(job.persistedModel)
        self.assertIsNotNone(job.persistedModelResults)
        self.assertTrue(job.newModelGenerated)
        self.assertIsInstance(job.kernelSize, Tuple)
        self.assertIsInstance(job.poolSize, Tuple)

        self.__assert_nameWithJobId__(job.persistedModel, jobId, ".libjob")
        self.__assert_nameWithJobId__(job.persistedModelResults, jobId, ".txt")

        assert "2025-03-16T12-41-11.676368" not in job.persistedModel
        assert "2025-03-16T12-41-11.676368" not in job.persistedModelResults


    # -------------------------------------------------------------------------
    def test_loadConfig_persistedModelProvided(self):
        # given
        config = configuration.ConfigLoader('testvalues/config-for-unit-test.yml')

        # when
        jobId = "ASVspoof-2019-2"
        job: configuration.Job = config.getJobConfig(jobId)

        # then
        self.assertIsNotNone(job.persistedModel)
        self.assertIsNotNone(job.persistedModelResults)
        self.assertFalse(job.newModelGenerated)

        self.__assert_nameWithoutJobId__(job.persistedModel, jobId, ".libjob")
        self.__assert_nameWithoutJobId__(job.persistedModelResults, jobId, ".txt")

        assert "2025-03-16T12-41-11.676368" in job.persistedModel
        assert "2025-03-16T12-41-11.676368" in job.persistedModelResults


    # -------------------------------------------------------------------------
    def test_loadConfig_noPersistedPreprocessedDataProvided(self):
        # given
        config = configuration.ConfigLoader('testvalues/config-for-unit-test.yml')

        # when
        jobId = "ASVspoof-2019-1"
        job: configuration.Job = config.getJobConfig(jobId)

        # then
        self.assertIsNotNone(job.preprocessDataFilename)
        self.assertTrue(job.newPreprocessData)
        self.__assert_nameWithJobId__(job.preprocessDataFilename, jobId, ".pp-bin")
        assert "2025-03-16T12-41-11.676368" not in job.preprocessDataFilename


    # -------------------------------------------------------------------------
    def test_loadConfig_persistedPreprocessedDataProvided(self):
        # given
        config = configuration.ConfigLoader('testvalues/config-for-unit-test.yml')

        # when
        jobId = "ASVspoof-2019-2"
        job: configuration.Job = config.getJobConfig(jobId)

        # then
        self.assertIsNotNone(job.preprocessDataFilename)
        self.assertFalse(job.newPreprocessData)
        self.__assert_nameWithoutJobId__(job.preprocessDataFilename, jobId, ".pp-bin")
        assert "2025-03-16T12-41-11.676368" in job.preprocessDataFilename


    # -------------------------------------------------------------------------
    @parameterized.expand([
        ("(2, 3)", (2, 3)),
        ("(3,4)", (3, 4)),
        (" (  10 , 15   )  ", (10, 15))
    ])
    def test_to_tuple(self, input, expected):
        # given
        config = configuration.ConfigLoader('testvalues/config-for-unit-test.yml')
        job: configuration.Job = config.getJobConfig("ASVspoof-2019-2")

        # when
        tuple = job.__to_tuple_2_ints__(input)

        # then
        self.assertIsInstance(tuple, Tuple)
        self.assertEqual(expected, tuple)


    # -------------------------------------------------------------------------
    def __assert_nameWithJobId__(self, value: str, jobId: str, ext: str):
        print(f"jobId: {jobId}, ext: {ext}, value: {value}")
        assert jobId in value
        assert f"{ext}" in value
        assert f"{ext}{ext}" not in value


    # -------------------------------------------------------------------------
    def __assert_nameWithoutJobId__(self, value: str, jobId: str, ext: str):
        print(f"jobId: {jobId}, ext: {ext}, value: {value}")
        assert jobId not in value
        assert f"{ext}" in value
        assert f"{ext}{ext}" not in value


if __name__ == '__main__':
    unittest.main()