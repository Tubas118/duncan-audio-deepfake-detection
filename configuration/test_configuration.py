import unittest

from configuration import ConfigLoader, Job


class TestConfiguration(unittest.TestCase):

    def setUp(self):
        pass

    def test_loadConfig_noPersistedModelProvided(self):
        # given
        config = ConfigLoader('testvalues/config-for-unit-test.yml')

        # when
        job: Job = config.getJobConfig("ASVspoof-2019-1")

        # then
        assert "/workspace/Deepfake/data/ASVspoof-2019" in job.dataPathRoot
        assert "$HOMEDRIVE$HOMEPATH" in job.dataPathRootRaw
        assert "$HOMEDRIVE$HOMEPATH" not in job.dataPathRoot
        self.assertIsNotNone(job.persistedModel)
        self.assertIsNotNone(job.persistedModelResults)

        assert "ASVspoof-2019-1" in job.persistedModel
        assert ".libjob" in job.persistedModel
        assert "ASVspoof-2019-1" in job.persistedModelResults
        assert ".txt" in job.persistedModelResults

        assert "2025-03-16T12-41-11.676368" not in job.persistedModel
        assert "2025-03-16T12-41-11.676368" not in job.persistedModelResults

    def test_loadConfig_persistedModelProvided(self):
        # given
        config = ConfigLoader('testvalues/config-for-unit-test.yml')

        # when
        job: Job = config.getJobConfig("ASVspoof-2019-2")

        # then
        assert "/workspace/Deepfake/data/ASVspoof-2019" in job.dataPathRoot
        assert "$HOMEDRIVE$HOMEPATH" in job.dataPathRootRaw
        assert "$HOMEDRIVE$HOMEPATH" not in job.dataPathRoot
        self.assertIsNotNone(job.persistedModel)
        self.assertIsNotNone(job.persistedModelResults)

        assert "ASVspoof-2019-1" in job.persistedModel
        assert ".libjob" in job.persistedModel
        assert "ASVspoof-2019-1" in job.persistedModelResults
        assert ".txt" in job.persistedModelResults

        assert "2025-03-16T12-41-11.676368" in job.persistedModel
        assert "2025-03-16T12-41-11.676368" in job.persistedModelResults


if __name__ == '__main__':
    unittest.main()