import unittest
import os
import path
import sys

# -- from parent directory
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
from configuration.configuration import ConfigLoader
from model_definitions.model_abstract_definition import ModelAbstractDefinition

class ModelDefinitionForTesting(ModelAbstractDefinition):

    def __init__(self, job):
        super().__init__(job)

    def buildModel(self):
        return None
    
class TestModelAbstractDefinition(unittest.TestCase):

    def setUp(self):
        pass

    # def test_getRootPath(self):
    #     print(os.getcwd())
    #     print(os.pardir)

    def test_copyJob(self):
        # given
        config = ConfigLoader('config.yml')
        job = config.getJobConfig(config.activeJobId)
        checkJobId = job.jobId

        # when #1
        modelDef = ModelDefinitionForTesting(job)

        # then #1
        self.assertEqual(job.__dict__, modelDef.__job__.__dict__)

        # when #2
        job.jobId = f'{job.jobId}_123'

        # then #2
        self.assertNotEqual(job.__dict__, modelDef.__job__.__dict__)
        self.assertEqual(modelDef.__job__.jobId, checkJobId)
        self.assertNotEqual(job.jobId, checkJobId)



if __name__ == '__main__':
    unittest.main()