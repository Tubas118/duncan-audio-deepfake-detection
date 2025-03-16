import unittest
import path
import sys

# -- from parent directory
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
from configuration.configuration import ConfigLoader
from model_cnn_definition import ModelCnnDefinition

class TestModelCnnDefinition(unittest.TestCase):

    def setUp(self):
        pass

    def test_buildModel(self):
        # given
        config = ConfigLoader('config.yml')
        job = config.getJobConfig(config.activeJobId)
        width = 109
        channels = 1
        modelDef = ModelCnnDefinition(job, width, channels)

        # when - numMels: 128, width: 109
        model = modelDef.buildModel()

        # then
        self.assertIsNotNone(model)
        self.assertDictEqual(job.__dict__, modelDef.__job__.__dict__)
        


if __name__ == '__main__':
    unittest.main()