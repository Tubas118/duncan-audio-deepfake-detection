import unittest
import path
import sys

# -- from parent directory
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
from config.configuration import ConfigLoader
import model_definitions.model_cnn_definition as model_cnn_definition

class TestModelCnnDefinition(unittest.TestCase):

    def setUp(self):
        pass

    def test_buildModel(self):
        # given
        config = ConfigLoader('testvalues/config-for-unit-test.yml')
        job = config.getJobConfig(config.activeJobId)
        width = 109
        channels = 1
        modelDef = model_cnn_definition.ModelCnnDefinition(job, width, channels)

        # when - numMels: 128, width: 109
        model = modelDef.buildModel()

        # then
        self.assertIsNotNone(model)
        self.assertDictEqual(job.__dict__, modelDef.__job__.__dict__)
        modelDef.printModelDefintions()
        


if __name__ == '__main__':
    unittest.main()