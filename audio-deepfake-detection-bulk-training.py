# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: audio-deepfake-detection
#     language: python
#     name: python3
# ---

# +
from config.configuration import RunDetails

# runDetail = RunDetails('config.yml', 'ASVspoof-2019_training')
runDetail = RunDetails('config-mfcc.yml', 'ASVspoof-2019_training_mfcc')

notebookName = 'audio-deepfake-detection-bulk-training'
random_state_lowValue = 1
random_state_highValue = 200
# -

configFilename = runDetail.configFilename
runJobId = runDetail.jobId

import config.configuration as configuration
import model_definitions.model_cnn_definition as model_cnn_definition
from preprocessors.preprocessor_factory import PreprocessorFactory
from notebook_utils import notebookToPython
from processors.basic_model_training_processor import BasicModelTrainingProcessor
from processors.basic_model_evaluation_processor import BasicModelEvaluationProcessor
from processors.bulk_model_training_processor import BulkModelTrainingProcessor

# +
config = configuration.ConfigLoader(configFilename)

notebookToPython(notebookName)
job = config.getJobConfig(runJobId)

if (job.newModelGenerated == False):
    raise ValueError("This notebook is meant for training. Select a job without a value for 'persisted-model' set.")
# -

preproc_factory = PreprocessorFactory()
preprocessor = preproc_factory.newPreprocessor(job.preprocessor)

X, y_encoded = preprocessor.extract_features_multipleSource(job, job.dataPathSuffix)

# +
bulkTrainingProc = BulkModelTrainingProcessor(job,
                                              model_cnn_definition.ModelCnnDefinition,
                                              BasicModelTrainingProcessor,
                                              BasicModelEvaluationProcessor)

bulkTrainingProc.process(random_state_lowValue, random_state_highValue, X, y_encoded, 1)
