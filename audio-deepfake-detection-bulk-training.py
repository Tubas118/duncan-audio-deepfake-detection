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
from config.configuration import BulkRunDetails

runDetail = BulkRunDetails('config.yml', 'ASVspoof-2019_training')

notebookName = 'audio-deepfake-detection-bulk-training'

# +
# --------------------------------------------------------------
large_runDetail = BulkRunDetails.DERIVE_BULK_RUN(runDetail,
                                                 'mel_spectrogram',
                                                 range(1, 200))

mel_spec_runDetail = BulkRunDetails.DERIVE_BULK_RUN(runDetail,
                                                    'mel_spectrogram',
                                                    [186, 133, 147, 69, 105])

mfcc_runDetail = BulkRunDetails.DERIVE_BULK_RUN(runDetail,
                                                'mfcc',
                                                mel_spec_runDetail.random_state_array)
# --------------------------------------------------------------


# runDetail = large_runDetail
runDetail = mel_spec_runDetail
# runDetail = mfcc_runDetail
# -

configFilename = runDetail.configFilename
runJobId = runDetail.jobId

import config.configuration as configuration
import model_definitions.model_cnn_definition as model_cnn_definition
from preprocessors.abstract_preprocessor import AbstractPreprocessor
from preprocessors.preprocessor_factory import PreprocessorFactory
from notebook_utils import notebookToPython
from processors.basic_model_training_processor import BasicModelTrainingProcessor
from processors.basic_model_evaluation_processor import BasicModelEvaluationProcessor
from processors.bulk_model_training_processor import BulkModelTrainingProcessor

# +
config = configuration.ConfigLoader(configFilename)

notebookToPython(notebookName)
job = config.getJobConfig(runJobId)

if (runDetail.preprocessor != None):
    job.preprocessor = runDetail.preprocessor

print(f'job.preprocessor={job.preprocessor}')

if (job.newModelGenerated == False):
    raise ValueError("This notebook is meant for training. Select a job without a value for 'persisted-model' set.")
# -

preproc_factory = PreprocessorFactory()
preprocessor: AbstractPreprocessor = preproc_factory.newPreprocessor(job.preprocessor)

X, y_encoded = preprocessor.extract_features_multipleSource(job, job.dataPathSuffix)

# +
bulkTrainingProc = BulkModelTrainingProcessor(job,
                                              model_cnn_definition.ModelCnnDefinition,
                                              BasicModelTrainingProcessor,
                                              BasicModelEvaluationProcessor)

bulkTrainingProc.processAsArray(runDetail.random_state_array, X, y_encoded, 1)
