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

notebookName = 'audio-deepfake-detection-bulk-training'
runJobId = 'ASVspoof-2019_training'
# runJobId = 'ASVspoof-2019_small-eval-1'     # This should fail
random_state_lowValue = 4
random_state_highValue = 200

import configuration.configuration as configuration
import model_definitions.model_cnn_definition as model_cnn_definition
from preprocessors.mel_spectrogram import MelSpectrogramPreprocessor
from notebook_utils import notebookToPython
from processors.basic_model_training_processor import BasicModelTrainingProcessor
from processors.basic_model_evaluation_processor import BasicModelEvaluationProcessor
from processors.bulk_model_training_processor import BulkModelTrainingProcessor


# +
config = configuration.ConfigLoader('config.yml')

notebookToPython(notebookName)
job = config.getJobConfig(runJobId)

if (job.newModelGenerated == False):
    raise ValueError("This notebook is meant for training. Select a job without a value for 'persisted-model' set.")
# -

generator = MelSpectrogramPreprocessor()
X, y_encoded = generator.extract_features_multipleSource(job, job.dataPathSuffix)

# +
bulkTrainingProc = BulkModelTrainingProcessor(job,
                                              model_cnn_definition.ModelCnnDefinition,
                                              BasicModelTrainingProcessor,
                                              BasicModelEvaluationProcessor)

bulkTrainingProc.process(random_state_lowValue, random_state_highValue, X, y_encoded, 1)
