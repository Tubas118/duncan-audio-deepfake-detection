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

runDetail = RunDetails('config.yml', 'GitLab-training-data')

notebookName = 'audio-deepfake-detection-training'
# -

configFilename = runDetail.configFilename
runJobId = runDetail.jobId

import config.configuration as configuration
import model_definitions.model_cnn_definition as model_cnn_definition
from preprocessors.preprocessor_factory import PreprocessorFactory
from notebook_utils import notebookToPython
from processors.basic_model_training_processor import BasicModelTrainingProcessor
from processors.basic_model_evaluation_processor import BasicModelEvaluationProcessor

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

trainingProc = BasicModelTrainingProcessor(job, model_cnn_definition.ModelCnnDefinition)
model, X_train, X_test, y_train, y_test = trainingProc.process(X, y_encoded, 1)

# ### Test Model

evaluationProc = BasicModelEvaluationProcessor(job, model)
evaluationProc.process(X_test, y_test)


print("\n")
report = evaluationProc.reportSnapshot(trainingProc)
evaluationProc.writeReportToFile(job.persistedModelResults, report)
