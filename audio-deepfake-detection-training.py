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
# runDetail = RunDetails('config-mfcc.yml', 'ASVspoof-2019_training_mfcc')
# runDetail = RunDetails('config.yml', 'ASVspoof-2019_small-eval-1')
runDetail = RunDetails('config.yml', 'Compare-Sksmta-training')

notebookName = 'audio-deepfake-detection-training'
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

# +
config = configuration.ConfigLoader(configFilename)

notebookToPython(notebookName)
job = config.getJobConfig(runJobId)

if (job.newModelGenerated == False):
    raise ValueError("This notebook is meant for training. Select a job without a value for 'persisted-model' set.")
# -

preproc_factory = PreprocessorFactory()
preprocessor: AbstractPreprocessor = preproc_factory.newPreprocessor(job.preprocessor)


X, y_encoded, true_labels = preprocessor.extract_features_jobSource(job, job.dataPathSuffix)

trainingProc = BasicModelTrainingProcessor(job, model_cnn_definition.ModelCnnDefinition)
model, X_train, X_test, y_train, y_test = trainingProc.process(X, y_encoded, 1)

# ### Test Model

# +
from processors.model_evaluation_result import ModelEvaluationResult


evaluationProc = BasicModelEvaluationProcessor(job, model)
results: ModelEvaluationResult = evaluationProc.process(X_test, y_test, true_labels)


# +
from postprocessors.plot_confusion_matrix import ConfusionMatrixDetails, PlotConfusionMatrix
import json

print(f'results.test: {results.test}')
print(f'results.pred: {results.pred}')
print(f'classes: {job.classes}')
print(f'y_encoded: {y_encoded}')
# confusionMatrixPlot = ConfusionMatrixPlot()
# cmDetails: ConfusionMatrixDetails = confusionMatrixPlot.plot(job.classes, results.test, results.pred)

# -

print("\n")
report = evaluationProc.reportSnapshot(trainingProc)
evaluationProc.writeReportToFile(job.persistedModelResults, report)
print(f'Results:\n{report}')
