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

runDetail = RunDetails('config.yml', 'Compare-Sksmta-eval')
# runDetail = RunDetails('config.yml', 'ASVspoof-2019_2025-03-24-1_large-batch')

notebookName = 'audio-deepfake-detection-testing'
# -

configFilename = runDetail.configFilename
runJobId = runDetail.jobId

# +
import joblib

import config.configuration as configuration
from preprocessors.mel_spectrogram import MelSpectrogramPreprocessor
from notebook_utils import notebookToPython
from postprocessors.plot_confusion_matrix import ConfusionMatrixPlot
from postprocessors.plot_roc_curve import RocCurvePlot
from processors.basic_model_evaluation_processor import BasicModelEvaluationProcessor
from processors.model_evaluation_result import ModelEvaluationResult
from readers.label_reader import readLabelsWithJob

# +
config = configuration.ConfigLoader(configFilename)

notebookToPython(notebookName)
job = config.getJobConfig(runJobId)

import json
prettyJson = json.dumps(job.__dict__, indent=4)
print(f"job: {prettyJson}")

if (job.newModelGenerated):
    raise ValueError("This notebook is meant for testing. Select a job with a value for 'persisted-model' set.")
# -

generator = MelSpectrogramPreprocessor()
model = joblib.load(job.persistedModel)
evaluationProc = BasicModelEvaluationProcessor(job, model)

fullDataPath = job.fullJoinFilePath(job.dataPathRoot, job.dataPathSuffix)
y_test = readLabelsWithJob(job)

# +
from preprocessors.abstract_preprocessor import AbstractPreprocessor
from preprocessors.preprocessor_factory import PreprocessorFactory


preproc_factory = PreprocessorFactory()
preprocessor: AbstractPreprocessor = preproc_factory.newPreprocessor(job.preprocessor)
# -

X_test, y_test, true_labels = preprocessor.extract_features_jobSource(job, job.dataPathSuffix)

results: ModelEvaluationResult = evaluationProc.process(X_test, y_test, true_labels)

cm_plot = ConfusionMatrixPlot()
cm_plot.plotFromResults(results, job)

roc_plot = RocCurvePlot()
roc_plot.plotFromResults(results)

# +
print("\n")
report = evaluationProc.reportSnapshot()
evaluationProc.writeReportToFile(job.persistedModelResults, report)

print(report)
