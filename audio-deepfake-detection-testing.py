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

notebookName = 'audio-deepfake-detection-testing'
# -

configFilename = runDetail.configFilename
runJobId = runDetail.jobId

# +
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix

import config.configuration as configuration
from preprocessors.mel_spectrogram import MelSpectrogramPreprocessor
from notebook_utils import notebookToPython
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

X_test, y_test, true_labels = preprocessor.extract_features_jobSource(job, job.dataPathSuffix, True)

results = evaluationProc.process(X_test, y_test, true_labels)

# +
from postprocessors.confusion_matrix_plot import ConfusionMatrixPlot


cm_plot = ConfusionMatrixPlot()
cm_plot.plot(trueAry=results.true, predAry=results.pred, classes=job.classes)

# +
# Get the predicted probabilities for the positive class
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve



# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(y_true, results.pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# +
print("\n")
report = evaluationProc.reportSnapshot()
evaluationProc.writeReportToFile(job.persistedModelResults, report)

print(report)
