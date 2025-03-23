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
notebookName = 'audio-deepfake-detection-testing'
# runJobId = 'ASVspoof-2019_small-eval-1'
# runJobId = 'ASVspoof-2019_eval-1'
# runJobId = 'ASVspoof-2019_eval-1_large-batch'
runJobId = 'ASVspoof-2019_eval-1_huge-batch'
# runJobId = 'ASVspoof-2019_training'     # This should fail

import joblib
import numpy as np
from tensorflow.keras.utils import to_categorical

import configuration.configuration as configuration
from mel_spectrogram.mel_spectrogram import MelSpectrogramGenerator
from notebook_utils import notebookToPython
from processors.basic_model_evaluation_processor import BasicModelEvaluationProcessor
from readers.label_reader import readTrainingLabelsWithJob

# +
config = configuration.ConfigLoader('config.yml')

notebookToPython(notebookName)
job = config.getJobConfig(runJobId)

import json
prettyJson = json.dumps(job.__dict__, indent=4)
print(f"job: {prettyJson}")

if (job.newModelGenerated):
    raise ValueError("This notebook is meant for testing. Select a job with a value for 'persisted-model' set.")
# -

generator = MelSpectrogramGenerator()
model = joblib.load(job.persistedModel)
evaluationProc = BasicModelEvaluationProcessor(job, model)

fullDataPath = job.fullJoinFilePath(job.dataPathRoot, job.dataPathSuffix)
labels = readTrainingLabelsWithJob(job)


def processArrays(X, y):
    _X = np.array(X)
    _y = np.array(y)
    evaluationProc.process(_X, _y)



# +
X = []
y = []

for filename, label in labels.items():
    _X, _y = generator.generateMelSpectrogram(job, fullDataPath, filename, label)
    X.append(_X)
    y.append(_y)

    if (len(X) >= job.inputFileBatchSize):
        processArrays(X, y)
        X = []
        y = []

if (len(X) > 0):
    processArrays(X, y)

print("\n")
report = evaluationProc.reportSnapshot()
evaluationProc.writeReportToFile(job.persistedModelResults, report)
