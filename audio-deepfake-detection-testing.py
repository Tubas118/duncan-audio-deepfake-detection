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

# ### Setup

# +
from config.configuration import RunDetails

# runDetail = RunDetails('config.yml', 'GitLab-eval-data')
# runDetail = RunDetails('config.yml', 'ASVspoof-2019_testing')
runDetail = RunDetails('config.yml', 'ASVspoof-2019_testing_epoch-100')

notebookName = 'audio-deepfake-detection-testing'
plot_title_suffix = "(Testing)"
# -

configFilename = runDetail.configFilename
runJobId = runDetail.jobId

# +
import joblib
import numpy as np
from tensorflow.keras.utils import to_categorical

import config.configuration as configuration
from notebook_utils import notebookToPython
from postprocessors.plot_confusion_matrix import PlotConfusionMatrix
from postprocessors.plot_precision_recall_curve import PlotPrecisionRecallCurve
from postprocessors.plot_roc_curve import PlotRocCurve
from postprocessors.plot_spectrogram import PlotSpectrogram
from preprocessors.abstract_preprocessor import AbstractPreprocessor
from preprocessors.preprocessor_factory import PreprocessorFactory
from processors.basic_model_evaluation_processor import BasicModelEvaluationProcessor
from readers.label_reader import readTrainingLabelsWithJob

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

model = joblib.load(job.persistedModel)
evaluationProc = BasicModelEvaluationProcessor(job, model)

preproc_factory = PreprocessorFactory()
preprocessor: AbstractPreprocessor = preproc_factory.newPreprocessor(job.preprocessor)

fullDataPath = job.fullJoinFilePath(job.dataPathRoot, job.dataPathSuffix)
labels = readTrainingLabelsWithJob(job)


# ### Model processing of extracted features

def processArrays(X, y):
    _X = np.array(X)
    _y = np.array(y)
    evaluationProc.process(_X, _y)



# +
preprocessed_X_test = []
preprocessed_filenames = []
preprocessed_labels = []
MAX_INDEX_PREPROCESS_X_TEST = 5

X = []
y = []

for filename, label in labels.items():
    _X, _y = preprocessor.extract_features_singleSource(job, fullDataPath, filename, label)
    X.append(_X)
    y.append(_y)

    if (len(preprocessed_X_test) < MAX_INDEX_PREPROCESS_X_TEST):
        preprocessed_X_test.append(_X)
        preprocessed_filenames.append(filename)
        preprocessed_labels.append(label)

    if (job.inputFileBatchSize != None and len(X) >= job.inputFileBatchSize):
        processArrays(X, y)
        X = []
        y = []

if (len(X) > 0):
    processArrays(X, y)
# -

# ### Feature extract spectrogram samples

# +
print(f"Preprocessor: {job.preprocessor}")

plot_spectrogram = PlotSpectrogram()

display_spectrogram_count = len(preprocessed_X_test)
if (display_spectrogram_count > MAX_INDEX_PREPROCESS_X_TEST):
    display_spectrogram_count = MAX_INDEX_PREPROCESS_X_TEST

for idx in range(0, display_spectrogram_count):
    filename = preprocessed_filenames[idx]
    data = preprocessed_X_test[idx]
    title = f"{job.preprocessor}: {filename} ({idx + 1} of {display_spectrogram_count})"
    plot_spectrogram.plot(data, job, title)
# -

# ### Spectrogram with and without "power_to_db" transformation applied

# +
compareIdx = 0
fullDataPath = job.fullJoinFilePath(job.dataPathRoot, job.dataPathSuffix)

preproc_noExec_power_to_db: AbstractPreprocessor = preproc_factory.newPreprocessor(job.preprocessor, exec_power_to_db=False)
X_test_noPowerToDb, _ = preproc_noExec_power_to_db.extract_features_singleSource(job, fullDataPath, preprocessed_filenames[compareIdx], preprocessed_labels[compareIdx])

# +
filename = preprocessed_filenames[compareIdx]
plot_spectrogram.plot(preprocessed_X_test[compareIdx], job, f"{job.preprocessor}: {filename} (with power_to_db)")

plot_spectrogram_noPowerToDb = PlotSpectrogram()
plot_spectrogram_noPowerToDb.plot(X_test_noPowerToDb, job, f"{job.preprocessor}: {filename} (without power_to_db)")
# -

# ### Plots

results = evaluationProc.batchResults[0]

CM_TITLE = f"{PlotConfusionMatrix.DEFAULT_TITLE} {plot_title_suffix}"
cm_plot = PlotConfusionMatrix()
cm_plot.plotFromResults(results, job, CM_TITLE)

RC_TITLE = f"{PlotRocCurve.DEFAULT_TITLE} {plot_title_suffix}"
roc_plot = PlotRocCurve()
roc_plot.plotFromResults(results, RC_TITLE)

PP_TITLE = f"{PlotPrecisionRecallCurve.DEFAULT_TITLE} {plot_title_suffix}"
roc_plot = PlotPrecisionRecallCurve()
roc_plot.plotFromResults(results, PP_TITLE)

# ### Final Results

# +
print("\n")
report = evaluationProc.reportSnapshot()
evaluationProc.writeReportToFile(job.persistedModelResults, report)

print(report)
