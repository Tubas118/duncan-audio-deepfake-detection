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

runDetail = RunDetails('config.yml', 'Compare-Sksmta-eval')
# runDetail = RunDetails('config.yml', 'ASVspoof-2019_2025-03-24-1_large-batch')

notebookName = 'audio-deepfake-detection-testing'
# -

configFilename = runDetail.configFilename
runJobId = runDetail.jobId

# +
import joblib

import config.configuration as configuration
from notebook_utils import notebookToPython
from postprocessors.plot_confusion_matrix import PlotConfusionMatrix
from postprocessors.plot_roc_curve import PlotRocCurve
from postprocessors.plot_spectrogram import PlotSpectrogram
from preprocessors.mel_spectrogram import MelSpectrogramPreprocessor
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

# ### Mel Spectrogram preprocessing with "power_to_db" applied

# +
from preprocessors.abstract_preprocessor import AbstractPreprocessor
from preprocessors.preprocessor_factory import PreprocessorFactory


preproc_factory = PreprocessorFactory()
preprocessor: AbstractPreprocessor = preproc_factory.newPreprocessor(job.preprocessor)

# -

X_test, y_test, true_labels, source_filenames = preprocessor.extract_features_jobSource(job, job.dataPathSuffix)

# ### Mel Spectrogram samples

# +
plot_mel_spectrogram = PlotSpectrogram()

display_melSpectrogram_count = len(X_test)

if (display_melSpectrogram_count > 5): display_melSpectrogram_count = 5

for idx in range(0, display_melSpectrogram_count):
    filename = source_filenames[idx]
    data = X_test[idx]
    title = f"Mel Spectrogram: {filename} ({idx + 1} of {display_melSpectrogram_count})"
    plot_mel_spectrogram.plot(data, job, title)
# -

# ### Mel Spectrogram with and without "power_to_db" transformation applied

# +
fullDataPath = job.fullJoinFilePath(job.dataPathRoot, job.dataPathSuffix)

preproc_noExec_power_to_db: AbstractPreprocessor = preproc_factory.newPreprocessor(job.preprocessor, False)
X_test_noPowerToDb = preproc_noExec_power_to_db.extract_features_singleSource(job, fullDataPath, source_filenames[0])

# +

filename = source_filenames[idx]
plot_mel_spectrogram.plot(X_test[0], job, f"Mel Spectrogram: {filename} (with power_to_db)")

plot_mel_spectrogram_noPowerToDb = PlotSpectrogram()
plot_mel_spectrogram_noPowerToDb.plot(X_test_noPowerToDb, job, f"Mel Spectrogram: {filename} (without power_to_db)")
# -

# ### Scoring

results: ModelEvaluationResult = evaluationProc.process(X_test, y_test, true_labels)

# ### Plots

cm_plot = PlotConfusionMatrix()
cm_plot.plotFromResults(results, job)

roc_plot = PlotRocCurve()
roc_plot.plotFromResults(results)

# ### Final Results

# +
print("\n")
report = evaluationProc.reportSnapshot()
evaluationProc.writeReportToFile(job.persistedModelResults, report)

print(report)
