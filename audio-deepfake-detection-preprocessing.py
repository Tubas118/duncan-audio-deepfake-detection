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

# ### Setup - Preprocess

# +
from config.configuration import RunDetails

# runDetail = RunDetails('config.yml', 'Compare-Sksmta-eval')
runDetail = RunDetails('config.yml', 'ASVspoof-2019_2025-03-24-1_large-batch')

notebookName = 'audio-deepfake-detection-preprocessing'
# -

configFilename = runDetail.configFilename
runJobId = runDetail.jobId

# +
import json

import config.configuration as configuration
from notebook_utils import notebookToPython
from preprocessors.abstract_preprocessor import AbstractPreprocessor
from preprocessors.preprocess_persistance import PreprocessPersistance
from preprocessors.preprocessor_factory import PreprocessorFactory
from readers.label_reader import readLabelsWithJob

# +
config = configuration.ConfigLoader(configFilename)

notebookToPython(notebookName)
job = config.getJobConfig(runJobId)

prettyJson = json.dumps(job.__dict__, indent=4)
print(f"job: {prettyJson}")

if (job.newPreprocessData == False):
    raise ValueError("This notebook is meant for persisting preprocessed data. Select a job without a value for 'preprocessed-data' set.")
# -

# ### Preprocess

preproc_factory = PreprocessorFactory()
preprocessor: AbstractPreprocessor = preproc_factory.newPreprocessor(job.preprocessor)

X_test, y_test, true_labels, source_filenames = preprocessor.extract_features_jobSource(job, job.dataPathSuffix)

# ### Save and validate

persist = PreprocessPersistance(X_test, y_test, true_labels, source_filenames)
persist.save(job.preprocessDataFilename)

reloaded = persist.load(job.preprocessDataFilename)
if (persist.compare(reloaded)):
    print(f"Successfully saved preprocessed data: {job.preprocessDataFilename}")
else:
    print(f"An problem occurred while attempting to save preprocessed data: {job.preprocessDataFilename}")

