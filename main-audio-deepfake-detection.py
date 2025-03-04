# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: duncan-audio-deepfake-detection
#     language: python
#     name: python3
# ---

# +
import configuration
from notebook_utils import notebookToPython


config = configuration.ConfigLoader('config.yml')

notebookToPython(config.projectName)
job = config.getJobConfig(config.activeJobId)

# +
from label_reader import readLabels


trainingLabels = readLabels(job)

# +
fullDataPath = job.fullJoinFilePath(job.dataPath, job.trainingDataPath)

for file_name, label in trainingLabels.items():
    audioSourceFilename = job.fullJoinFilePath(fullDataPath, file_name + job.trainingDataExtension)
    print(f'{audioSourceFilename}')
