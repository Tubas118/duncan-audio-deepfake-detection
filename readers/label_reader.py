import os
from config.configuration import Job

def readTrainingLabelsWithJob(job: Job):
    labelFileName = job.fullJoinFilePath(job.dataPathRoot, job.labelFilename)
    return readLabelsWithJob(labelFileName)


def readLabelsWithJob(labelFileName: str) -> list[str]:
    print(f'Loading {labelFileName}...')
    labels = {}

    with open(labelFileName, 'r') as label_file:
        lines = label_file.readlines()

    for line in lines:
        parts = line.strip().split()
        file_name = parts[1]
        label = 1 if parts[-1] == "bonafide" else 0
        labels[file_name] = label
        
    return labels