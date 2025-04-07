import os
from config.configuration import Job

def readLabelsWithJob(job: Job):
    labelFileName = job.fullJoinFilePath(job.dataPathRoot, job.labelFilename)
    return readLabelsWithFilename(labelFileName, job.classes)


def readLabelsWithFilename(labelFileName: str, classes) -> list[str]:
    print(f'Loading {labelFileName}...')
    labels = {}

    with open(labelFileName, 'r') as label_file:
        lines = label_file.readlines()

    for line in lines:
        parts = line.strip().split()
        file_name = parts[1]
        label = 1 if parts[-1] == classes[1] else 0
        labels[file_name] = label
        
    return labels