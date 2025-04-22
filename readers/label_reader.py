import os
from config.configuration import Job

# TODO - Update to pass "job.classes" as a parameter to label reader
def readTrainingLabelsWithJob(job: Job):
    labelFileName = job.fullJoinFilePath(job.dataPathRoot, job.labelFilename)
    return readLabelsWithJob(labelFileName, job.classes, job.positive_class)


# TODO - Update to receive class/label array
def readLabelsWithJob(labelFileName: str, classes: list[str], positive_class: str) -> list[str]:
    print(f'Loading {labelFileName}...')
    labels = {}

    with open(labelFileName, 'r') as label_file:
        lines = label_file.readlines()

    usePositive_class = positive_class
    if (usePositive_class == None):
        usePositive_class = classes[1]

    print(f"readLabelsWithJob: positive class/label={usePositive_class}")
    for line in lines:
        parts = line.strip().split()
        file_name = parts[1]
        label = 1 if parts[-1] == usePositive_class else 0
        labels[file_name] = label
        
    return labels