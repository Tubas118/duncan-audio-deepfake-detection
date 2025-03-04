import os
import yaml

class ConfigLoader:

    def __init__(self, configFilename):
        self.configFilename = configFilename

        with open(self.configFilename, "r") as f:
            self.__configLoader = yaml.safe_load(f.read())

        self.activeJobId = self.__configLoader['active-job-id']
        self.projectName = self.__configLoader['project-name']

    def getJobConfig(self, jobId):
        selectedJob = self.__configLoader['jobs'][jobId]
        return Job(jobId, selectedJob)
    
# ======================================================================
class Job:

    def __init__(self, jobId, source):
        self.jobId = jobId
        self.dataPath = source['data-path']
        self.trainingDataPath = source['training-data-path']
        self.trainingDataExtension = source['training-data-extension']
        self.trainingLabelFilename = source['training-label-filename']
        self.persistedModel = jobId + '.libjob'

    def fullJoinFilePath(self, path, filename):
        return self.fullFilePath(os.path.join(path, filename))

    def fullFilePath(self, filepath):
        expanded = os.path.expandvars(filepath)
        return expanded.replace("\\", "/")

