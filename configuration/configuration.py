import datetime
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

    def __init__(self, jobId: int, source):
        self.jobId: int = jobId
        self.dataPathRoot: str = source['data-path-root']
        self.trainingDataPathSuffix: str = source['training-data-path-suffix']
        self.trainingDataExtension: str = source['training-data-extension']
        self.trainingLabelFilename: str = source['training-label-filename']
        self.executeToCategoricalForTrainingLabels = source['training-labels-execute-to-categorical']
        self.numClasses: int = source['num-classes']
        self.sampleRate: int = source['sample-rate']
        self.duration: int = source['duration']
        self.numMels: int = source['num-mels']
        self.maxTimeSteps: int = source['max-time-steps']
        self.optimizer: str = source['optimizer']
        self.loss: str = source['loss']
        self.metrics = source['metrics']
        self.batchSize: str = source['batch-size']
        self.numEpochs: str = source['num-epochs']


        nowStr = datetime.datetime.now().isoformat()
        nowStr = nowStr.replace(":", "-")
        self.persistedModel: str = jobId + nowStr + '.libjob'
        self.persistedModelResults: str = jobId + nowStr + '.txt'

    def fullJoinFilePath(self, path, filename):
        return self.fullFilePath(os.path.join(path, filename))

    def fullFilePath(self, filepath):
        expanded = os.path.expandvars(filepath)
        return expanded.replace("\\", "/")

