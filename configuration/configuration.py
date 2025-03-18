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
        self.dataPathRootRaw: str = source['data-path-root']
        self.dataPathRoot: str = self.fullFilePath(self.dataPathRootRaw)
        self.dataPathSuffix: str = source['data-path-suffix']
        self.dataExtension: str = source['data-extension']
        self.labelFilename: str = source['label-filename']
        self.executeToCategoricalForLabels = source.get('labels-execute-to-categorical', True)
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
        self.__determine_persistedModelValue__(source, 'persisted-model')

    def fullJoinFilePath(self, path, filename):
        return self.fullFilePath(os.path.join(path, filename))

    def fullFilePath(self, filepath):
        expanded = os.path.expandvars(filepath)
        return expanded.replace("\\", "/")

    def __determine_persistedModelValue__(self, source, keyName: str):
        checkValue: str = source.get(keyName, "")
        jobExt: str = ".libjob"
        resultsExt: str = ".txt"

        if (len(checkValue) > 0):
            print(f"Using configured model name: {checkValue}")
            useNewModelGenerated = False
            usePersistedModel: str = checkValue
            usePersistedModelResults: str = usePersistedModel.removesuffix(jobExt) + resultsExt
        else:
            useNewModelGenerated = True
            nowStr = datetime.datetime.now().isoformat()
            nowStr = nowStr.replace(":", "-")
            persistedModelRootFilename = self.jobId + "_" + nowStr
            usePersistedModel: str = persistedModelRootFilename + jobExt
            usePersistedModelResults: str = persistedModelRootFilename + resultsExt
            print(f"Generating new model name: {usePersistedModel}")

        self.newModelGenerated = useNewModelGenerated
        self.persistedModel: str = usePersistedModel
        self.persistedModelResults: str = usePersistedModelResults
        print(f"Assigned model name: {self.persistedModel}")