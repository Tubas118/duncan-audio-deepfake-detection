import datetime
import os
import re
import yaml

JOB_EXT: str = ".libjob"
RESULTS_EXT: str = ".txt"

# ======================================================================
class ConfigLoader:

    def __init__(self, configFilename):
        self.configFilename = configFilename

        with open(self.configFilename, "r") as f:
            self.__configLoader = yaml.safe_load(f.read())

        self.activeJobId = self.__configLoader['active-job-id']
        self.projectName = self.__configLoader['project-name']

    def getJobConfig(self, jobId):
        selectedJob = self.__configLoader['job-defaults']
        overlayJob = self.__configLoader['jobs'][jobId]
        selectedJob.update(overlayJob)
        return Job(jobId, selectedJob)
    
# ======================================================================
class Job:

    def __init__(self, jobId: int, source):
        self.jobId: int = jobId
        # IGNORED: self.inputFileBatchSize: str = source['input-file-batch-size']
        self.inputFileBatchSize = None  # Will add back in later
        self.outputFolder: str = source['output-folder']
        self.dataPathRootRaw: str = source['data-path-root']
        self.dataPathRoot: str = self.fullFilePath(self.dataPathRootRaw)
        self.dataPathSuffix: str = source['data-path-suffix']
        self.dataExtension: str = source['data-extension']
        self.trainingSplitRandomState: int = source['training-split-random-state']
        self.labelFilename: str = source['label-filename']
        self.executeToCategoricalForLabels = source.get('labels-execute-to-categorical', True)
        self.classes = source.get('classes')
        self.numClasses: int = len(self.classes)
        self.sampleRate: int = source['sample-rate']
        self.duration: int = source['duration']
        self.numMels: int = source['num-mels']
        self.maxTimeSteps: int = source['max-time-steps']
        self.kernelSize = self.__to_tuple_2_ints__(source['kernel-size'])   # tuple of 2 ints
        self.poolSize = self.__to_tuple_2_ints__(source['pool-size'])       # tuple of 2 ints
        self.optimizer: str = source['optimizer']
        self.loss: str = source['loss']
        self.metrics = source['metrics']
        self.preprocessor: str = source['preprocessor']
        self.batchSize: str = source['batch-size']
        self.numEpochs: str = source['num-epochs']
        self.cv: int = source.get('cv', 5)
        self.__determine_persistedModelValue__(source, 'persisted-model')

        self.__check_for_output_folder__()

    def fullJoinFilePath(self, path, filename):
        return self.fullFilePath(os.path.join(path, filename))

    def fullFilePath(self, filepath):
        expanded = os.path.expandvars(filepath)
        return expanded.replace("\\", "/")
    
    def newPersistedModelResultsName(self, persistedModelRootFilename: str = None, generateTimestamp = False):
            mid_section = ""

            if (persistedModelRootFilename == None):
                rootFilename = self.persistedModel
            else:
                rootFilename = persistedModelRootFilename

            rootFilename = rootFilename.removesuffix(JOB_EXT)

            if (generateTimestamp):
                mid_section = datetime.datetime.now().isoformat()
                mid_section = "_" + mid_section.replace(":", "-")

            return rootFilename + mid_section + RESULTS_EXT

    def __check_for_output_folder__(self):
        if (len(self.outputFolder) > 0):
            self.outputFolder = self.outputFolder.rstrip().lstrip()

            if (self.outputFolder.startswith("/") or ":" in self.outputFolder):
                raise ValueError("Invalid output folder configured")
            
            if (not os.path.exists(self.outputFolder)):
                print(f"Output folder does not exist. Creating '{self.outputFolder}'.")
                os.makedirs(self.outputFolder)
                print("Output folder created.")

    def __determine_persistedModelValue__(self, source, keyName: str):
        checkValue: str = source.get(keyName, "")

        if (len(checkValue) > 0):
            useNewModelGenerated = False
            usePersistedModel: str = checkValue
            usePersistedModelResults: str = self.newPersistedModelResultsName(usePersistedModel, True)
            print(f"Using configured model name: {checkValue}")
        else:
            useNewModelGenerated = True
            nowStr = datetime.datetime.now().isoformat()
            nowStr = nowStr.replace(":", "-")
            persistedModelRootFilename = self.jobId + "_" + nowStr

            if (len(self.outputFolder) > 0):
                persistedModelRootFilename = self.fullJoinFilePath(self.outputFolder, persistedModelRootFilename)

            usePersistedModel: str = persistedModelRootFilename + JOB_EXT
            usePersistedModelResults: str = self.newPersistedModelResultsName(usePersistedModel)
            print(f"Generating new model name: {usePersistedModel}")

        self.newModelGenerated = useNewModelGenerated
        self.persistedModel: str = usePersistedModel
        self.persistedModelResults: str = usePersistedModelResults
        print(f"Assigned model name: {self.persistedModel}")

    def __to_tuple_2_ints__(self, value: str):
        splitValue = list(filter(None, re.split('[ (,)]', value)))
        num1 = int(splitValue[0])
        num2 = int(splitValue[1])
        return (num1, num2)

# ======================================================================
class RunDetails:

    def __init__(self, configFilename: str, jobId: str):
        self.configFilename = configFilename
        self.jobId = jobId

# ======================================================================
class BulkRunDetails(RunDetails):

    @staticmethod
    def DERIVE_BULK_RUN(sourceDetails: RunDetails, preprocessor: str = None, random_state_array = None):
        return BulkRunDetails(sourceDetails.configFilename,
                              sourceDetails.jobId,
                              preprocessor,
                              random_state_array)

    def __init__(self, configFilename: str, jobId: str, preprocessor: str = None, random_state_array = None):
        super().__init__(configFilename, jobId)
        self.preprocessor = preprocessor
        self.random_state_array = random_state_array
