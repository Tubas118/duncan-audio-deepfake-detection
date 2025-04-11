import datetime
import os
import re
import yaml

JOB_EXT: str = ".libjob"
PREPROC_EXT: str = ".pp-bin"
RESULTS_EXT: str = ".txt"

# =============================================================================
class ConfigLoader:

    # -------------------------------------------------------------------------
    def __init__(self, configFilename):
        self.configFilename = configFilename

        with open(self.configFilename, "r") as f:
            self.__configLoader = yaml.safe_load(f.read())

        self.activeJobId = self.__configLoader['active-job-id']
        self.projectName = self.__configLoader['project-name']


    # -------------------------------------------------------------------------
    def getJobConfig(self, jobId):
        selectedJob = self.__configLoader['job-defaults']
        overlayJob = self.__configLoader['jobs'][jobId]
        selectedJob.update(overlayJob)
        return Job(jobId, selectedJob)


# =============================================================================
class Job:

    # -------------------------------------------------------------------------
    def __init__(self, jobId: int, source):
        self.jobId: int = jobId
        # IGNORED: self.inputFileBatchSize: str = source['input-file-batch-size']
        self.outputFolder: str = source['output-folder']
        self.dataPathRootRaw: str = source['data-path-root']
        self.dataPathRoot: str = self.fullFilePath(self.dataPathRootRaw)
        self.dataPathSuffix: str = source['data-path-suffix']
        self.dataExtension: str = source['data-extension']
        self.trainingSplitRandomState: int = source['training-split-random-state']
        self.labelFilename: str = source['label-filename']
        self.executeToCategoricalForLabels = source.get('labels-execute-to-categorical', True)
        self.classes: list[str] = source.get('classes')
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
        self.batchSize: int = source['batch-size']
        self.numEpochs: int = source['num-epochs']
        self.__determine_persistedModelValue__(source, 'persisted-model')
        self.__determine_preprocessDataValue__(source, 'preprocessed-data')

        self.__check_for_output_folder__()


    # -------------------------------------------------------------------------
    def fullJoinFilePath(self, path, filename):
        return self.fullFilePath(os.path.join(path, filename))


    # -------------------------------------------------------------------------
    def fullFilePath(self, filepath):
        expanded = os.path.expandvars(filepath)
        return expanded.replace("\\", "/")
    

    # -------------------------------------------------------------------------
    def __check_for_output_folder__(self):
        if (len(self.outputFolder) > 0):
            self.outputFolder = self.outputFolder.rstrip().lstrip()

            if (self.outputFolder.startswith("/") or ":" in self.outputFolder):
                raise ValueError("Invalid output folder configured")
            
            if (not os.path.exists(self.outputFolder)):
                print(f"Output folder does not exist. Creating '{self.outputFolder}'.")
                os.makedirs(self.outputFolder)
                print("Output folder created.")


    # -------------------------------------------------------------------------
    def __determine_persistedModelValue__(self, source: dict, keyName: str):
        results: "__DerivedFilename__" = self.__determine_filename__(source, keyName, JOB_EXT)
        self.newModelGenerated = results.generateNew
        self.persistedModel: str = results.filenameRoot
        self.persistedModelResults: str = self.__newPersistedModelResultsName__(self.persistedModel)


    # -------------------------------------------------------------------------
    def __determine_preprocessDataValue__(self, source: dict, keyName: str):
        results: "__DerivedFilename__" = self.__determine_filename__(source, keyName, PREPROC_EXT)
        self.newPreprocessData = results.generateNew
        self.preprocessDataFilename = results.filenameRoot

        if (self.newPreprocessData):
            print(f"Generating new preprocessed binary file: {self.preprocessDataFilename}")
        else:
            print(f"Using existing preprocessed binary file: {self.preprocessDataFilename}")


    # -------------------------------------------------------------------------
    def __determine_filename__(self, source: dict, keyName: str, ext: str) -> "__DerivedFilename__":
        checkValue: str = source.get(keyName, "")

        if (len(checkValue) > 0):
            generateNew = False
            useFilename: str = checkValue
        else:
            generateNew = True
            nowStr = datetime.datetime.now().isoformat()
            nowStr = nowStr.replace(":", "-")
            useFilename = f"{self.jobId}_{nowStr}"

            if (len(self.outputFolder) > 0):
                useFilename = self.fullJoinFilePath(self.outputFolder, useFilename)

            useFilename = f"{useFilename}{ext}"

        return __DerivedFilename__(generateNew, useFilename)


    # -------------------------------------------------------------------------
    def __newPersistedModelResultsName__(self, filenameRoot: str = None, generateTimestamp = False):
        if (filenameRoot == None):
            rootFilename = self.persistedModel
        else:
            rootFilename = filenameRoot

        return self.__newFilename__(rootFilename, RESULTS_EXT, generateTimestamp)
    

    # -------------------------------------------------------------------------
    def __newFilename__(self, filenameRoot: str, ext: str, generateTimestamp = False) -> str:
        mid_section = ""
        rootFilename = filenameRoot.removesuffix(JOB_EXT)

        if (generateTimestamp):
            mid_section = datetime.datetime.now().isoformat()
            mid_section = "_" + mid_section.replace(":", "-")

        return rootFilename + mid_section + ext


    # -------------------------------------------------------------------------
    def __to_tuple_2_ints__(self, value: str):
        splitValue = list(filter(None, re.split('[ (,)]', value)))
        num1 = int(splitValue[0])
        num2 = int(splitValue[1])
        return (num1, num2)


# =============================================================================
class RunDetails:

    # -------------------------------------------------------------------------
    def __init__(self, configFilename: str, jobId: str):
        self.configFilename = configFilename
        self.jobId = jobId


# =============================================================================
class __DerivedFilename__:

    def __init__(self, generateNew: bool, filenameRoot: str):
        self.generateNew = generateNew
        self.filenameRoot = filenameRoot