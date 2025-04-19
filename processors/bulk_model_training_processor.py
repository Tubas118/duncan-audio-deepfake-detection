import pytz
from datetime import datetime

from config.configuration import JOB_EXT, Job
from processors.abstract_model_processor import AbstractModelProcessor
from processors.abstract_model_training_processor import AbstractModelTrainingProcessor
from processors.basic_model_evaluation_processor import BasicModelEvaluationProcessor

class BulkModelTrainingProcessor(AbstractModelProcessor):

    # -------------------------------------------------------------------------
    def __init__(self, job: Job, modelDefType, trainingProcType, evalProcType):
        super().__init__(job)
        self.resetStatistics()
        self.modelDefType = modelDefType
        self.trainingProcType = trainingProcType
        self.evalProcType = evalProcType

    # -------------------------------------------------------------------------
    def resetStatistics(self):
        self.jobStartTime = None
        self.reportSnapshotList = []

    # -------------------------------------------------------------------------
    def processAsRange(self, random_state_lowValue: int, random_state_highValue: int, X, y_encoded, channels, test_size = 0.2):
        if (random_state_highValue < random_state_lowValue):
            raise ValueError(f"random_state_lowValue: {random_state_lowValue}, random_state_highValue: {random_state_highValue} - high value below low value")
        
        self.process(range(random_state_lowValue, random_state_highValue + 1), X, y_encoded, channels, test_size)

    # -------------------------------------------------------------------------
    def processAsArray(self, random_state_array, X, y_encoded, channels, test_size = 0.2):
        if (self.jobStartTime == None):
            self.jobStartTime = datetime.now(pytz.utc)

        origPersistedModel = self.__job__.persistedModel
        persistedModelRoot = origPersistedModel.removesuffix(JOB_EXT)

        try:
            for random_state in random_state_array:
                self.__process_single__(persistedModelRoot, random_state, X, y_encoded, channels, test_size)
        finally:
            self.__job__.persistedModel = origPersistedModel

    # -------------------------------------------------------------------------
    def __process_single__(self, persistedModelRoot, random_state, X, y_encoded, channels, test_size):
        self.__job__.persistedModel = f"{persistedModelRoot}_random_{random_state:06d}{JOB_EXT}"
        self.__job__.persistedModelResults = self.__job__.newPersistedModelResultsName()

        singleTrainingProc: AbstractModelTrainingProcessor = self.trainingProcType(self.__job__, self.modelDefType)
        model, X_train, X_test, y_train, y_test, cross_validation_scores = singleTrainingProc.process(X, y_encoded, channels, test_size, random_state)

        singleEvalProc: BasicModelEvaluationProcessor = self.evalProcType(self.__job__, model)
        singleEvalProc.process(X_test, y_test, cross_validation_scores)

        run_report = singleEvalProc.reportSnapshot(singleTrainingProc)
        self.reportSnapshotList.append(run_report)
        singleEvalProc.writeReportToFile(self.__job__.persistedModelResults, run_report)

    # -------------------------------------------------------------------------
    def reportSnapshot(self):
        timestamp_utc = datetime.now(pytz.utc)
        elapsed_time = timestamp_utc - self.jobStartTime

        report = f"---- Bulk (start) ----\n"
        report = report + f"start time: {self.jobStartTime.isoformat()}\n"
        report = report + f"end time: {timestamp_utc.isoformat()}\n"
        report = report + f"elapsed: {elapsed_time}\n\n"
        report = report + self.reportSnapshotList
        report = report + "\n\n"
        report = f"---- Bulk (end) ----\n"
