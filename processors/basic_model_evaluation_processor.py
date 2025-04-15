import joblib
import json
import numpy as np
import pytz
from datetime import datetime
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Model

from config.configuration import Job
from model_definitions.model_abstract_definition import ModelAbstractDefinition
from postprocessors.metrics import Metrics
from processors.abstract_model_processor import AbstractModelProcessor
from processors.model_evaluation_result import ModelEvaluationResult
from utils.safe_len import safe_len

class BasicModelEvaluationProcessor(AbstractModelProcessor):

    # -------------------------------------------------------------------------
    def __init__(self, job: Job, model: Model = None):
        super().__init__(job)

        if (model == None):
            self.model = joblib.load(self.__job__.persistedModel)
        else:
            self.model = model

    # -------------------------------------------------------------------------
    def resetStatistics(self):
        self.jobStartTime = None
        self.inputFileBatchCount = 0
        self.inputFileCount = 0
        # self.score = 0
        self.batchResults: list[ModelEvaluationResult] = []

    # -------------------------------------------------------------------------
    def process(self, X_test, y_test) -> ModelEvaluationResult:
        if (self.jobStartTime == None):
            self.jobStartTime = datetime.now(pytz.utc)

        y_pred = self.model.predict(X_test)
        y_pred_work = np.argmax(y_pred, axis=1)
        y_test_work = np.argmax(y_test, axis=1)

        results = ModelEvaluationResult(testAry=y_test_work, predAry=y_pred_work)

        metrics = Metrics()
        metrics.evaluateResults(results)

        self.inputFileBatchCount = self.inputFileBatchCount + results.batchSize
        self.inputFileCount = self.inputFileCount + safe_len(X_test)


        print(f"  Batches: {self.inputFileBatchCount} - Files: {self.inputFileCount} - Accuracy Score: {results.accuracy_score} - Elements: {len(X_test)}")
        self.lastResults = results
        self.batchResults.append(results)

        return results

    # -------------------------------------------------------------------------
    def reportSnapshot(self, initialProcessor: AbstractModelProcessor = None):
        report = ""

        if (initialProcessor != None):
            report = initialProcessor.reportSnapshot()
            report = report + "\n"

        timestamp_utc = datetime.now(pytz.utc)
        elapsed_time = timestamp_utc - self.jobStartTime

        report = report + f"---- Testing (start) ----\n"
        report = report + f"start time: {self.jobStartTime.isoformat()}\n"
        report = report + f"end time: {timestamp_utc.isoformat()}\n"
        report = report + f"elapsed: {elapsed_time}\n\n"
        report = report + f"model file: {self.__job__.persistedModel}\n"
        report = report + f"batch count: {self.inputFileBatchCount}\n"
        report = report + f"file count: {self.inputFileCount}\n"

        if (len(self.batchResults) > 0):
            report = report + "\n"
            for batchResult in self.batchResults:
                report = report + f"{batchResult.reportSnaphot()}\n"

        report = report + "\n"
        report = report + ModelAbstractDefinition.MODEL_SUMMARY(self.model) + "\n"

        if ("job: {" not in report):
            prettyJson = json.dumps(self.__job__.__dict__, indent=4)
            report = report + f"job: {prettyJson}\n\n"

        report = report + f"---- Testing (end) ----\n"

        return report