from io import StringIO
import joblib
import json
import numpy as np
import pytz
from datetime import datetime
from sklearn.metrics import accuracy_score, root_mean_squared_error
from tensorflow.keras.models import Model

from config.configuration import Job
from postprocessors.metrics import Metrics
from postprocessors.plot_confusion_matrix import ConfusionMatrixPlot
from processors.abstract_model_processor import AbstractModelProcessor
from processors.model_evaluation_result import ModelEvaluationResult

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

    # -------------------------------------------------------------------------
    def process(self, X_test, y_test, true_labels) -> ModelEvaluationResult:
        if (self.jobStartTime == None):
            self.jobStartTime = datetime.now(pytz.utc)

        y_pred = self.model.predict(X_test)
        y_pred_work = np.argmax(y_pred, axis=1)
        y_test_work = y_test    # np.argmax(y_test, axis=1)

        if (len(true_labels) > 0):
            y_test_work = np.array([label for label in true_labels.values()])

        self.inputFileBatchCount = self.inputFileBatchCount + 1
        self.inputFileCount = self.inputFileCount + len(X_test)

        results = ModelEvaluationResult(testAry=y_test_work, predAry=y_pred_work)

        metrics = Metrics()
        metrics.evaluateResults(results)
        print(f'Metric results: {results.reportSnaphot()}')

        print(f"  Batches: {self.inputFileBatchCount} - Files: {self.inputFileCount} - Accuracy Score: {results.accuracy_score} - Elements: {len(X_test)}")
        self.lastResults = results

        return results

    # -------------------------------------------------------------------------
    def reportSnapshot(self, initialProcessor: AbstractModelProcessor = None) -> str:
        report = ""

        if (initialProcessor != None):
            report = initialProcessor.reportSnapshot()
            report = report + "\n"

        timestamp_utc = datetime.now(pytz.utc)
        elapsed_time = timestamp_utc - self.jobStartTime
        prettyJson = json.dumps(self.__job__.__dict__, indent=4)

        report = report + f"{self.__model_summary_to_string__()}"

        report = report + f"---- Testing (start) ----\n"
        report = report + f"start time: {self.jobStartTime.isoformat()}\n"
        report = report + f"end time: {timestamp_utc.isoformat()}\n"
        report = report + f"elapsed: {elapsed_time}\n\n"
        report = report + f"model file: {self.__job__.persistedModel}\n"
        report = report + f"batch count: {self.inputFileBatchCount}\n"
        report = report + f"file count: {self.inputFileCount}\n"

        if (self.lastResults != None):
            report = report + f"\n{self.lastResults.reportSnaphot()}\n"

        report = report + "\n"
        report = report + f"job: {prettyJson}\n\n"
        report = report + f"---- Testing (end) ----\n"

        return report

    # -------------------------------------------------------------------------
    def __model_summary_to_string__(self) -> str:
        string_io = StringIO()
        self.model.summary(print_fn=lambda x: string_io.write(x + '\n'))
        return string_io.getvalue()