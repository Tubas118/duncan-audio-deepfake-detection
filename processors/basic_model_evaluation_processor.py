import joblib
import json
import numpy as np
import pytz
from datetime import datetime
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Model

from configuration.configuration import Job

class BasicModelEvaluationProcessor:

    # -------------------------------------------------------------------------
    def __init__(self, job: Job, model: Model = None):
        self.job = job
        self.resetStatistics()

        if (model == None):
            self.model = joblib.load(self.job.persistedModel)
        else:
            self.model = model

    def resetStatistics(self):
        self.jobStartTime = None
        self.inputFileBatchCount = 0
        self.inputFileCount = 0
        self.score = 0

    # -------------------------------------------------------------------------
    def process(self, X_test, y_test):
        if (self.jobStartTime == None):
            self.jobStartTime = datetime.now(pytz.utc)

        y_pred = self.model.predict(X_test)
        y_pred_work = np.argmax(y_pred, axis=1)
        y_test_work = np.argmax(y_test, axis=1)

        score = accuracy_score(y_test_work, y_pred_work)
        self.score = self.score + score

        self.inputFileBatchCount = self.inputFileBatchCount + 1
        self.inputFileCount = self.inputFileCount + len(X_test)
        print(f"  Batches: {self.inputFileBatchCount} - Files: {self.inputFileCount} - Score: {score} - Elements: {len(X_test)}")

    # -------------------------------------------------------------------------
    def reportSnapshot(self):
        timestamp_utc = datetime.now(pytz.utc)
        elapsed_time = timestamp_utc - self.jobStartTime
        prettyJson = json.dumps(self.job.__dict__, indent=4)

        report = f"start time: {self.jobStartTime.isoformat()}\n"
        report = report + f"end time: {timestamp_utc.isoformat()}\n"
        report = report + f"elapsed: {elapsed_time}\n\n"
        report = report + f"model file: {self.job.persistedModel}\n"
        report = report + f"batch count: {self.inputFileBatchCount}\n"
        report = report + f"file count: {self.inputFileCount}\n"
        report = report + f"accuracy_score: {(float) (self.score) / self.inputFileBatchCount}\n\n"
        report = report + f"job: {prettyJson}\n"

        print(report)

        with open(self.job.persistedModelResults, "w") as file:
            file.write(report)