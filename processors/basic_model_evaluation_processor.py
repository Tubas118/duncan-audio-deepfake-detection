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
        self.processedCount = 0
        self.successCount = 0

    # -------------------------------------------------------------------------
    def process(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_pred_work = np.argmax(y_pred, axis=1)
        y_test_work = np.argmax(y_test, axis=1)

        score = accuracy_score(y_test_work, y_pred_work)
        self.successCount = self.successCount + score

        # self.reportSnapshot(len(X_test), score, timestamp_utc)
        self.processedCount = self.processedCount + len(X_test)
        print(f"  Processed: {len(X_test)} - Score: {score}")

    # -------------------------------------------------------------------------
    def reportSnapshot(self):
        timestamp_utc = datetime.now(pytz.utc)
        prettyJson = json.dumps(self.job.__dict__, indent=4)

        report = f"timestamp: {timestamp_utc.isoformat()}\n"
        report = report + f"model file: {self.job.persistedModel}\n"
        report = report + f"test file count: {self.processedCount}\n"
        report = report + f"accuracy_score: {(float) (self.successCount) / self.processedCount}\n\n"
        report = report + f"job: {prettyJson}\n"

        print(report)

        with open(self.job.persistedModelResults, "w") as file:
            file.write(report)