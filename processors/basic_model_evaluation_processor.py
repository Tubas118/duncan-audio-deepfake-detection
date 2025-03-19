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

        if (model == None):
            self.model = joblib.load(self.job.persistedModel)
        else:
            self.model = model

    # -------------------------------------------------------------------------
    def process(self, identifiers, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_pred_work = np.argmax(y_pred, axis=1)
        y_test_work = np.argmax(y_test, axis=1)

        score = accuracy_score(y_test_work, y_pred_work)
        timestamp_utc = datetime.now(pytz.utc)

        self.__generate_results__(identifiers, score, timestamp_utc)

    # -------------------------------------------------------------------------
    def __generate_results__(self, identifiers, score, timestamp_utc):
        prettyJson = json.dumps(self.job.__dict__, indent=4)
        processedFiles = json.dumps(identifiers, indent=4)

        report = f"job completed: {timestamp_utc.isoformat()}\n"
        report = report + f"model file: {self.job.persistedModel}\n"
        report = report + f"processed files: {processedFiles}\n"
        report = report + f"accuracy_score: {score}\n\n"
        report = report + f"job: {prettyJson}\n"

        print(report)

        with open(self.job.persistedModelResults, "w") as file:
            file.write(report)