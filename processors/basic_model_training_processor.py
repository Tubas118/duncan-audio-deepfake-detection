import joblib
import json
import pytz
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model

from configuration.configuration import Job
from model_definitions.model_abstract_definition import ModelAbstractDefinition
from processors.abstract_model_processor import AbstractModelProcessor

class BasicModelTrainingProcessor(AbstractModelProcessor):

    # -------------------------------------------------------------------------
    def __init__(self, job: Job, modelDefType):
        self.job = job
        self.resetStatistics()
        self.modelDefType = modelDefType

    # -------------------------------------------------------------------------
    def resetStatistics(self):
        self.jobStartTime = None
        self.inputFileBatchCount = 0
        self.inputFileCount = 0

    # -------------------------------------------------------------------------
    def process(self, X, y_encoded, channels, test_size = 0.2):
        if (self.job.newModelGenerated == False):
            raise ValueError("The job is configured to re-use an existing model, not generate a new model.")
        
        if (self.jobStartTime == None):
            self.jobStartTime = datetime.now(pytz.utc)

        print("Selecting training and test data")
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=self.job.trainingSplitRandomState)
        
        print(f"Training using {len(X_train)} files.")
        model = self.__train_model__(X_train, X_test, y_train, y_test, channels)

        return model, X_train, X_test, y_train, y_test
    
    # -------------------------------------------------------------------------
    def reportSnapshot(self):
        timestamp_utc = datetime.now(pytz.utc)
        elapsed_time = timestamp_utc - self.jobStartTime
        prettyJson = json.dumps(self.job.__dict__, indent=4)

        report = f"---- Training (start) ----\n"
        report = report + f"start time: {self.jobStartTime.isoformat()}\n"
        report = report + f"end time: {timestamp_utc.isoformat()}\n"
        report = report + f"elapsed: {elapsed_time}\n\n"
        report = report + f"model file: {self.job.persistedModel}\n"
        report = report + f"batch count: {self.inputFileBatchCount}\n"
        report = report + f"file count: {self.inputFileCount}\n"
        report = report + f"job: {prettyJson}\n\n"
        report = report + f"---- Training (end) ----\n"

        return report

    # -------------------------------------------------------------------------
    def __train_model__(self, X_train, X_test, y_train, y_test, channels) -> Model:
        modelDef: ModelAbstractDefinition = self.modelDefType(self.job, X_train.shape[2], channels)
        print(f"Model definition:\n{modelDef.printModelDefintions()}")

        model = modelDef.buildModel()
        model.compile(optimizer=self.job.optimizer, loss=self.job.loss, metrics=self.job.metrics)

        print("Training the Model...")
        model.fit(X_train, y_train, batch_size=self.job.batchSize, epochs=self.job.numEpochs, validation_data=(X_test, y_test))

        print(f"Saving model: {self.job.persistedModel}")
        joblib.dump(model, self.job.persistedModel)

        return model