import joblib
import json
import pytz
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model

from config.configuration import Job
from model_definitions.model_abstract_definition import ModelAbstractDefinition
from processors.abstract_model_training_processor import AbstractModelTrainingProcessor

class BasicModelTrainingProcessor(AbstractModelTrainingProcessor):

    # -------------------------------------------------------------------------
    def __init__(self, job: Job, modelDefType):
        super().__init__(job)
        self.resetStatistics()
        self.modelDefType = modelDefType

    # -------------------------------------------------------------------------
    def resetStatistics(self):
        self.jobStartTime = None
        self.inputFileBatchCount = 0
        self.inputFileCount = 0

    # -------------------------------------------------------------------------
    def process(self, X, y_encoded, channels, test_size = 0.2, trainingSplitRandomState: int = None):
        if (self.jobStartTime == None):
            self.jobStartTime = datetime.now(pytz.utc)

        useTrainingSplitRandomState: int = self.__get_training_split_random_state__(trainingSplitRandomState)
        print(f'useTrainingSplitRandomState: {useTrainingSplitRandomState}')

        if (useTrainingSplitRandomState != None and useTrainingSplitRandomState < 0):
            X_train, X_test, y_train, y_test = self.__split_sksmta_method__(X, y_encoded)
        else:
            print(f"Selecting training and test data - traininSplitRandomState: {useTrainingSplitRandomState}")
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=useTrainingSplitRandomState)
        
        print(f"Training using {len(X_train)} files.")
        model = self.__train_model__(X_train, X_test, y_train, y_test, channels)

        return model, X_train, X_test, y_train, y_test
    
    # -------------------------------------------------------------------------
    def reportSnapshot(self):
        timestamp_utc = datetime.now(pytz.utc)
        elapsed_time = timestamp_utc - self.jobStartTime
        prettyJson = json.dumps(self.__job__.__dict__, indent=4)

        report = f"---- Training (start) ----\n"
        report = report + f"start time: {self.jobStartTime.isoformat()}\n"
        report = report + f"end time: {timestamp_utc.isoformat()}\n"
        report = report + f"elapsed: {elapsed_time}\n\n"
        report = report + f"model file: {self.__job__.persistedModel}\n"
        report = report + f"batch count: {self.inputFileBatchCount}\n"
        report = report + f"file count: {self.inputFileCount}\n"
        report = report + f"job: {prettyJson}\n\n"
        report = report + f"---- Training (end) ----\n"

        return report

    # -------------------------------------------------------------------------
    def __split_sksmta_method__(self, X, y_encoded):
        print('IMPORTANT: Splitting using the same method as in "Sksmta" base code')
        split_index = int(0.8 * len(X))
        X_train, X_test = X[:split_index], X[split_index:]                  # X_test: var names 'X_val' in Sksmta
        y_train, y_test = y_encoded[:split_index], y_encoded[split_index:]  # y_test: var names 'y_val' in Sksmta
        return X_train, X_test, y_train, y_test

    # -------------------------------------------------------------------------
    def __train_model__(self, X_train, X_test, y_train, y_test, channels) -> Model:
        modelDef: ModelAbstractDefinition = self.modelDefType(self.__job__, X_train.shape[2], channels)
        print(f"Model definition:\n{modelDef.printModelDefintions()}")

        model = modelDef.buildModel()
        model.compile(optimizer=self.__job__.optimizer, loss=self.__job__.loss, metrics=self.__job__.metrics)

        print("Training the Model...")
        model.fit(X_train, y_train, batch_size=self.__job__.batchSize, epochs=self.__job__.numEpochs, validation_data=(X_test, y_test))

        print(f"Saving model: {self.__job__.persistedModel}")
        joblib.dump(model, self.__job__.persistedModel)

        return model