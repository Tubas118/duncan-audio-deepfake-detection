import joblib
import json
import pytz
from datetime import datetime
from sklearn.model_selection import cross_val_score, train_test_split
from tensorflow.keras.models import Model

from config.configuration import Job
from model_definitions.model_abstract_definition import ModelAbstractDefinition
from processors.abstract_model_training_processor import AbstractModelTrainingProcessor

# =============================================================================
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
    def process(self, X, y_encoded, channels, test_size = 0.2, trainingSplitRandomState: int = None, 
                        scoring = ['test_score', 'fit_time', 'score_time']):
        
        if (self.jobStartTime == None):
            self.jobStartTime = datetime.now(pytz.utc)

        useTrainingSplitRandomState: int = self.__get_training_split_random_state__(trainingSplitRandomState)

        print(f"Selecting training and test data - traininSplitRandomState: {useTrainingSplitRandomState}")
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=useTrainingSplitRandomState)
        
        print(f"Training using {len(X_train)} files.")
        model = self.__train_model__(X_train, X_test, y_train, y_test, channels, scoring)

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
    def __train_model__(self, X_train, X_test, y_train, y_test, channels, scoring) -> Model:
        
        modelDef: ModelAbstractDefinition = self.modelDefType(self.__job__, X_train.shape[2], channels)

        model = modelDef.buildModel()
        model.compile(optimizer=self.__job__.optimizer, loss=self.__job__.loss, metrics=self.__job__.metrics)

        print("Training the Model...")
        model.fit(X_train, y_train, batch_size=self.__job__.batchSize, epochs=self.__job__.numEpochs, validation_data=(X_test, y_test))

        print(f"Saving model: {self.__job__.persistedModel}")
        joblib.dump(model, self.__job__.persistedModel)

        scores = cross_val_score(model, X_train, y_train, cv=self.__job__.cv, scoring=scoring)
        print(f"cross validation scores: {scores}")

        return model