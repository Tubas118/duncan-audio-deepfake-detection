import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model

from configuration.configuration import Job
from model_definitions.model_abstract_definition import ModelAbstractDefinition

class BasicModelTrainingProcessor:

    # -------------------------------------------------------------------------
    def __init__(self, job: Job, modelDefType):
        self.job = job
        self.modelDefType = modelDefType

    # -------------------------------------------------------------------------
    def process(self, X, y_encoded, channels, test_size = 0.2):
        if (self.job.newModelGenerated == False):
            raise ValueError("The job is configured to re-use an existing model, not generate a new model.")
        
        print("Selecting training and test data")
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size)
        
        print(f"Training using {len(X_train)} files.")
        model = self.__train_model__(X_train, X_test, y_train, y_test, channels)

        return model, X_train, X_test, y_train, y_test
    
    # -------------------------------------------------------------------------
    def __train_model__(self, X_train, X_test, y_train, y_test, channels) -> Model:
        modelDef: ModelAbstractDefinition = self.modelDefType(self.job, X_train.shape[2], channels)
        model = modelDef.buildModel()
        model.compile(optimizer=self.job.optimizer, loss=self.job.loss, metrics=self.job.metrics)

        print("Training the Model...")
        model.fit(X_train, y_train, batch_size=self.job.batchSize, epochs=self.job.numEpochs, validation_data=(X_test, y_test))

        print(f"Saving model: {self.job.persistedModel}")
        joblib.dump(model, self.job.persistedModel)

        return model