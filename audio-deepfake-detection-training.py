# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: audio-deepfake-detection
#     language: python
#     name: python3
# ---

# +
from config.configuration import RunDetails

runDetail = RunDetails('config.yml', 'GitLab-training-data')
# runDetail = RunDetails('config.yml', 'ASVspoof-2019_training')
# runDetail = RunDetails('config.yml', 'ASVspoof-2019_training_epoch-100')

notebookName = 'audio-deepfake-detection-training'
plot_title_suffix = "(Training)"
# -

configFilename = runDetail.configFilename
runJobId = runDetail.jobId

import config.configuration as configuration
import model_definitions.model_cnn_definition as model_cnn_definition
from postprocessors.plot_confusion_matrix import PlotConfusionMatrix
from postprocessors.plot_roc_curve import PlotRocCurve
from preprocessors.abstract_preprocessor import AbstractPreprocessor
from preprocessors.preprocessor_factory import PreprocessorFactory
from notebook_utils import notebookToPython
from processors.basic_model_training_processor import BasicModelTrainingProcessor
from processors.basic_model_evaluation_processor import BasicModelEvaluationProcessor

# +
config = configuration.ConfigLoader(configFilename)

notebookToPython(notebookName)
job = config.getJobConfig(runJobId)

if (job.newModelGenerated == False):
    raise ValueError("This notebook is meant for training. Select a job without a value for 'persisted-model' set.")
# -

preproc_factory = PreprocessorFactory()
preprocessor: AbstractPreprocessor = preproc_factory.newPreprocessor(job.preprocessor)

X, y_encoded = preprocessor.extract_features_multipleSource(job, job.dataPathSuffix)

# +
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
class BasicModelTrainingProcessor2(AbstractModelTrainingProcessor):

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
                        scoring = 'accuracy'):
        
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

        # scores = cross_val_score(model, X_train, y_train, cv=self.__job__.cv, scoring=scoring)
        # print(f"cross validation scores: {scores}")

        return model


# -

trainingProc = BasicModelTrainingProcessor2(job, model_cnn_definition.ModelCnnDefinition)
model, X_train, X_test, y_train, y_test = trainingProc.process(X, y_encoded, 1)

# +
from scikeras.wrappers import KerasClassifier

kerasModel = KerasClassifier(build_fn=model, batch_size=job.batchSize, epochs=job.numEpochs,
                             optimizer=job.optimizer, loss=job.loss, metrics=job.metrics)

scores = cross_val_score(model, X_train, y_train, cv=job.cv)
print(f"cross validation scores: {scores}")

# -

# ### Test Model

evaluationProc = BasicModelEvaluationProcessor(job, model)
results = evaluationProc.process(X_test, y_test)
print(f"{results.reportSnaphot()}")


CM_TITLE = f"{PlotConfusionMatrix.DEFAULT_TITLE} {plot_title_suffix}"
cm_plot = PlotConfusionMatrix()
cm_plot.plotFromResults(results, job, CM_TITLE)


RC_TITLE = f"{PlotRocCurve.DEFAULT_TITLE} {plot_title_suffix}"
roc_plot = PlotRocCurve()
roc_plot.plotFromResults(results, RC_TITLE)

# +
print("\n")
report = evaluationProc.reportSnapshot(trainingProc)
evaluationProc.writeReportToFile(job.persistedModelResults, report)

print(report)
