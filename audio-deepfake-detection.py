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
import json
import pytz
import joblib
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

import configuration.configuration as configuration
import model_definitions.model_cnn_definition as model_cnn_definition
from mel_spectrogram.mel_spectrogram import MelSpectrogramGenerator
from notebook_utils import notebookToPython

# +
config = configuration.ConfigLoader('config.yml')

notebookToPython(config.projectName)
job = config.getJobConfig(config.activeJobId)
# -

generator = MelSpectrogramGenerator()
X, y_encoded = generator.generateMelSpectrograms(job, job.dataPathSuffix)

if (job.newModelGenerated):
    print("Selecting training and test data")
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)    # test data is 20% of all data
else:
    print("Assigning all data for evaluation")
    y_train = None
    X_train = None
    X_test = X
    y_test = y_encoded
    job.persistedModelResults = job.newPersistedModelResultsName(job.persistedModel, True)

if (job.newModelGenerated):
    modelDef = model_cnn_definition.ModelCnnDefinition(job, X_train.shape[2], 1)
    model = modelDef.buildModel()

if (job.newModelGenerated):
    model.compile(optimizer=job.optimizer, loss=job.loss, metrics=job.metrics)

if (job.newModelGenerated):
    print("Training the Model...")
    model.fit(X_train, y_train, batch_size=job.batchSize, epochs=job.numEpochs, validation_data=(X_test, y_test))

if (job.newModelGenerated):
    print(f"Saving model: {job.persistedModel}")
    joblib.dump(model, job.persistedModel)
else:
    print(f"Loading model: {job.persistedModel}")
    model = joblib.load(job.persistedModel)

# ### Test Model

y_pred = model.predict(X_test)
y_pred_work = np.argmax(y_pred, axis=1)
y_test_work = np.argmax(y_test, axis=1)
y_pred_work

y_test_work

# +
from sklearn.base import accuracy_score

score = accuracy_score(y_test_work, y_pred_work)

timestamp_utc = datetime.now(pytz.utc)

# +
prettyJson = json.dumps(job.__dict__, indent=4)

report = f"job completed: {timestamp_utc.isoformat()}\n"
report = report + f"model file: {job.persistedModel}\n"
report = report + f"accuracy_score: {score}\n\n"
report = report + f"job: {prettyJson}\n"

print(report)

with open(job.persistedModelResults, "w") as file:
    file.write(report)
