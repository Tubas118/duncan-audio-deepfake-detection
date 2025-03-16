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
import joblib
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

import configuration.configuration as configuration
from mel_spectrogram.mel_spectrogram import MelSpectrogramGenerator
from notebook_utils import notebookToPython
from readers.label_reader import readTrainingLabelsWithJob

# +
config = configuration.ConfigLoader('config.yml')

notebookToPython(config.projectName)
job = config.getJobConfig(config.activeJobId)
# -

generator = MelSpectrogramGenerator()
X, y_encoded = generator.generateMelSpectrograms(job, job.trainingDataPathSuffix)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)    # test data is 20% of all data

# +
import model_definitions.model_cnn_definition as model_cnn_definition

modelDef = model_cnn_definition.ModelCnnDefinition(job, X_train.shape[2], 1)
model = modelDef.buildModel()
# -

model.compile(optimizer=job.optimizer, loss=job.loss, metrics=job.metrics)

# Train the Model
model.fit(X_train, y_train, batch_size=job.batchSize, epochs=job.numEpochs, validation_data=(X_test, y_test))

joblib.dump(model, job.persistedModel)

# ### Test Model

# +
import joblib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

import configuration.configuration as configuration
from notebook_utils import notebookToPython
from readers.label_reader import readLabelsWithJob

# +
oldJob = job
config = configuration.ConfigLoader('config.yml')

notebookToPython(config.projectName)
job = config.getJobConfig(config.activeJobId)

if (len(oldJob.persistedModel) > 0 and oldJob.persistedModel != job.persistedModel):
    job.persistedModel = oldJob.persistedModel

# -

model = joblib.load(job.persistedModel)

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
y_pred

y_test

# +
import json
from datetime import datetime
import pytz

score = accuracy_score(y_test, y_pred)

timestamp_utc = datetime.now(pytz.utc)

with open(job.persistedModelResults, "w") as file:
    file.write(f"job completed: {timestamp_utc.isoformat()}\n")
    file.write(f"model file: {job.persistedModel}\n")
    file.write(f"accuracy_score: {score}\n\n")
    prettyJson = json.dumps(job.__dict__, indent=4)
    file.write(f"job: {prettyJson}\n")
