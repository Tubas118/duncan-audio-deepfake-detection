# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import joblib
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

import configuration
from notebook_utils import notebookToPython
from label_reader import readLabels

# +
config = configuration.ConfigLoader('config.yml')

notebookToPython(config.projectName)
job = config.getJobConfig(config.activeJobId)
# -

trainingLabels = readLabels(job)

# +
X = []
y = []


fullDataPath = job.fullJoinFilePath(job.dataPath, job.trainingDataPath)

for filename, label in trainingLabels.items():
    audioSourceFilename = job.fullJoinFilePath(fullDataPath, filename + job.trainingDataExtension)
    
    audio, _ = librosa.load(audioSourceFilename, sr = job.sampleRate, duration = job.duration)

    mel_spectrogram = librosa.feature.melspectrogram(y = audio, sr = job.sampleRate, n_mels = job.numMels)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    if (mel_spectrogram.shape[1] < job.maxTimeSteps):
        padWidth = ((0, 0), (0, job.maxTimeSteps - mel_spectrogram.shape[1]))
        mel_spectrogram = np.pad(array=mel_spectrogram, pad_width=padWidth, mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :job.maxTimeSteps]

    X.append(mel_spectrogram)
    y.append(label)
# -

X = np.array(X)
y = np.array(y)
y_encoded = to_categorical(y, job.numClasses)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)    # test data is 20% of all data

# Define CNN model architecture
input_shape = (job.numMels, X_train.shape[2], 1)  # Input shape for CNN (height, width, channels)
model_input = Input(shape=input_shape)


# +
# TODO - why were these parameters selected? What purpose do they serve? Should they be configurable?
x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(model_input)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(units=128, activation='relu')(x)
x = Dropout(0.5)(x)

model_output = Dense(job.numClasses, activation='softmax')(x)
# -

model = Model(inputs=model_input, outputs=model_output)


model.compile(optimizer=job.optimizer, loss=job.loss, metrics=job.metrics)

# Train the Model
model.fit(X_train, y_train, batch_size=job.batchSize, epochs=job.numEpochs, validation_data=(X_test, y_test))

joblib.dump(model, job.persistedModel)

# ### Test Model

# +
import joblib
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

import configuration
from notebook_utils import notebookToPython
from label_reader import readLabels

# +
config = configuration.ConfigLoader('config.yml')

notebookToPython(config.projectName)
job = config.getJobConfig(config.activeJobId)
# -

model = joblib.load(job.persistedModel)

predictions = model.predict(X_train)
predictions

score = accuracy_score(y_train, predictions)
score
