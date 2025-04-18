import librosa
import numpy as np
from matplotlib import pyplot as plt

from config.configuration import Job

class PlotSpectrogram:

    # -------------------------------------------------------------------------
    def __init__(self):
        pass

    def plot(self, spectrogramAry: np.array, job: Job, title: str, hopLength = 512):
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(spectrogramAry, x_axis='time', y_axis='mel', sr=job.sampleRate, hop_length=hopLength)
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.show()
