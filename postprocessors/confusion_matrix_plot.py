import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class ConfusionMatrixDetails:

    def __init__(self, cm: confusion_matrix, display: ConfusionMatrixDisplay, y_true, y_pred):
        self.cm = cm
        self.display = display
        self.y_true = y_true
        self.y_pred = y_pred


# References:
#   1. Chan, J. (2021). Machine learning with Python for beginners: A step-by-step guide to hands-on projects [Kindle edition].
#   2. Derived from https://github.com/sksmta/audio-deepfake-detection/blob/main/main.ipynb
class ConfusionMatrixPlot:

    def __init__(self, debugOn = False):
        self.debugOn = debugOn


    # *** Commenting out for the moment - this is probably not the right place for this
    # def preprocessAndPlot(self, classes, y_true_dict, y_pred, title = 'Confusion Matrix') -> ConfusionMatrixDetails:
    #     y_true = np.array([1 if y_true_dict[key] == classes[1] else 0 for key in y_true_dict.keys()])

    #     return self.plot(classes, y_true, y_pred, title)


    def plot(self, classes, trueAry: np.array, predAry: np.array, title = 'Confusion Matrix') -> ConfusionMatrixDetails:
        if (self.debugOn):
            print(f'trueAry: {trueAry}')
            print(f'predAry: {predAry}')
            print(f'classes: {classes}')

        details = ConfusionMatrixDetails(
            cm = [[0, 0], [0, 0]],
            display = None,
            y_true = [1, 1, 1, 1],
            y_pred = [0, 1, 0, 1]
        )
        return details

