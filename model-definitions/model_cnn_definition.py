from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

from model_abstract_definition import ModelAbstractDefinition

class ModelCnnDefinition(ModelAbstractDefinition):

    def __init__(self, job, width, channels):
        super().__init__(job)

    def buildModel(self):
        pass