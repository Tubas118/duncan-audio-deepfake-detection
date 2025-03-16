from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

from configuration.configuration import Job
from model_definitions.model_abstract_definition import ModelAbstractDefinition

class ModelCnnDefinition(ModelAbstractDefinition):

    def __init__(self, job: Job, width, channels):
        super().__init__(job)
        self.width = width
        self.channels = channels

    def buildModel(self):
        print(f"__job__: {self.__job__}")
        input_shape = (self.__job__.numMels, self.width, self.channels)
        model_input = Input(shape = input_shape)

        # TODO - why were these parameters selected? What purpose do they serve? Should they be configurable?
        x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(model_input)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(units=128, activation='relu')(x)
        x = Dropout(0.5)(x)

        model_output = Dense(self.__job__.numClasses, activation='softmax')(x)

        return Model(inputs=model_input, outputs=model_output)