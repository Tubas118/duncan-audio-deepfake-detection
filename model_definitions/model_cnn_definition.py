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

    def buildModel(self) -> Model:
        print(f"__job__: {self.__job__}")
        input_shape = (self.__job__.numMels, self.width, self.channels)
        model_input = Input(shape = input_shape)

        # Derived from:
        # Anagha, R., Arya, A., Narayan, V. H., Abhishek, S., & Anjali, T. (2023).
        #   Audio Deepfake Detection Using Deep Learning.
        #   2023 12th International Conference on System Modeling & Advancement in Research Trends (SMART), System Modeling & Advancement in Research Trends (SMART), 2023 12th International Conference On, 176–181.
        #   https://doi.org/10.1109/SMART59791.2023.10428163
        x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(model_input)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(units=128, activation='relu')(x)
        x = Dropout(0.5)(x)

        model_output = Dense(self.__job__.numClasses, activation='softmax')(x)

        return Model(inputs=model_input, outputs=model_output)