import copy
import inspect
from abc import ABC, abstractmethod
from tensorflow.keras.models import Model

from config.configuration import Job

class ModelAbstractDefinition(ABC):

    def __init__(self, job):
        self.__job__: Job = copy.deepcopy(job)

    @abstractmethod
    def buildModel(self) -> Model:
        pass

    def printModelDefintions(self):
        lines = inspect.getsource(self.buildModel)
        print(lines)