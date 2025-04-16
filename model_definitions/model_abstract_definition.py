import copy
import inspect
from abc import ABC, abstractmethod
from io import StringIO
from tensorflow.keras.models import Model

from config.configuration import Job

class ModelAbstractDefinition(ABC):

    # -------------------------------------------------------------------------
    @staticmethod
    def MODEL_SUMMARY(model) -> str:
        string_io = StringIO()
        model.summary(print_fn=lambda x: string_io.write(x + '\n'))
        return string_io.getvalue()
    
    # -------------------------------------------------------------------------
    def __init__(self, job):
        self.__job__: Job = copy.deepcopy(job)
        self.lastModel = None

    # -------------------------------------------------------------------------
    @abstractmethod
    def buildModel(self) -> Model:
        pass

    # -------------------------------------------------------------------------
    def getModelDefinitions(self) -> str:
        if (self.lastModel != None):
            return ModelAbstractDefinition.MODEL_SUMMARY(self.lastModel)
        
        return inspect.getsource(self.buildModel)
    
