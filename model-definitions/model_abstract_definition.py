from abc import ABC, abstractmethod
import copy

class ModelAbstractDefinition(ABC):

    def __init__(self, job):
        self.__job__ = job

    @abstractmethod
    def buildModel(self):
        pass