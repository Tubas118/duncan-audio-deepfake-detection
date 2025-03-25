from abc import ABC, abstractmethod
import copy

from configuration.configuration import Job

# ======================================================================
class AbstractModelProcessor(ABC):

    def __init__(self, job: Job):
        self.__job__: Job = copy.deepcopy(job)
        self.resetStatistics()

    @abstractmethod
    def resetStatistics(self):
        pass

    @abstractmethod
    def reportSnapshot(self, initialProcessor = None):
        pass

    def writeReportToFile(self, outputFilename, content):
        with open(outputFilename, "w") as file:
            file.write(content)
