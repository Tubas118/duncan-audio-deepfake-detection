from abc import ABC, abstractmethod

class AbstractModelProcessor(ABC):

    @abstractmethod
    def reportSnapshot(self, initialProcessor = None):
        pass

    def writeReportToFile(self, outputFilename, content):
        with open(outputFilename, "w") as file:
            file.write(content)