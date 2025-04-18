from abc import abstractmethod

from config.configuration import Job
from processors.abstract_model_processor import AbstractModelProcessor

# =============================================================================
class AbstractModelTrainingProcessor(AbstractModelProcessor):

    def __init__(self, job: Job):
        super().__init__(job)
        if (self.__job__.newModelGenerated == False):
            raise ValueError("The job is configured to re-use an existing model, not generate a new model.")
        
    # -------------------------------------------------------------------------
    @abstractmethod
    def process(self, X, y_encoded, channels, test_size = 0.2, trainingSplitRandomState: int = None, scoring = None):
        pass

    # -------------------------------------------------------------------------
    def __get_training_split_random_state__(self, trainingSplitRandomState):
        useTrainingSplitRandomState: int = trainingSplitRandomState

        if (useTrainingSplitRandomState == None):
            useTrainingSplitRandomState = self.__job__.trainingSplitRandomState

        return useTrainingSplitRandomState