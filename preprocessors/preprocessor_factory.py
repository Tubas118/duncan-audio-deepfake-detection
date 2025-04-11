from types import MappingProxyType

from config.configuration import Job
from preprocessors.abstract_preprocessor import AbstractPreprocessor
from preprocessors.mel_freq_cepstral_coefficient import MelFrequencyCepstralCoeffiecient
from preprocessors.mel_spectrogram import MelSpectrogramPreprocessor

class PreprocessorFactory:

    def __init__(self):
        availablePreprocessors = {
            'mel_spectrogram': __typeEntry__(MelSpectrogramPreprocessor),
            'mfcc': __typeEntry__(MelFrequencyCepstralCoeffiecient)
        }
        self.availablePreprocessors = MappingProxyType(availablePreprocessors)
        

    def newPreprocessor(self, preprocessorId: str, exec_power_to_db=True) -> AbstractPreprocessor:
        entry = self.availablePreprocessors.get(preprocessorId, None)
        if (entry != None):
            preprocessorType = entry.get('type')
            return preprocessorType(exec_power_to_db=exec_power_to_db)
        
        return None


def __typeEntry__(type):
    instance = type(True)
    typeName = instance.__class__.__name__
    return {
        'type': type,
        'typeName': typeName
    }