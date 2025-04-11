import pickle

import numpy as np

class PreprocessPersistance:

    # -------------------------------------------------------------------------
    def __init__(self, X_test = None, y_test = None, true_labels = None, source_filenames = None):
        self.X_test = X_test
        self.y_test = y_test
        self.true_labels = true_labels
        self.source_filenames = source_filenames

    # -------------------------------------------------------------------------
    @staticmethod
    def load(filename: str) -> "PreprocessPersistance":
        return PreprocessPersistance()

    # -------------------------------------------------------------------------
    def save(self, filename: str):
        pass

    # -------------------------------------------------------------------------
    def compare(self, other: "PreprocessPersistance") -> bool:
        return (np.array_equal(self.X_test, other.X_test)
                and np.array_equal(self.y_test, other.y_test)
                and np.array_equal(self.true_labels, other.true_labels)
                and np.array_equal(self.source_filenames, other.source_filenames))