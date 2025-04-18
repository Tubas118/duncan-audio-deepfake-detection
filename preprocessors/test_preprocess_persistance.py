import os
import unittest
import numpy as np
import path
import sys
from numpy.testing import assert_array_equal


# -- from parent directory
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
from preprocessors.preprocess_persistance import PreprocessPersistance


# =============================================================================
class TestPreprocessPersistance(unittest.TestCase):

    # -------------------------------------------------------------------------
    def setUp(self):
        pass

    # -------------------------------------------------------------------------
    def test_persistModelResults(self):
        # given
        initData = __DataForTestPreprocessPersistance__(y_true = np.array([0, 0, 1, 1, 0, 1]),
                                                   y_pred = np.array([0, 0, 0, 1, 0, 0]),
                                                   labels = np.array(['Cat', 'Cat', 'Dog', 'Dog', 'Cat', 'Dog']),
                                                   filenames=np.array(['a', 'b', 'c', 'd', 'e', 'f']))
        
        persist = PreprocessPersistance(initData.y_true, initData.y_pred, initData.labels, initData.filenames)

        # when / then
        self.__perform_persistanceTests__(persist, initData)


    # -------------------------------------------------------------------------
    def __perform_persistanceTests__(self, persist: PreprocessPersistance,
                                     initData: "__DataForTestPreprocessPersistance__"):
        # given (inner)
        persist_fn = 'temp.bin.tmp'

        try:
            # -- sanity check
            self.__remove_file__(persist_fn)
            self.assertFalse(os.path.exists(persist_fn))

            assert_array_equal(persist.X_test, initData.y_true)
            assert_array_equal(persist.y_test, initData.y_pred)
            assert_array_equal(persist.true_labels, initData.labels)
            assert_array_equal(persist.source_filenames, initData.filenames)

            # when #1 (inner)
            persist.save(persist_fn)

            # then #1 (inner)
            self.assertTrue(os.path.exists(persist_fn))

            # when #2 (inner)
            reloaded1 = persist.load(persist_fn)

            # then #2 (inner)
            self.assertTrue(persist.compare(reloaded1))
            self.assertTrue(reloaded1.compare(persist))

            # when #3 (inner)
            reloaded2 = persist.load(persist_fn)

            # then #3 (inner)
            self.assertTrue(persist.compare(reloaded2))
            self.assertTrue(reloaded2.compare(persist))

        finally:
            self.__remove_file__(persist_fn)


    # -------------------------------------------------------------------------
    def __remove_file__(self, filename):
        if (os.path.exists(filename)):
            os.remove(filename)


# =============================================================================
class __DataForTestPreprocessPersistance__:

    # -------------------------------------------------------------------------
    def __init__(self, y_true: np.array, y_pred: np.array, labels: np.array, filenames: np.array):
        self.y_true = y_true
        self.y_pred = y_pred
        self.labels = labels
        self.filenames = filenames


if __name__ == '__main__':
    unittest.main()