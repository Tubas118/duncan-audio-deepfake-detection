import unittest
import path
import sys


# -- from parent directory
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)

from utils.safe_len import safe_len


class TestSafeLen(unittest.TestCase):

    def setUp(self):
        pass

    # -------------------------------------------------------------------------
    def test_string(self):
        # given
        source = "The rain in Spain falls mainly in the plains"

        # when
        source_len = safe_len(source)

        # then
        self.assertGreater(source_len, 0)

    # -------------------------------------------------------------------------
    def test_integer(self):
        # given
        source = 1492

        # when
        source_len = safe_len(source)

        # then
        self.assertEqual(source_len, 0)

    # -------------------------------------------------------------------------
    def test_none(self):
        # given
        source = None

        # when
        source_len = safe_len(source)

        # then
        self.assertEqual(source_len, 0)



if __name__ == '__main__':
    unittest.main()