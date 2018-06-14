import unittest

from engine.utils.data_sets import load_adults_dataset


class TestAdultsDataset(unittest.TestCase):
    def test(self):
        x, y = load_adults_dataset()
        print(x)
        print(y)
