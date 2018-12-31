import os
import unittest
import pandas as pd

from ..pycurve.utils import load_pickle
from ..pycurve.parametric import Parametric
from ..pycurve.spline import SmoothingSpline


def load_data():
    return load_pickle(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'swiss.pkl'))


class TestCurve(unittest.TestCase):

    def test_ns(self):
        self.data = load_pickle(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'swiss.pkl'))
        print(self.data.head())
        p = Parametric(self.data, 1)
        p.fit('ns')
        print(p.instruments.head())
        self.assertIsInstance(p.instruments, pd.DataFrame)
