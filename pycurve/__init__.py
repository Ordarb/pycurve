__title__ = 'pycurve'
__author__ = 'Sandro Braun'
__copyright__ = 'Copyright 2017, Ordarb'
__version__ = '0.1.0'


from .curve import Curve
from .data import Data

import logging as _logging

try:  # Python 2.7+
    from logging import NullHandler as _NullHandler
except ImportError:
    class _NullHandler(_logging.Handler):
        def emit(self, record):
            pass

_logging.getLogger(__name__).addHandler(_NullHandler())
