__title__ = 'pycurve'
__author__ = 'SandroBraun'
__copyright__ = 'Copyright 2017'


from .curve import Curve


import logging

try:  # Python 2.7+
    from logging import NullHandler
except ImportError:
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass

logging.getLogger(__name__).addHandler(NullHandler())
