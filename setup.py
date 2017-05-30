
'''
Yield Curve Estimation
'''

import sys
import os
import codecs

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

packages = ['pycurve']

with codecs.open('README.txt','r','utf-8') as f:
    readme = f.read()

setup(
    name                = 'pycurve',
    author              = 'Sandro Braun',
    author_email        = 'ordnas.nuarb@gmail.com',
    version             = '0.1.0',
    description         = 'Yield Curve Estimation - parametric and smoothing splines',
    url                 = 'No Homepage',
    long_description    = readme,
    packages            = packages
    )
