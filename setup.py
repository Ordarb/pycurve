import codecs

from setuptools import setup, find_packages

packages = find_packages(include=['pycurve', 'pycurve.*'])

with codecs.open('README.txt','r', 'utf-8') as f:
    readme = f.read()

setup(
    name='NewsMeta',
    author='Quassel',
    author_email='sandro.braun@quassel.li',
    version='0.1',
    description='NewsMeta - Meta Data Enrichment',
    url='www.quassel.li',
    keywords=['data', 'news', 'meta', 'nlp'],
    long_description=readme,
    packages=packages
)
