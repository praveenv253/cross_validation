#!/usr/bin/env python3

from setuptools import setup

setup(
    name='cross_validation',
    version='0.1.0',
    description=('A module implementing a general-purpose '
                 'cross validation framework'),
    author='Praveen Venkatesh',
    url='https://github.com/praveenv253/cross_validation',
    packages=['cross_validation', ],
    install_requires=['numpy', ],
    setup_requires=['pytest-runner', ],
    tests_require=['pytest', ],
    license='MIT',
)
