#from distutils.core import setup, Extension
from setuptools import setup , find_packages
import numpy as np


setup(
        name = 'spatial_spin_monte_carlo',
        version = '1.0',
		author = 'Matthew Garrod',
        packages=['spatial_spin_monte_carlo']
      )
