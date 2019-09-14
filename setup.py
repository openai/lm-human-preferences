import os
import sys
import pkg_resources

from setuptools import setup, find_packages

os.environ['CC'] = 'g++'

setup(name='lm_human_preferences',
      version='0.0.1',
      packages=find_packages(include=['lm_human_preferences']),
      include_package_data=True,
)
