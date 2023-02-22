import os
import re
from pathlib import Path

from setuptools import find_packages, setup

setup(
    name='dcnet',
    author='yingkaining',
    license='GPLv2',
    url='https://github.com/yingkaining/dcnet',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Utilities',
    ],
    python_requires='>=3.7',
    install_requires=['scipy>=1.6', 'torch>=1.10', 'torchvision>=0.7.0', 'matplotlib',
                      'tensorboard', 'terminaltables'],
    extras_require={
        'full': [
            'torchvision>=0.7.0'
        ]
    },
    packages=find_packages(include=['dcnet']))
