import os
import re
from pathlib import Path

from setuptools import find_packages, setup


# def get_version():
#     version_file = Path('mmhoidet') / '__init__.py'
#     with open(version_file, encoding='utf-8') as f:
#         lines = f.readlines()
#     for line in lines:
#         if line.startswith('__version__'):
#             exec(line.strip())
#     return locals()['__version__']


def get_readme():
    with open('README.md', encoding='utf-8') as f:
        content = re.sub(r'## Model([\s\S]*)respectively.\n\n', '', f.read())
    return content


setup(
    name='mmhoidet',
    # version=get_version(),
    author='kaining',
    author_email='kennying99@gmail.com',
    license='GPLv2',
    url='https://github.com/noobying/mmhoidett',
    description='A HOI DET codebase based on MMDet',
    long_description=get_readme(),
    long_description_content_type='text/markdown',
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
    install_requires=['scipy>=1.6', 'torch>=1.6', 'mmcv-full>=1.3,<1.4', 'torchvision>=0.7.0', 'matplotlib',
                      'tensorboard', 'terminaltables'],
    extras_require={
        'full': [
            'mmcv-full>=1.3,<1.4', 'torchvision>=0.7.0'
        ]
    },
    packages=find_packages(include=['mmhoidet', 'zjutcv']))


# setup(
#     name='zjutcv',
#     packages=find_packages(),
# )

