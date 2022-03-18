# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup script for PLUR.

This script will install PLUR as a Python module.

See: https://github.com/google-research/plur
"""

import pathlib
from setuptools import find_packages
from setuptools import setup

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

# Keep this list in sync with `requirements.txt`.
install_requires = [
    'absl-py >= 0.12.0',
    'apache-beam >= 2.34.0',
    'GitPython >= 3.1.24',
    'google-cloud-storage >= 1.43.0',
    'immutabledict >= 2.2.1',
    'javalang >=0.13.0',
    'tensorflow >= 2.5.0',
    'flax >= 0.3.6',
    'jax >= 0.2.26',
    'jaxlib >= 0.1.75',
    'trax >= 1.4.1',
    'numpy >=1.19.5',
    'scipy >= 1.6.1 ',
    'requests >= 2.24.0',
    'tqdm >= 4.62.3',
    'tensor2tensor >= 1.15.7',
    'unidiff >= 0.7.0',
    'nltk >= 3.6.6'
]

plur_description = (
    'PLUR: A framework for Program Learning Understanding and Repair.')

setup(
    name='plur',
    version='0.0.1',
    description=plur_description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/google-research/plur',
    author='The PLUR Team',
    classifiers=[
        'Development Status :: 0 - Alpha',

        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',

        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',

        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',

    ],
    keywords='plur, program-synthesis, machine-learning, research, software-engineering',
    include_package_data=True,
    packages=find_packages(exclude=['docs']),
    install_requires=install_requires,
    project_urls={  # Optional
        'Documentation': 'https://github.com/google-research/plur',
        'Bug Reports': 'https://github.com/google-research/plur/issues',
        'Source': 'https://github.com/google-research/plur',
    },
    license='Apache 2.0',
)
