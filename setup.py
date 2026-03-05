# Author: Yipeng Sun
# Last Change: Sun Oct 31, 2021 at 04:58 AM +0100

import setuptools
import subprocess
import codecs
import os.path

from distutils.core import setup


###########
# Helpers #
###########

with open('README.md', 'r') as ld:
    long_description = ld.read()


def get_pipe_output(cmd):
    cmd_splitted = cmd.split(' ')
    proc = subprocess.Popen(cmd_splitted, stdout=subprocess.PIPE)
    result = proc.stdout.read().decode('utf-8')
    return result.strip('\n')


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]

    raise RuntimeError("Unable to find version string.")


#########
# Setup #
#########

setup(
    name='TrackerOnlyEmu',
    version=get_version('TrackerOnlyEmu/__init__.py'),
    author='Yipeng Sun',
    author_email='syp@umd.edu',
    description='A collection of tools for emulating tracker responses',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/umd-lhcb/TrackOnlyEmu',
    packages=setuptools.find_packages(),
    scripts=[
        'scripts/run2-rdx-trg_emu.py',
        'scripts/run2-rdx-hlt1.py',
        'scripts/run2-rdx-l0_global_tis.py',
        'scripts/run2-rdx-l0_hadron_tos.py',
    ],
    include_package_data=True,
    install_requires=[
        'numpy',
        'pyyaml',
        'scikit-learn~=1.0.0',
        'xgboost~=1.5.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        # 'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent'
    ],
)
