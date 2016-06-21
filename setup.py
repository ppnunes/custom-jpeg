#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vector Quantization coding
===================
A file compactor
"""
from setuptools import setup, find_packages

# http://sourceforge.net/projects/pyqt/files/PyQt4/PyQt-4.11.4/PyQt-x11-gpl-4.11.4.tar.gz
install_requires = [
    'bitarray>=0.8.1',
    'ipdb>=0.9.0',
    'optparse-pretty>=0.1.1',
    'bintrees>=2.0.2',
    'numpy>=1.11.0',
    'writebits>=0.0.9',
    'matplotlib>=1.5.1'
]


setup(
    name="cjpeg",
    version='0.0.8',
    author='Luiz Oliveira, João Marcello',
    author_email='ziuloliveira@gmail.com',
    url='https://github.com/Ziul/custom-jpeg',
    entry_points={
        'console_scripts': [
            'cjpeg = cjpeg:main',
        ]},
    description='A program to compact/descompact images',
    long_description=__doc__,
    license='GPLv3',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    zip_safe=False,
    test_suite="tests.run.runtests",
    dependency_links=[
        'git+https://github.com/Ziul/writebits.git#egg=writebits-0.1.0'],
    install_requires=install_requires,
    include_package_data=True,
    # package_data={'': ['codebook.lbg']},
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3',
        'Topic :: Utilities',
    ],
)
