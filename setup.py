#!/usr/bin/env python
from setuptools import setup


setup(
    name="nmf_imaging",
    version="1.0",
    author="Bin Ren",
    author_email="bin.ren@jhu.edu",
    url="https://github.com/seawander/nmf_imaging",
    py_modules=["nmf_imaging"],
    description="Postprocessing Code for High Contrast Imaging using Vectorized Nonnegative Matrix Factorization",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    install_requires=['numpy', 'scipy']
)
