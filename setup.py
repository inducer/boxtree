#!/usr/bin/env python
# -*- coding: latin1 -*-

import distribute_setup
distribute_setup.use_setuptools()

from setuptools import setup

try:
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
    # 2.x
    from distutils.command.build_py import build_py

setup(name="htree",
      version="2013.1",
      description="Hierarchical tree building in Python",
      long_description=open("README.rst", "rt").read(),
      author="Andreas Kloeckner",
      author_email="inform@tiker.net",
      license = "MIT",
      url="http://wiki.tiker.net/HTree",
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Other Audience',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Software Development :: Libraries',
        'Topic :: Utilities',
        ],

      packages=["htree"],
      install_requires=[
          "pyopencl>=2012.2",
          "Mako>=0.7",
          "pytest>=2",
          ],

      # 2to3 invocation
      cmdclass={'build_py': build_py})
