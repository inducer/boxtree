#!/usr/bin/env python
# -*- coding: utf-8 -*-


def main():
    from setuptools import setup

    version_dict = {}
    init_filename = "boxtree/version.py"
    exec(compile(open(init_filename, "r").read(), init_filename, "exec"),
            version_dict)

    setup(name="boxtree",
          version=version_dict["VERSION_TEXT"],
          description="Quadtree/octree building in Python and OpenCL",
          long_description=open("README.rst", "rt").read(),
          author="Andreas Kloeckner",
          author_email="inform@tiker.net",
          license="MIT",
          url="http://wiki.tiker.net/BoxTree",
          classifiers=[
              'Development Status :: 3 - Alpha',
              'Intended Audience :: Developers',
              'Intended Audience :: Other Audience',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: MIT License',
              'Natural Language :: English',

              'Programming Language :: Python',
              'Programming Language :: Python :: 3',

              'Topic :: Scientific/Engineering',
              'Topic :: Scientific/Engineering :: Information Analysis',
              'Topic :: Scientific/Engineering :: Mathematics',
              'Topic :: Scientific/Engineering :: Visualization',
              'Topic :: Software Development :: Libraries',
              'Topic :: Utilities',
              ],

          packages=["boxtree"],
          python_requires="~=3.6",
          install_requires=[
              "pytools>=2018.4",
              "pyopencl>=2018.2.2",
              "Mako>=0.7.3",
              "pytest>=2.3",
              "cgen>=2013.1.2",
              ])


if __name__ == '__main__':
    main()
