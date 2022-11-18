#!/usr/bin/env python


def main():
    from setuptools import setup, find_packages

    version_dict = {}
    version_filename = "boxtree/version.py"
    with open(version_filename) as version_file:
        exec(compile(version_file.read(), version_filename, "exec"), version_dict)

    setup(
        name="boxtree",
        version=version_dict["VERSION_TEXT"],
        description="Quadtree/octree building in Python and OpenCL",
        long_description=open("README.rst").read(),
        author="Andreas Kloeckner",
        author_email="inform@tiker.net",
        license="MIT",
        url="https://documen.tician.de/boxtree/",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Other Audience",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Natural Language :: English",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Information Analysis",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Topic :: Scientific/Engineering :: Visualization",
            "Topic :: Software Development :: Libraries",
            "Topic :: Utilities",
        ],
        packages=find_packages(),
        python_requires="~=3.8",
        install_requires=[
            "Mako>=0.7.3",

            "pytools>=2018.4",
            "pyopencl>=2018.2.2",
            "cgen>=2013.1.2",
            "arraycontext>=2021.1",
        ],
        extras_require={
            "test": ["pytest>=2.3"],
        },
    )


if __name__ == "__main__":
    main()
