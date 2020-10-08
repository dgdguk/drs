"""
A simple setup.py

Thanks to https://realpython.com/pypi-publish-python-package/ for the rather
good walkthrough on how to do this.
"""

import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="drs",
    version="1.0.0",
    description="Dirichlet Rescale Algorithm",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/dgdguk/drs",
    author="David Griffin",
    author_email="dgdguk@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["drs"],
    include_package_data=True,
    install_requires=["numpy", "scipy"],
)
