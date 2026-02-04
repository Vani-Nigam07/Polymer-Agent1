from setuptools import setup, find_packages

setup(
    name="open-macromolecular-genome-polyagent",
    version="0.1.0",
    packages=find_packages(include=["OpenMacromolecularGenome", "OpenMacromolecularGenome.*"]),
)
