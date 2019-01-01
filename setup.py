import setuptools
from pyACS import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyACS",
    version=__version__,
    author="Nils Haentjens",
    author_email="nils.haentjens@maine.edu",
    description="Unpack and calibrate binary data from a WetLabs ACS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OceanOptics/pyACS/",
    packages=setuptools.find_packages(),
    install_requires=['numpy'],
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research"
    ]
)