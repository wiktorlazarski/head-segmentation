import runpy

import setuptools

DEV_REQUIREMENTS = [
    "black",
    "flake8",
    "isort",
    "jupyterlab",
    "pre-commit",
    "pytest",
]

# Parse requirements.txt file
install_requires = [line.strip() for line in open("requirements.txt").readlines()]

# Get long description
with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

__version__ = runpy.run_path("head_segmentation/_version.py")["__version__"]

# Setup package
setuptools.setup(
    name="head_segmentation",
    version=__version__,
    author="Wiktor Łazarski, Jakub Szumski",
    author_email="wjlazarski@gmail.com",
    description="Human head semantic segmentation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wiktorlazarski/head-segmentation",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    include_package_data=True,
    extras_require={"dev": DEV_REQUIREMENTS},
)
