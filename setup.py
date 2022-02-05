import runpy

import setuptools

# Parse requirements
install_requires = [line.strip() for line in open("requirements.txt").readlines()]

# Get long description
with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

__version__ = runpy.run_path("ai_awesome/_version.py")["__version__"]

# Setup package
setuptools.setup(
    name="Replace with project name",
    version=__version__,
    author="Replace with your name",
    author_email="Replace with your email",
    description="Give short description of your project.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your_project_link",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
)
