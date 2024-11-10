from setuptools import setup, find_packages

setup(
    name="dmbreg",
    version="0.1.1",
    description="A library of simple regression models.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="WhatsWrongHere",
    author_email="zaboradaniil@gmail.com",
    url="https://github.com/WhatsWrongHere/dmbreg",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
       "scipy>=1.7.0"
    ],
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
