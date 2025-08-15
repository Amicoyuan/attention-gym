from setuptools import find_packages, setup

setup(
    name="attention-gym",
    version="0.1.0",
    author="xingyuanjie.xyj",
    description="attention-gym triton-base library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/RiseAI-Sys/attention-gym",
    packages=find_packages(),
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    entry_points={},
)
