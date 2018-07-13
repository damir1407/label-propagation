import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="labelpropagation",
    version="0.0.1",
    author="Damir Varešanović",
    author_email="damir.varesanovic@gmail.com",
    description="Community detection using Label Propagation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/damir1407/labelpropagation",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)