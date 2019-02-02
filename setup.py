import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="labelpropagation",
    version="0.0.1",
    author="Damir Varešanović",
    author_email="damir.varesanovic@gmail.com",
    description="Programming library for network community detection based on label propagation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://bitbucket.org/Damir1407/label-propagation",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3.5",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    install_requires=['numpy', 'networkx'],
)