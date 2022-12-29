import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mvtst",
    version="0.0.1",
    author="Isaac Sears",
    author_email="isaac.j.sears@gmail.com",
    description="Package around George Zerveas' time series transformer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/isears/mvts_transformer",
    project_urls={"Bug Tracker": "https://github.com/isears/mvts_transformer/issues",},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"mvtst": "mvtst"},
    install_requires=["torch", "numpy"],  # TODO: ?
    packages=["mvtst"],
    python_requires=">=3.6",
)
