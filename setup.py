import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="FundamentalAnalysis",
    packages=["FundamentalAnalysis"],
    version="0.2.0",
    license="MIT",
    description="Collects a large set of fundamental and stock data.",
    author="JerBouma",
    author_email="jer.bouma@gmail.com",
    url="https://github.com/JerBouma/FundamentalAnalysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=["fundamental", "analysis", "finance"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3"
    ],
)