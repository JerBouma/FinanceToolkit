import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="FundamentalAnalysis",
    packages=["FundamentalAnalysis"],
    version="0.2.5",
    license="MIT",
    description="Fully-fledged Fundamental Analysis package capable of collecting 20 years of Company Profiles,\
    Financial Statements, Ratios and Stock Data of 13.000+ companies.",
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