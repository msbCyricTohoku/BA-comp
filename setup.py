from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="tse_ba_comp",
    version="0.1.6",
    description="A library for estimating Biological Age using classical and ML methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mehrdad S. Beni & Gary Tse",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "matplotlib",
        "scipy"
    ],
    python_requires='>=3.7',
)
