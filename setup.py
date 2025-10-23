"""
Setup script for ml-optimizer-selector package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="ml-optimizer-selector",
    version="1.0.0",
    author="ML Labs",
    description="Automatically select the optimal solver for machine learning optimization problems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ml-optimizer-selector",
    py_modules=["optimizer_selector"],
    python_requires=">=3.8",
    install_requires=[
        "scikit-learn>=1.0.0",
        "numpy>=1.20.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="machine-learning optimization solver sklearn logistic-regression l-bfgs",
    project_urls={
        "Documentation": "https://github.com/yourusername/ml-optimizer-selector/blob/main/README.md",
        "Source": "https://github.com/yourusername/ml-optimizer-selector",
        "Bug Reports": "https://github.com/yourusername/ml-optimizer-selector/issues",
    },
)

