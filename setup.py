from setuptools import setup, find_packages

setup(
    name="topoEEG",
    version="0.1.0",
    description="Topological Data Analysis in EEG recordings",
    author="Miriam Esteve",
    author_email="miriam.estevecampello@uchceu.es",
    maintainer="Antonio Falco",
    maintainer_email="afalco@uchceu.es",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas",
        "numpy>=1.21.0",
        "matplotlib",
        "scipy",
        "scikit-learn",
        "persim",
        "ripser",
        "gudhi",
        "giotto-tda",
        "mne>=0.23.0",
        "POT",
        "concurrent.futures",
        "torch",
        "tensorflow",
        "imbalanced-learn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
