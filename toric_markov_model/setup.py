from setuptools import find_packages, setup

setup(
    name="toric_markov_model",
    version="0.2.0",
    description="Experimental trading research with Toric encoders; no demonstrated profitable edge",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(include=["toric_markov_model", "toric_markov_model.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "joblib",
        "requests",
    ],
)
