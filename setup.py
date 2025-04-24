from setuptools import setup, find_packages

setup(
    name="ai_trading",
    version="1.0.0",
    description="Suite d'outils pour l'apprentissage par renforcement appliqué au trading de cryptomonnaies",
    author="Crypto Trading AI Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "tensorflow",
        "scikit-learn",
        "requests",
        "nltk",
    ],
) 