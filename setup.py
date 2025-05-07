#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from packaging import version
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppression des avertissements de dépendance
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration du projet
setup(
    name="ai_trading",
    version="0.1.0",
    description="Système de trading algorithmique avec IA",
    author="Crypto AI Team",
    author_email="contact@example.com",
    url="https://github.com/yourusername/Crypto",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        # Les dépendances sont gérées par requirements.txt
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
) 