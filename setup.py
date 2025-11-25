from setuptools import setup, find_packages

setup(
    name="tasas",
    version="0.1.0",
    description="Interest Rate Change Probability Model - Fed Funds Rate",
    author="Rodrigo Arana",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "yfinance>=0.2.28",
        "fredapi>=0.5.1",
        "streamlit>=1.28.0",
    ],
    python_requires=">=3.9",
)
