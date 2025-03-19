from setuptools import setup, find_packages

setup(
    name="pyplot2csv",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy", "matplotlib", "pandas", "scipy", "sklearn"],
    entry_points={
        "console_scripts": [
            "pyplot2csv=pyplot2csv.cli:main",
        ],
    },
)
