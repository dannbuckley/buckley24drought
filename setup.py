from setuptools import setup, find_packages

setup(
    packages=find_packages(
        where="src",
        include=["buckley24drought*"],
    ),
    package_dir={"": "src"},
)
