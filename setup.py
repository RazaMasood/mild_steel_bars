from setuptools import find_packages, setup


setup(
    name="mild_steel_bars",
    version="0.0.0",
    author="Mohd_Masood_Raza",
    author_email="razamasood954@gami.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[]
)