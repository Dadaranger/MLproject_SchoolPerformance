from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    """
    This function reads a requirements file and returns a list of packages.

    """
    requirements = []
    with open(file_path, "r") as f:
       requirements = f.readlines()
       requirements = [req.replace("\n", "") for req in requirements]
       
       if HYPEN_E_DOT in requirements:
           requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name="MLproject_SchoolPerformance",
    version="0.1",
    author="Lei",
    author_email="akatoshleiwu@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)