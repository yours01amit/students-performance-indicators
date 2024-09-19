from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requires(file_path:str)->List[str]:

    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [i.replace('\n','')for i in requirements]

    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)


    


setup(
    name="Machine Learning project",
    version='0.0.1',
    author='Amit Kumar',
    author_email='emmysingh019@gmail.com',
    packages=find_packages(),
    install_requires=get_requires('requirements.txt')

)