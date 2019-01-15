from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'tensorflow-probability',
    'gym',
    'attrs',
    'pyoneer',
]

setup(
    name='tool_use',
    version='0.0.0',
    url='https://github.com/fomorians/tool-use',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True)
