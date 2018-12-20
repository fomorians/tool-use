from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['gym[box2d]', 'attrs', 'tqdm', 'trfl']

setup(
    name='tool_use',
    version='0.0.0',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    dependency_links=[
        'http://github.com/deepmind/trfl/tarball/master#egg=trfl=0.0.0'
    ])
