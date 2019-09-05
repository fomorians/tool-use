from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    "gym==0.12.5",
    "attrs==19.1.0",
    "fomoro-pyoneer==0.3.0",
    "pycolab==1.2",
    "gym_pycolab==1.0.0",
    "gym_tool_use==1.0.0",
    "tf-nightly-gpu-2.0-preview==2.0.0.dev20190903",
    "tfp-nightly==0.9.0.dev20190904",
    "imageio==2.5.0",
]

setup(
    name="tool_use",
    version="0.0.0",
    url="https://github.com/fomorians/tool-use",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
)
