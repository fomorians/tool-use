from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    "gym",
    "ray[debug,rllib]",
    "attrs",
    "pyoneer",
    "pycolab",
    "gym_pycolab",
    "gym_tool_use",
    "tf-nightly-2.0-preview",
    "tfp-nightly",
]

setup(
    name="tool_use",
    version="0.0.0",
    url="https://github.com/fomorians/tool-use",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
)
