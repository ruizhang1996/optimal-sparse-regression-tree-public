import platform
import os
import pathlib
import distro

from setuptools import find_packages
from skbuild import setup

cmake_args = []

if platform.system() == "Windows" or (platform.system() == "Linux" and distro.id() == "centos"):
    assert "VCPKG_INSTALLATION_ROOT" in os.environ, \
        "The environment variable \"VCPKG_INSTALLATION_ROOT\" must be set before running this script."
    toolchain_path = pathlib.Path(os.getenv("VCPKG_INSTALLATION_ROOT")) / "scripts/buildsystems/vcpkg.cmake"
    cmake_args.append("-DCMAKE_TOOLCHAIN_FILE={}".format(toolchain_path))

print("Additional CMake Arguments = {}".format(cmake_args))

setup(
    name="osrt",
    version="0.1.1",
    description="Implementation of Optimal Sparse Regression Trees",
    author="UBC Systopia Research Lab",
    url="https://github.com/ruizhang1996/optimal-sparse-regression-tree-public",
    license="BSD 3-Clause",
    packages=find_packages(where='.'),
    cmake_install_dir="osrt",
    cmake_args=cmake_args,
    python_requires=">=3.7",
    long_description=pathlib.Path("README_PyPI.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    install_requires=["setuptools",
                      "wheel",
                      "attrs",
                      "packaging>=20.9",
                      "editables==0.2;python_version>'3.0'",
                      "pandas",
                      "sklearn",
                      "sortedcontainers",
                      "gmpy2",
                      "matplotlib"
                    ]
)
