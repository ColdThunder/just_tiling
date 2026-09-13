#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Backward-compatible setup.py for older pip/setuptools builds.

The canonical project configuration lives in pyproject.toml. This file is
kept so the package remains installable with legacy pip and Python versions
that do not recognize pyproject.toml-based builds.
"""

import os

try:
    from setuptools import find_namespace_packages, setup
except ImportError:
    from setuptools import setup
    find_namespace_packages = None


HERE = os.path.dirname(os.path.abspath(__file__))
VERSION_FILE = os.path.join(HERE, "py", "just_tiling", "_version.py")

# Read the version without importing the package.
about = {}
with open(VERSION_FILE, "r") as f:
    exec(f.read(), about)
__version__ = about["__version__"]


# Read runtime dependencies from requirements.txt so they stay in one place.
def read_requirements():
    req_file = os.path.join(HERE, "requirements.txt")
    requirements = []
    if os.path.exists(req_file):
        with open(req_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    requirements.append(line)
    return requirements


# Read the long description from README.rst.
def read_readme():
    readme_file = os.path.join(HERE, "README.rst")
    if os.path.exists(readme_file):
        with open(readme_file, "r") as f:
            return f.read()
    return ""


def find_packages_compat(where):
    """Find packages under ``where``, with a manual fallback for old setuptools.

    Old setuptools does not provide ``find_namespace_packages``, so we walk
    the directory tree ourselves and return all subdirectories as package
    names. This keeps the package installable with legacy environments while
    still supporting namespace packages on modern setuptools.
    """
    if find_namespace_packages is not None:
        return find_namespace_packages(where=where)

    base = os.path.join(HERE, where)
    packages = []
    for root, dirs, files in os.walk(base):
        # Do not recurse into hidden, cache, or packaging directories.
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".")
            and d != "__pycache__"
            and not d.endswith(".egg-info")
        ]
        rel = os.path.relpath(root, base)
        if rel == ".":
            continue
        parts = rel.split(os.sep)
        packages.append(".".join(parts))
    # Add the top-level package if any subpackages were found.
    if packages:
        top_pkg = packages[0].split(".")[0]
        if top_pkg not in packages:
            packages.insert(0, top_pkg)
    return packages


setup(
    name="just_tiling",
    version=__version__,
    description="Tile scheduling for the JUST galaxy/galaxy cluster survey",
    long_description=read_readme(),
    url="https://github.com/zjdingastro/just_tiling",
    license="BSD-3-Clause",
    package_dir={"": "py"},
    packages=find_packages_compat("py"),
    python_requires=">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*",
    install_requires=read_requirements(),
)
