#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Backward-compatible setup.py for older pip/setuptools builds.

The canonical project configuration lives in pyproject.toml. This file is
kept so the package remains installable with legacy pip versions that only
recognize setup.py-based builds.
"""

import os
from pathlib import Path

from setuptools import find_namespace_packages, setup


HERE = Path(__file__).parent.resolve()
VERSION_FILE = HERE / "py" / "just_tiling" / "_version.py"

# Read the version without importing the package.
about = {}
with VERSION_FILE.open(encoding="utf-8") as f:
    exec(f.read(), about)
__version__ = about["__version__"]


# Read runtime dependencies from requirements.txt so they stay in one place.
def read_requirements():
    req_file = HERE / "requirements.txt"
    requirements = []
    if req_file.exists():
        with req_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    requirements.append(line)
    return requirements


setup(
    name="just_tiling",
    version=__version__,
    description="Tile scheduling for the JUST galaxy/galaxy cluster survey",
    long_description=(HERE / "README.rst").read_text(encoding="utf-8"),
    long_description_content_type="text/x-rst",
    url="https://github.com/zjdingastro/just_tiling",
    license="BSD-3-Clause",
    package_dir={"": "py"},
    packages=find_namespace_packages(where="py"),
    python_requires=">=3.8",
    install_requires=read_requirements(),
)
