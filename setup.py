# setup.py
#
# Packaging metadata now lives in pyproject.toml. This shim remains only for
# `pip install -e .` with older tooling; it reads configuration from
# pyproject.toml.

from setuptools import setup

setup()
