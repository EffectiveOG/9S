# setup.py

from setuptools import setup, find_packages

setup(
    name="jarvis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "asyncio",
        "aioconsole",
        "logging",
    ],
)