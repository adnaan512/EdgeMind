"""EdgeMind AI — A Modular Framework for Efficient Deep Learning and Edge AI."""

from setuptools import setup, find_packages

setup(
    name="edgemind",
    version="0.1.0",
    author="EdgeMind AI Contributors",
    description="A Modular Framework for Efficient Deep Learning and Edge AI",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/edgemind-ai",
    packages=find_packages(exclude=["tests*", "notebooks*", "docs*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pyyaml>=6.0",
        "rich>=13.0.0",
        "Pillow>=9.0.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
