from setuptools import setup, find_packages

setup(
    name="contextstress",
    version="0.1.0",
    description="Benchmarking Reasoning Collapse in Long-Context Language Models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="ContextStress Authors",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "pandas>=2.0",
    ],
    extras_require={
        "openai": ["openai>=1.0"],
        "anthropic": ["anthropic>=0.25"],
        "viz": ["matplotlib>=3.7"],
        "all": ["openai>=1.0", "anthropic>=0.25", "matplotlib>=3.7"],
    },
    entry_points={
        "console_scripts": [
            "contextstress=contextstress.__main__:main",
        ],
    },
)
