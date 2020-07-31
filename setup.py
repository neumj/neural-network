from setuptools import setup, find_packages

reqs = [
    "matplotlib",
    "numpy",
    "pandas",
    "sklearn",
    "yaml"
]

conda_reqs = [
    "matplotlib",
    "numpy",
    "pandas",
    "sklearn",
    "yaml"
]

test_pkgs = []

setup(
    name="shallownn",
    python_requires='>3.4',
    description="Package for neural network experimentation",
    url="https://github.com/neumj/shallow-network",
    install_requires=reqs,
    conda_install_requires=conda_reqs,
    test_requires=test_pkgs,
    packages=find_packages(),
    include_package_data=True
)
