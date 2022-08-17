# from distutils.core import setup
from setuptools import find_packages, setup

reqs = ["numpy", 'dataclasses;python_version<"3.7"']
# extra_reqs = {"pres": ["plotly", "jupyterlab"]}
setup(
    name="CGN",
    version="1.0",
    description="CGN w/ dynamics",
    author="Iakovos Tsouros",
    # packages=find_packages(include=["src*", "util*"]),
    packages=find_packages(),
    python_requires=">3.10.*",
    install_requires=reqs,
    entry_points={
        "console_scripts": [
            "cgn-run=cgndyna.src.experiment:test_only",
        ]
    },
    # extras_require=extra_reqs,
)
