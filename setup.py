from setuptools import setup, find_packages

meta = {}
with open("weigps/meta.py") as fp:
    exec(fp.read(), meta)

# Package meta-data.
NAME = meta["__title__"]
DESCRIPTION = "Wei's Gas-phase Polariton Simulation."
URL = "https://github.com/Weitheskmt/WeiGPS"
MAIL = meta["__mail__"]
AUTHOR = meta["__author__"]
VERSION = meta["__version__"]
KEYWORDS = "gas-phase rovibrational-polariton gps"

REQUIRED = ["numpy<2", "scipy", "matplotlib", "scikit-learn"]

EXTRAS = {
    "docs": ["Sphinx>=1.4", "sphinx_rtd_theme"],
    "test": ["pytest", "pytest-cov", "pytest-mock", "ezyrb>=v1.2.1.post2205"],
}

LDESCRIPTION = (
    "The main challenge in polariton chemistry is the collective effect. "
    "When studying individual molecules in a Fabry-P\ ́erot cavity, "
    "the interaction between light and matter is so small that it’s often ignored. "
    "To achieve strong light-matter interactions, a large number of molecules must collectively couple with light.\n" 
    "In our study, we simulate up to a million gas-phase molecules in a Fabry-P\ ́erot cavity, "
    "each with weak individual interactions with light. This approach allows us to address the collective challenges in polariton chemistry.\n"
    "Our simulation demonstrates that even with disorder and rotations, collective effects are observed in the spectra. "
    "Specifically, we find that increasing rotational frequencies result in larger Rabi splitting between lower and upper polaritons. "
    "However, randomness in rotational phases tends to reduce this splitting. "
    "Larger Rabi splitting helps polaritons withstand molecular level disorder. "
    "These findings are expected to be validated by gas-phase polariton experiments.\n"
)

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LDESCRIPTION,
    author=AUTHOR,
    author_email=MAIL,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords=KEYWORDS,
    url=URL,
    license="MIT",
    packages=find_packages(),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    zip_safe=False,
)
