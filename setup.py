from setuptools import setup

setup(
    name = "MyMovingMNIST",
    version = "0.0.1",
    author = "Yi Huang",
    author_email = "yhuang2@bnl.gov",
    description = ("Generate moving number pairs from MNIST"),
    license = "MIT",
    keywords = "example documentation tutorial",
    url = "TBD",
    packages=['moving_mnist'],
    long_description="",
    install_requires=[
        "numpy",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
                "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
)
