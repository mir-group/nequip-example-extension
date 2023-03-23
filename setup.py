from setuptools import setup, find_packages

setup(
    name="nequip-example-extension",
    version="0.1.0",
    author="Albert Musaelian",
    python_requires=">=3.8",
    packages=find_packages(
        include=["nequip_example_extension", "nequip_example_extension.*"]
    ),
    install_requires=["nequip"],
    zip_safe=True,
)
