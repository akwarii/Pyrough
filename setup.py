from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Pyrough",
    version="1.0",
    description="A tool for rough samples constructions",
    author="Hugo Iteney",
    url="https://github.com/jamodeo12/Pyrough",
    author_email="hugo.iteney@im2np.fr",
    packages=find_packages(),
    install_requires=requirements,
)
