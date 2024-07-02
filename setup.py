from pathlib import Path
from setuptools import find_namespace_packages, setup


readme_file = Path(__file__).parent / "README.md"
requirements_file = Path(__file__).parent / "requirements" / "requirements.in"

setup(
    name='geqie',
    version='1.0',
    description='General Equation of Quantum Image Encoding Framework',
    long_description=readme_file.read_text(),
    license="Apache v2.0",
    author='Rafał Potempa',
    author_email='rafal.potem@gmail.com',

    install_requires=requirements_file.read_text().splitlines(),
    packages=find_namespace_packages(".", exclude=["build*"]),
    entry_points={
        'console_scripts': [
            'geqie = geqie.cli:cli',
        ],
    }
)
