from setuptools import setup, find_packages
import io
from os import path

here = path.abspath(path.dirname(__file__))
with io.open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.rstrip() for line in f]

setup(
    name='qmoms',
    version='0.1',
    packages=find_packages(),
    description='Option-implied moments from implied volatility surface data',
    author='Grigory Vilkov',
    author_email='vilkov@vilkov.net',
    license='MIT',
    platforms=['any'],
    keywords=['implied variance', 'variance swap', 'VIX', 'MFIV', 'CVIX', 'skewness'],
    install_requires=requirements,
    package_data={
        'qmoms': ['data/*.csv'],
    },
    include_package_data=True,
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    )
