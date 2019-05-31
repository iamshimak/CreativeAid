from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install
from pip._internal.req import parse_requirements
from pip._internal.download import PipSession

packages = [
    'nltk',
    'numpy==1.16.2',
    'gensim==3.7.1',
    'spacy==2.1.0',
    'setuptools==40.2.0',
    'vc== 2018.7.10',
    'wheel==0.31.1',
    'wincertstore==0.2',
    'scikit-learn==0.19.2',
    'networkx', 'summa', 'pywikibot'
]

setup(
    name='creativeaid',
    version='0.3.7',
    author='Ahamed Shimak',
    author_email='shimak2013@gmail.com',
    install_requires=packages,
    packages=find_packages(exclude=[]),
    description='Creative title generation framework',
    cmdclass={
        # 'install': DownloadGloveModel
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
