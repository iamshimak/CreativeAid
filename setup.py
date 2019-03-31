from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install

setup(
    name='creativeaid',
    version='0.3.4',
    author='Ahamed Shimak',
    author_email='shimak2013@gmail.com',
    scripts=['creativeaid'],
    install_requires=[],
    packages=find_packages(),
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
