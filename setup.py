from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install

setup(
    name='creativeaid!',
    version='0.1',
    author='Ahamed Shimak',
    author_email='shimak2013@gmail.com',
    install_requires=[],
    packages=find_packages(),
    description='Creative title generation framework',
    cmdclass={
        # 'install': DownloadGloveModel
    }
)
