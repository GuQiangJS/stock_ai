from setuptools import setup, find_packages

import versioneer

NAME = 'stock_ai'


def readme():
    with open('README.md', encoding='utf-8') as f:
        return f.read()


setup(
    name=NAME,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description=readme(),
    long_description=readme(),
    author='GuQiangJS',
    author_email='guqiangjs@gmail.com',
    url='https://github.com/GuQiangJS/stock_ai.git',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
    ],
    keywords='data',
    install_requires=['QUANTAXIS', 'TensorFlow', 'pandas'],
    packages=find_packages(),
    zip_safe=False,
    exclude_package_data={'': ['test_*.py']}
)