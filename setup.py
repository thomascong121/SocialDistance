from setuptools import setup, find_packages
with open("README.md","r") as fh:
    long_description = fh.read()
requirements = [
    'numpy',
    'tqdm',
    'mxnet',
    'matplotlib',
    'gluoncv'
]
setup(
    # Metadata
    name = 'SDD',
    version = '0.1.9',
    description = 'lightweight video detection',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url = 'https://github.com/thomascong121/SocialDistance',
    author = 'SocialDistance model contributors',
    author_email = 'thomascong@outlook.com',
    packages = find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements,
    include_package_data= True,
    zip_safe= True
)

#steps:
'''
rm -r SDD.egg-info/ build/ dist/
python3 -m pip install --user --upgrade setuptools wheel
python3 -m pip install --user --upgrade twine
python3 setup.py sdist bdist_wheel
python3 -m twine upload dist/*
'''