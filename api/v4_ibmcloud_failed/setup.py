from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath( path.dirname( __file__ ) )

setup(
    name='app-name',
    version='1.0.2',
    description='Python Flask app for uploading images to a pre-built WAtson custom classifier and viewing results in a web app',
    license='Apache-2.0'
)

