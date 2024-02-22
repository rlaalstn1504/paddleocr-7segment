from setuptools import setup, find_packages

setup(
    name='paddleocr-7segment',
    version='0.1.0', 
    packages=['PaddleOCR/paddleocr'],
    # packages=find_packages(include=['*']),
    package_dir={"": "."},
)