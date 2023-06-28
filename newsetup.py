from setuptools import setup

def parse_requirements(filename):
    with open(filename) as f:
        lines = [line.strip() for line in f if not line.startswith("#")]
    return lines

setup(
    name='DALL-E',
    version='0.2',
    description='PyTorch package for the discrete VAE used for DALL·E.',
    url='http://github.com/openai/DALL-E',
    author='Aditya Ramesh',
    author_email='aramesh@openai.com',
    license='BSD',
    packages=['dall_e'],
    install_requires=parse_requirements('requirements.txt'),
    zip_safe=True
)
