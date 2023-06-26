from setuptools import setup

def parse_requirements(filename):
	lines = []
	for line in open(filename):
		line = line.strip()
		if line and not line.startswith("#"):
			lines.append(line)
	return lines

setup(name='DALL-E',
        version='0.1',
        description='PyTorch package for the discrete VAE used for DALL·E.',
        url='http://github.com/openai/DALL-E',
        author='Aditya Ramesh',
        author_email='aramesh@openai.com',
        license='BSD',
        packages=['dall_e'],
        install_requires=parse_requirements('requirements.txt'),
        zip_safe=True)
