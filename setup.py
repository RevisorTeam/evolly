import os

from setuptools import find_packages, setup


def read(rel_path):
	"""
	Read file
	Reference:
	https://github.com/tensorflow/similarity/blob/master/setup.py
	"""
	here = os.path.abspath(os.path.dirname(__file__))
	# intentionally *not* adding an encoding option to open, See:
	#   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
	with open(os.path.join(here, rel_path), "r") as fp:
		return fp.read()


def get_version(rel_path):
	for line in read(rel_path).splitlines():
		if line.startswith("__version__"):
			delim = '"' if '"' in line else "'"
			return line.split(delim)[1]
	raise RuntimeError("Unable to find version string.")


setup(
	name='evolly',
	version=get_version("evolly/__init__.py"),
	description="Evolutionary NAS framework",
	long_description=read("README.MD"),
	long_description_content_type="text/markdown",
	author="The Revisor Team",
	author_email="revisorteam@pm.me",
	url="https://github.com/revisorteam/evolly",
	license="MIT",
	install_requires=[
		"yacs",
		"numpy>=1.19.5",
		"pandas",
		"matplotlib",
		"tabulate",
		"diagrams",
		"fvcore",
		"opencv-python>=4.5.5.0",
		"tensorflow_similarity==0.16.7",
	],
	extras_require={
		"tensorflow": ["tensorflow>=2.3"],
		"tensorflow-gpu": ["tensorflow-gpu>=2.3"],
		"tensorflow-cpu": ["tensorflow-cpu>=2.3"],
		"torch": ["torch>=1.9.0"],
		"torchvision": ["torchvision>=0.10.0"],
	},
	classifiers=[
		"Development Status :: 5 - Production/Stable",
		"Environment :: Console",
		"License :: OSI Approved :: MIT Software License",
		"Intended Audience :: Science/Research",
		"Programming Language :: Python :: 3",
		"Topic :: Scientific/Engineering :: Artificial Intelligence",
	],
	packages=find_packages(),
)
