from setuptools import setup, find_packages

setup(name='ML_Tensorflow',
      version='0.1',
      description="Conventional modular elements of ML models using TensorFlow",
      packages=find_packages(where="python"),
      package_dir={'':'python'},
      install_requires=["tensorflow","numpy"],
      )
