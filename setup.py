from setuptools import setup

setup(name='aviary',
      version='0.1rc0',
      description='Tools for calculating kinematic ages',
      url='http://github.com/RuthAngus/aviary',
      author='Ruth Angus',
      author_email='ruthangus@gmail.com',
      license='MIT',
      packages=['aviary'],
      install_requires=['numpy', 'tqdm', 'astropy', 'matplotlib'],
      zip_safe=False)
