from setuptools import setup

setup(
  name='retinawarp',
  version='0.1',
  packages=['retina'],
  install_rquires=['numpy', 'scipy', 'skimage', 'tensorflow'],
  url='https://github.com/dicarlolab/retinawarp',
  license='MIT',
  author='Pouya Bashivan',
  author_email='bashivan@mit.edu',
  description='Fisheye (Retina) transformation on images'
)
