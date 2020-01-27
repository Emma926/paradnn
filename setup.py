from setuptools import setup

setup(
   name='paradnn',
   version='1.0',
   description='A tool that generates parameterized deep neural network models. It provides large “end-to-end” models covering current and future applications, and parameterizing the models to explore a much larger design space of DNN model attributes.',
   author='Emma Wang',
   author_email='emmawong926@gmail.com',
   packages=['paradnn'],
   install_requires=['python3', 'tensorflow'],
)
