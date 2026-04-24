import os
from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'sphviewer2.core',             
        ['src/main.cpp'],              
        include_dirs=[
            pybind11.get_include(),    
            'include'                  
        ],
        language='c++',
        extra_compile_args=[
            '-std=c++14', 
            '-O3', 
            '-pthread', 
            '-ffast-math'
            ], 
    ),
]

setup(
    name='py-sphviewer2',              
    version='2.0.3',
    description='Efficient SPH projection using the Benitez-Llambay (2025) algorithm',
    ext_modules=ext_modules,
    packages=['sphviewer2'],         
    install_requires=['numpy', 'pybind11'],
    zip_safe=False,
)