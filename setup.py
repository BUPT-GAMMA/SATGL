from setuptools import setup, find_packages

setup(
    name='satgl',
    version='0.1',     
    author='clinozoisite',  
    license='MIT',      
    packages=find_packages(), 
    package_data={
        'satgl': ['external/*'],
    },
)
