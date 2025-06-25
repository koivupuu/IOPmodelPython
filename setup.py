from setuptools import setup, find_packages

setup(
    name='iopmodel',  # package name on PyPI or your choice
    version='0.1.0',
    description='IOP model Python package',
    author='Miika Koivisto',
    author_email='miika.j.koivisto@student.jyu.fi',
    packages=find_packages(include=['iopmodel', 'iopmodel.*']),
    include_package_data=True,  # to include data files if configured
    install_requires=[
        'numpy',
        'pandas',
        'scipy'
        # add other dependencies your package needs
    ],
    python_requires='>=3.7',
)
