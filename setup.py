from setuptools import setup, find_packages

setup(
    name='mmseg_denoiser',
    version='0.1.0',
    description='Pseudo-label denoiser based on mmsegmentation',
    author='Hung Nguyen',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.8',
        'mmcv-full>=1.3.0',
        'mmsegmentation>=0.20.0',
    ],
)
