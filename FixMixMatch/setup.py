#nsml: pytorch/pytorch
from distutils.core import setup

setup(
    name='ladder_networks',
    version='1.0',
    install_requires=[
        'tqdm',
        'numpy',
        'efficientnet_pytorch'
    ]
)
