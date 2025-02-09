import os
from setuptools import find_packages, setup

cwd = os.path.dirname(os.path.abspath(__file__))


def _get_pytorch_version():
    return "torch"


setup(
    # Metadata
    name="lr_challenge",
    version="0.1.0",
    author="Carl Mueller",
    description="Demonstration Assisted Natural Policy Gradient Implementation using the Panda Gymnasium environment",
    license="MIT",
    # Package info
    packages=find_packages(exclude=()),
    install_requires=[
        _get_pytorch_version(),
        "gymnasium",
        "gymnasium[classic-control]",
        "gymnasium[other]",
        "numpy",
        "packaging",
        "cloudpickle",
        "matplotlib",
        "panda-gym",
        "tensorboard",
        "stable-baselines3",
        "opencv-python",
    ],
    extras_require={
        "tests": ["pytest"],
    },
)
