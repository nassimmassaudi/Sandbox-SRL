from setuptools import setup, find_packages

setup(
    name='DMLab',
    version='0.1.0',
    description='Gym Environment for Deepmind Lab',
    author='Nassim Massaudi',
    author_email='nassim.el.massaudi@umontreal.ca',
    packages=find_packages(),  # Automatically finds all packages in your directory
    install_requires=[
        'gym',  # Add other dependencies if required
        'numpy',
        'opencv-python',
        'pygame',
        'deepmind_lab',  # Ensure this is installed in your environment
    ],
    entry_points={
        'gym.envs': [
            'DeepmindLabEnv-v0 = DMLab.env:DeepmindLabEnvironment',
            'DeepmindLabNavEnv-v0 = DMLab.env:DeepmindLabMazeNavigationEnvironment',
            'ContinuousDeepmindLabEnv-v0 = DMLab.env:ContinuousDeepmindLabEnvironment',
        ],
    },
)
