#!/bin/bash

conda activate myenv
pip install xvfbwrapper
pip install imageio[ffmpeg]
pip install PyOpenGL
pip install matplotlib packaging pandas pyyaml requests scikit-learn scipy torch torchvision
pip install pydantic tqdm
conda install gym
conda install pyvirtualdisplay
conda install -c conda-forge ffmpeg
conda install -c anaconda python-opengl

# or use pip if the package is not available through conda
pip install gym[atari]
pip install 'pyqt5<5.13'
pip install 'pyqtwebengine<5.13'
pip install gym==0.25.2