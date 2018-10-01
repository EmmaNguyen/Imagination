# Script to download and setup Anaconda for python 3
# Source: https://medium.com/@GalarnykMichael/install-python-on-ubuntu-anaconda-65623042cb5a
# Usage: $bash download_and_setup_anaconda.sh

# You can change what anaconda version you want at
# https://repo.continuum.io/archive/
wget https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh
bash Anaconda3-4.2.0-Linux-x86_64.sh -b -p ./third_party/anaconda

rm Anaconda3-4.2.0-Linux-x86_64.sh

# Refresh bashrc for using this login
echo 'export PATH="./Imagination/third_party/anaconda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Create a PATH to a virtual environment under this project for next time
echo 'export PATH="./Imagination/third_party/anaconda/bin:$PATH"' >> ~/.bash_profile

# Get the latest version
conda update conda --yes
