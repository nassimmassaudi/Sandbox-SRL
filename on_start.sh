#!/bin/bash

# Log the start time
echo 'Starting the installation at $(date)' | tee setup.log

# Check if the script is run as root, exit if not
if [ $(id -u) -ne 0 ]; then 
    echo 'Script must be run as root.' | tee -a setup.log
    exit 1
fi

# Update package lists
apt-get update | tee -a setup.log

# Install necessary packages
apt-get install -y wget git build-essential libglfw3 libglfw3-dev nano zip xvfb ffmpeg | tee -a setup.log

# Download and install rclone
wget -O- https://rclone.org/install.sh | sudo bash | tee -a setup.log

# Configure rclone with an access token
RCLONE_ACCESS_TOKEN='{"access_token":"ya29.a0Ad52N39d7N_u7mrZU7Qgp8BIM3-iXGWSf7tytbKBkb6HoOWxj2qlv3XQOEWeOp9qlpdUUpdVRfGG7rxgXTiuzvDVdTcfAcUhqgtE4BF8812K7UqVh07e9sJdHyqcUvxgFpnY8CmTi90qXE1OWdWCYy5KFYZ46SE_6VzpaCgYKAcgSARMSFQHGX2MiXTM7wf0K3R0UaPrBZuwxPg0171","token_type":"Bearer","refresh_token":"1//050ZrbMYodfUGCgYIARAAGAUSNwF-L9IrlYufMJ4aVLM-jnJS1uY_zQ0_XjFLyJR2CLp2fMl1VdYfoDy7yyWoy7Bm888pnzY4Xg8","expiry":"2024-04-14T06:03:10.933992-04:00"}'
echo "[n41000-homework-h27]
type = drive
scope = drive
token = ${RCLONE_ACCESS_TOKEN}
team_drive = " > ~/.config/rclone/rclone.conf

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p /opt/miniconda3 | tee -a setup.log
source /opt/miniconda3/etc/profile.d/conda.sh
/opt/miniconda3/bin/conda init

# Create a new Conda environment
/opt/miniconda3/bin/conda create -n cloudspace python=3.9 -y | tee -a setup.log
source /opt/miniconda3/bin/activate cloudspace

# Clone a Git repository
git clone https://github.com/nassimmassaudi/Sandbox-SRL.git | tee -a setup.log
cd Sandbox-SRL || exit
git checkout mico | tee -a setup.log

# Create a Conda environment from a file
conda env create --file environment.yaml --name cloudspace
conda activate cloudspace

# Install Python packages
pip install torch tensorboard torchvision rich scikit-learn wandb stable-baselines3 termcolor pillow opencv-python numpy matplotlib hydra-core gym gymnasium dm-control nvitop imageio scikit-image
pip install git+https://github.com/scikit-video/scikit-video.git

# Final verification and cleanup
sudo -v
wget -O- https://rclone.org/install.sh | sudo bash | tee -a setup.log
rm install.sh

# Log the completion time
echo 'Setup completed successfully at $(date)' | tee -a setup.log