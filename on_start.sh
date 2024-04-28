echo 'Starting the installation at $(date)' | tee setup.log; if [ $(id -u) -ne 0 ]; then echo 'Script must be run as root.' | tee -a setup.log; exit 1; fi; apt-get update | tee -a setup.log; apt-get install -y wget git build-essential | tee -a setup.log; sudo -v ;
sudo apt-get install libglfw3 | tee -a setup.log;
sudo apt-get install libglfw3-dev | tee -a setup.log;
sudo apt-get install nano | tee -a setup.log;
sudo apt-get install zip | tee -a setup.log;
apt-get install xvfb | tee -a setup.log;
wget https://rclone.org/install.sh | sudo bash;
source install.sh | tee -a setup.log; \
rm install.sh; \
  echo -e "[n41000-homework-h27]\ntype = drive\nscope = drive\ntoken = {\"access_token\":\"ya29.a0Ad52N39d7N_u7mrZU7Qgp8BIM3-iXGWSf7tytbKBkb6HoOWxj2qlv3XQOEWeOp9qlpdUUpdVRfGG7rxgXTiuzvDVdTcfAcUhqgtE4BF8812K7UqVh07e9sJdHyqcUvxgFpnY8CmTi90qXE1OWdWCYy5KFYZ46SE_6VzpaCgYKAcgSARMSFQHGX2MiXTM7wf0K3R0UaPrBZuwxPg0171\",\"token_type\":\"Bearer\",\"refresh_token\":\"1//050ZrbMYodfUGCgYIARAAGAUSNwF-L9IrlYufMJ4aVLM-jnJS1uY_zQ0_XjFLyJR2CLp2fMl1VdYfoDy7yyWoy7Bm888pnzY4Xg8\",\"expiry\":\"2024-04-14T06:03:10.933992-04:00\"}\nteam_drive = " > ~/.config/rclone/rclone.conf; rm -rf /opt/miniconda3; wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh; bash /tmp/miniconda.sh -b -p /opt/miniconda3 | tee -a setup.log; source /opt/miniconda3/etc/profile.d/conda.sh; /opt/miniconda3/bin/conda init; /opt/miniconda3/bin/conda create -n cloudspace python=3.9 -y | tee -a setup.log; source /opt/miniconda3/bin/activate cloudspace; git clone git clone https://github.com/nassimmassaudi/Sandbox-SRL.git | tee -a setup.log; cd Sandbox-SRL || exit; git checkout mico | tee -a setup.log; \ 
conda env create --file environment.yaml --name cloudspace;  \
conda activate cloudspace; \
pip install torch; \
pip install tensorboard; \
pip install torchvision; \
pip install rich; \
pip install scikit-learn; \
pip install scikit-video; \
pip install wandb; \
pip install stable-baseline3; \
pip install termcolor; \
pip install pillow; \
pip install opencv-python; \
pip install numpy; \
pip install matplotlib; \
pip install hydra-core; \
pip install gym; \
pip install gymnasium; \
pip install dm-control; \
pip install nvitop; \
pip install imageio; \
pip install scikit-image; \ 
pip install scikit-video; \
sudo -v ; wget https://rclone.org/install.sh | sudo bash; \
source install.sh | tee -a setup.log; \ 
rm install.sh; \
 echo 'Setup completed successfully at $(date)' | tee -a setup.log





# sudo apt-get install libglfw3
# sudo apt-get install libglfw3-dev
# sudo apt-get install nano
# sudo apt-get install zip


# conda activate cloudspace
# git clone https://github.com/nassimmassaudi/Sandbox-SRL.git
# git checkout mico
# cd Sandbox-SRL

# conda env remove --name cloudspace
# conda env create --file environment.yaml --name cloudspace
# conda activate cloudspace
# pip install torch
# pip install rich
# pip install scikit-learn
# pip install scikit-video
# pip install wandb
# pip install stable-baseline3
# pip install termcolor
# pip install pillow
# pip install opencv-python
# pip install numpy
# pip install matplotlib
# pip install hydra-core
# pip install gym
# pip install gymnasium
# pip install dm-control