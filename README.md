# FRTL Stage Four Vision

This repository contains the code responsible for detecting gestures and mapping them to commands necessary to complete the fourth stage of the Flying Robot Trial League 2024.

## Python Environment Setup

To run this project, you need to set up a Python environment. Follow the bash command below to create and configure the environment:

```bash
cd /tmp
echo "sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get install -y python3-pip
sudo apt-get install -y python3-venv
cd ~
python -m venv stage4_env
cd stage4_env
source bin/activate
pip install rospkg
pip install ultralytics" > config_env.sh && source config_env.sh
```
