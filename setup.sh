#!/bin/bash

# Check if the secret key is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <secret_key>"
    exit 1
fi

SECRET_KEY="$1"

apt update
apt install -y git-lfs

git lfs install

# Use the secret key variable in the Git clone URL
git clone https://${SECRET_KEY}@github.com/Maharshi-Pandya/YuE-optim.git

cd YuE-optim

git clone https://huggingface.co/m-a-p/xcodec_mini_infer

pip install -r requirements.txt
