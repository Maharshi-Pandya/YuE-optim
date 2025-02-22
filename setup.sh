#!/bin/bash

apt update
apt install -y git-lfs

git lfs install

git clone https://huggingface.co/m-a-p/xcodec_mini_infer

pip install -r requirements.txt

# download required models
huggingface-cli download m-a-p/YuE-s1-7B-anneal-en-cot --local-dir m-a-p/YuE-s1-7B-anneal-en-cot
huggingface-cli download m-a-p/YuE-s2-1B-general --local-dir m-a-p/YuE-s2-1B-general
