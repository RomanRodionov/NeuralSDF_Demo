#!/bin/bash

set -e

echo "Creating Python virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies from requirements.txt..."
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "Error: requirements.txt not found!"
    exit 1
fi

#echo "Cloning tiny-cuda-nn repository..."
#git clone --recursive git@github.com:NVlabs/tiny-cuda-nn.git

#echo "Installing tiny-cuda-nn PyTorch bindings..."
#cd tiny-cuda-nn/bindings/torch
#python setup.py install

echo "Setup complete."
