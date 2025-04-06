#!/bin/bash

set -e  # Exit on error
set -u  # Treat unset variables as errors

echo "▶️ Installing Python 3.10, venv, and pip tools..."

# Install Python 3.10 if not available
sudo apt update
sudo apt install -y software-properties-common curl gnupg lsb-release

# Add deadsnakes repo for Python 3.10
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev

echo "✅ Python 3.10 installed."

# Create virtual environment
echo "📦 Creating virtual environment with Python 3.10..."
python3.10 -m venv .venv
source .venv/bin/activate

# Confirm Python version
echo "🐍 Using Python version: $(python --version)"

# Install requirements
echo "📥 Installing required Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Environment setup complete!"
