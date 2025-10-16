#!/bin/bash

echo "=========================================="
echo "Churn Prediction System - Setup Script"
echo "=========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Download spaCy model
echo ""
echo "Downloading spaCy language model..."
python -m spacy download en_core_web_sm

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p data/raw data/processed models/saved logs

# Generate sample data
echo ""
echo "Generating sample dataset..."
python scripts/generate_data.py

# Train model
echo ""
echo "Training churn prediction model..."
python scripts/train_model.py

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start the API server, run:"
echo "  uvicorn src.api.main:app --reload"
echo ""
echo "Or use Docker:"
echo "  docker-compose up --build"
echo ""
