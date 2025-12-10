#!/bin/bash
# Quick start script for RL Training Lab

echo "🤖 Starting Reinforcement Learning Training Lab..."
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Install requirements
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

# Launch Streamlit app
echo "🚀 Launching Streamlit app..."
echo ""
streamlit run app.py
