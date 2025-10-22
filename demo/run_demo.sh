#!/bin/bash
# Quick start script for TEMPO Demo Application

echo "🚀 Starting TEMPO Forecasting Demo Application..."
echo

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Please run this script from the demo/ directory."
    exit 1
fi

# Check if we're in the parent directory and activate venv if available
if [ -f "../.venv/bin/activate" ]; then
    echo "🔍 Found virtual environment, activating..."
    source ../.venv/bin/activate
    echo "✅ Virtual environment activated"
elif [ -f ".venv/bin/activate" ]; then
    echo "🔍 Found virtual environment, activating..."
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
fi

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Error: Python not found. Please install Python 3.11."
    exit 1
fi

# Check if streamlit is installed
if ! python -c "import streamlit" &> /dev/null; then
    echo "⚠️  Streamlit not found. Installing demo requirements..."
    pip install -r requirements_demo.txt
fi

# Check if main TEMPO requirements are satisfied
echo "🔍 Checking TEMPO dependencies..."
if ! python -c "import tempo_forecasting" &> /dev/null; then
    echo "⚠️  TEMPO library not found. Please install main requirements first:"
    echo "   cd .."
    echo "   pip install -r requirements.txt"
    echo "   cd demo"
    echo ""
    echo "❌ Cannot continue without TEMPO dependencies."
    exit 1
fi

echo "✅ Dependencies verified."
echo "🌐 Launching Streamlit application..."
echo "📱 The app will open in your web browser at: http://localhost:8501"
echo
echo "🛑 Press Ctrl+C to stop the application"
echo

# Launch the application
streamlit run app.py