#!/bin/bash
# Setup script for TEMPO Demo Application

echo "🔧 Setting up TEMPO Demo Application..."
echo

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: Please run this script from the demo/ directory."
    exit 1
fi

# Go to parent directory to set up TEMPO
cd ..

echo "📦 Installing TEMPO in development mode..."
pip install -e .

echo "📦 Installing TEMPO requirements..."
pip install -r requirements.txt

echo "📦 Installing demo requirements..."
cd demo
pip install -r requirements_demo.txt

echo "🧪 Testing imports..."
python -c "
import tempo_forecasting
from synthetic_data_generator import SyntheticDataGenerator
from demo_pipeline import DemoPipeline
from visualization import DemoVisualizer
print('✅ All imports successful!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Setup complete! You can now run the demo with:"
    echo "   ./run_demo.sh"
    echo "   or"
    echo "   streamlit run app.py"
else
    echo ""
    echo "❌ Setup failed. Please check the error messages above."
    exit 1
fi