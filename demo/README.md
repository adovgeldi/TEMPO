# TEMPO Forecasting Demo Application

Interactive web application demonstrating the capabilities of the TEMPO time series forecasting library.

## 🎯 Overview

This demo application provides a user-friendly interface to the TEMPO forecasting library, allowing users to:
- Upload their own CSV data or use synthetic datasets
- Configure forecasting parameters through an intuitive UI
- Run multiple forecasting models with automated hyperparameter optimization
- Visualize results with interactive charts and performance metrics
- Download forecasts and model parameters

## 🚀 Quick Start

### Option 1: Use Virtual Environment (Recommended)

```bash
# 1. Activate the existing virtual environment
cd /path/to/TEMPO
source .venv/bin/activate

# 2. Install demo requirements
cd demo
pip install -r requirements_demo.txt

# 3. Run the demo
./run_demo.sh
```

### Option 2: Manual Setup

```bash
# 1. Navigate to TEMPO directory
cd /path/to/TEMPO

# 2. Install TEMPO in development mode
pip install -e .

# 3. Install main requirements
pip install -r requirements.txt

# 4. Install demo requirements
cd demo
pip install -r requirements_demo.txt

# 5. Run the demo
streamlit run app.py
```

### Option 3: With Virtual Environment

```bash
# 1. Activate the virtual environment (if using one)
cd /path/to/TEMPO
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 2. Follow Option 2 steps above
```

**Important**: The demo requires the TEMPO library to be importable as `tempo_forecasting`. The automated setup handles this correctly.

The application will open in your web browser at `http://localhost:8501`.

## 📁 Files Overview

- **`app.py`** - Main Streamlit web application
- **`demo_pipeline.py`** - Simplified wrapper around TEMPO pipelines
- **`synthetic_data_generator.py`** - Generate realistic synthetic datasets
- **`visualization.py`** - Enhanced interactive visualizations using Plotly
- **`data/`** - Directory for user-uploaded CSV files
- **`requirements_demo.txt`** - Additional Python dependencies

## 📊 Data Format

Your CSV file should contain the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `date` | Date/timestamp | `2023-01-01` |
| `category` | Time series identifier | `Product A` |
| `n_rented` | Target variable to forecast | `150` |

### Example Data:
```csv
date,category,n_rented
2023-01-01,Product A,150
2023-01-02,Product A,165
2023-01-01,Product B,89
2023-01-02,Product B,92
```

## 🎲 Synthetic Data Scenarios

The demo includes several realistic synthetic data scenarios:

1. **Retail Sales** - Seasonal patterns, holiday effects, promotions
2. **Equipment Rental** - Business day patterns, seasonal trends
3. **SaaS Metrics** - Growth trends, business cycles
4. **Energy Consumption** - Daily/seasonal cycles, usage patterns

## ⚙️ Configuration Options

### Forecasting Parameters
- **Test Periods**: Number of periods to hold out for accuracy testing (default: 6)
- **Optuna Trials**: Hyperparameter optimization trials per model (default: 10)
- **Model Selection**: Choose which models to include in the comparison

### Available Models
- **Prophet** - Facebook's time series forecasting tool
- **XGBoost** - Gradient boosting with feature engineering
- **LightGBM** - Fast gradient boosting
- **Exponential Smoothing** - Classical statistical method
- **K-NN** - K-nearest neighbors approach

## 📈 Output Features

### Interactive Visualizations
- **Time Series Plots**: Actual vs predicted values with confidence intervals
- **Model Performance**: Scatter plots comparing WMAPE vs MAE
- **Category Analysis**: Performance breakdown by time series
- **Error Distribution**: Histogram of forecast errors

### Performance Metrics
- **WMAPE** (Weighted Mean Absolute Percentage Error) - Primary metric
- **MAE** (Mean Absolute Error) - Secondary metric
- **Model comparison** across categories
- **Cross-validation results** with multiple time windows

### Export Options
- **Forecast Data**: CSV with predictions and actuals
- **Model Parameters**: Best hyperparameters for each model
- **Interactive Charts**: Can be saved as PNG/HTML

## 🔧 Customization

### Adding New Synthetic Scenarios
Extend the `SyntheticDataGenerator` class in `synthetic_data_generator.py`:

```python
def _generate_custom_scenario(self, **kwargs):
    # Your custom data generation logic
    return pd.DataFrame(data)
```

### Custom Visualizations
Add new plot types in `visualization.py` using the `DemoVisualizer` class:

```python
def create_custom_plot(self, data):
    # Your custom Plotly visualization
    return fig
```

### Model Configuration
Modify the `demo_pipeline.py` to adjust:
- Default model selection
- Optuna optimization parameters
- Cross-validation windows
- Performance metrics

## 🐛 Troubleshooting

### Common Issues

1. **Module Import Errors**
   ```bash
   # Ensure you're in the right directory and TEMPO is installed
   pip install -e . # From main TEMPO directory
   ```

2. **Data Validation Failures**
   - Check column names match expected format
   - Ensure dates can be parsed by pandas
   - Verify numeric target variable
   - Minimum 100 data points required

3. **Memory Issues with Large Datasets**
   - Reduce number of Optuna trials
   - Limit to fewer models
   - Use smaller test periods

4. **Slow Performance**
   - Reduce Optuna trials (5-10 for demo)
   - Select fewer models
   - Use synthetic data for testing

### Performance Tips

- Start with synthetic data to verify functionality
- Use 3-5 Optuna trials for quick testing
- Focus on 2-3 models for initial exploration
- Monitor memory usage with large datasets

## 📚 Further Information

For more details about the underlying TEMPO library:
- See `../README.md` for full library documentation
- Check `../claude.md` for developer guidance
- Review `../example_use/main.ipynb` for advanced usage

## 🤝 Contributing

To contribute to the demo application:
1. Follow the git workflow described in the main README
2. Test changes with multiple data scenarios
3. Update documentation for new features
4. Ensure UI remains intuitive and responsive

## 📄 License

This demo application follows the same proprietary license as the main TEMPO library. See `../LICENSE` for details.