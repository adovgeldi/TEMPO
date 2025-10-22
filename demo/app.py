"""
TEMPO Forecasting Demo Application

Interactive Streamlit web application showcasing the capabilities of the TEMPO
time series forecasting library with an intuitive interface and rich visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import traceback
from datetime import datetime
import plotly.graph_objects as go
from pathlib import Path

# Add the parent directory to the path to import TEMPO modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    from synthetic_data_generator import SyntheticDataGenerator
    from demo_pipeline import DemoPipeline
    from visualization import DemoVisualizer, quick_forecast_plot, quick_performance_plot
except ImportError as e:
    st.error(f"""
    ❌ **Import Error**: {str(e)}
    
    **Solution**: Install the main TEMPO requirements first:
    
    ```bash
    # From the main TEMPO directory (parent folder)
    pip install -r requirements.txt
    
    # Then install demo requirements
    cd demo
    pip install -r requirements_demo.txt
    ```
    
    **Note**: The demo app requires all TEMPO dependencies to be installed.
    """)
    st.stop()


# Page configuration
st.set_page_config(
    page_title="Stop Flying Blind - Make Confident Business Decisions with TEMPO",
    page_icon="🎯 ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False


def load_sample_datasets():
    """Load or generate sample datasets"""
    data_dir = Path(__file__).parent / "data"
    
    # Check for existing CSV files
    csv_files = list(data_dir.glob("*.csv"))
    
    sample_datasets = {}
    
    # Add any existing CSV files
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            sample_datasets[csv_file.stem] = df
        except Exception as e:
            st.warning(f"Could not load {csv_file.name}: {e}")
    
    # Add synthetic data options
    generator = SyntheticDataGenerator()
    scenarios = generator.get_available_scenarios()
    
    for scenario, description in scenarios.items():
        sample_datasets[f"Synthetic: {scenario.replace('_', ' ').title()}"] = {
            'type': 'synthetic',
            'scenario': scenario,
            'description': description
        }
    
    return sample_datasets


def display_data_preview(data: pd.DataFrame):
    """Display data preview and basic statistics"""
    st.subheader("📊 Data Preview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("**First 10 rows:**")
        st.dataframe(data.head(10))
    
    with col2:
        st.write("**Dataset Info:**")
        st.write(f"• **Rows:** {len(data):,}")
        st.write(f"• **Columns:** {len(data.columns)}")
        st.write(f"• **Categories:** {data['category'].nunique()}")
        st.write(f"• **Date Range:** {data['date'].min()} to {data['date'].max()}")
        
        # Target variable statistics - auto-detect target column
        # Use same logic as sidebar selection for consistency
        if 'n_rented' in data.columns:
            target_col_preview = 'n_rented'
        else:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            target_col_preview = numeric_cols[0] if len(numeric_cols) > 0 else data.columns[-1]
        
        if target_col_preview in data.columns and data[target_col_preview].dtype in ['int64', 'float64']:
            target_stats = data[target_col_preview].describe()
            st.write(f"**Target Variable ({target_col_preview}):**")
            st.write(f"• Mean: {target_stats['mean']:.1f}")
            st.write(f"• Std: {target_stats['std']:.1f}")
            st.write(f"• Min: {target_stats['min']:.0f}")
            st.write(f"• Max: {target_stats['max']:.0f}")
        else:
            st.write(f"**Target Variable ({target_col_preview}): Non-numeric**")


def run_forecasting_pipeline(data: pd.DataFrame, config: dict):
    """Run the forecasting pipeline with progress tracking"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def progress_callback(message: str, progress: float):
        progress_bar.progress(progress)
        status_text.text(message)
    
    try:
        pipeline = DemoPipeline(
            target_col=config['target_col'],
            date_col=config['date_col'],
            category_col=config['category_col']
        )
        
        results = pipeline.run_forecasting(
            data=data,
            test_periods=config['test_periods'],
            selected_models=config.get('selected_models'),
            n_trials=config['n_trials'],
            progress_callback=progress_callback
        )
        
        progress_bar.progress(1.0)
        status_text.text("✅ Forecasting completed successfully!")
        
        return pipeline, results
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        raise e


def display_results(pipeline: DemoPipeline, results: dict):
    """Display forecasting results with visualizations"""
    st.success("🎉 Forecasting completed successfully!")
    
    # Summary metrics
    st.subheader("📈 Results Summary")
    
    summary = results['summary_metrics']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Categories Processed", summary['total_categories'])
    
    with col2:
        st.metric("Average Accuracy", f"{(100 - summary['avg_wmape']):.1f}%")
    
    with col3:
        st.metric("Highest Accuracy", f"{(100 - summary['best_wmape']):.1f}%")
    
    with col4:
        st.metric("Models Used", len(summary['models_used']))
    
    # Visualizations
    st.subheader("📊 Forecast Visualizations")
    
    viz = DemoVisualizer()
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Time Series", "Model Performance", "Category Performance", "Error Analysis"])
    
    with tab1:
        st.write("**Individual Time Series Forecasts**")
        
        # Controls for visualization
        col1, col2, col3 = st.columns(3)
        with col1:
            categories = results['output_df']['category'].unique()
            selected_category = st.selectbox("Select category to view:", ['All Categories'] + list(categories))
        with col2:
            periods_to_show = st.slider("Periods to show:", min_value=10, max_value=100, value=18, 
                                      help="Number of recent time periods to display for better forecast visibility")
        with col3:
            available_models = results['output_df']['model_name'].unique()
            selected_models_viz = st.multiselect(
                "Select models to show:",
                available_models,
                default=list(available_models),
                help="Choose which models to display in the forecast plots"
            )
        
        # Filter data by selected models
        if selected_models_viz:
            filtered_df = results['output_df'][results['output_df']['model_name'].isin(selected_models_viz)]
        else:
            filtered_df = results['output_df']
            st.warning("⚠️ No models selected. Please select at least one model to display.")
        
        if not filtered_df.empty and selected_models_viz:
            if selected_category == 'All Categories':
                # Multiple categories subplot
                if len(categories) <= 6:
                    fig = viz.create_time_series_subplots(filtered_df, last_n_periods=periods_to_show)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Show first few categories
                    fig = viz.create_time_series_subplots(filtered_df, max_categories=6, last_n_periods=periods_to_show)
                    st.plotly_chart(fig, use_container_width=True)
                    st.info(f"Showing first 6 of {len(categories)} categories. Use the dropdown to view individual categories.")
            else:
                # Single category plot
                fig = viz.create_forecast_comparison_plot(filtered_df, selected_category, last_n_periods=periods_to_show)
                st.plotly_chart(fig, use_container_width=True)
            
            # Show model info
            model_count = len(selected_models_viz)
            category_count = len(filtered_df['category'].unique()) if selected_category == 'All Categories' else 1
            st.info(f"📊 Displaying {model_count} model(s) across {category_count} categor{'ies' if category_count > 1 else 'y'}")
    
    with tab2:
        st.write("**Model Performance Comparison**")
        
        comparison_data = pipeline.get_model_comparison_data()
        if not comparison_data.empty:
            fig = viz.create_model_performance_plot(comparison_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Model performance table
            st.write("**Detailed Model Performance:**")
            model_perf_df = pd.DataFrame(summary['model_performance']).T.round(3)
            st.dataframe(model_perf_df)
        else:
            st.warning("No model comparison data available.")
    
    with tab3:
        st.write("**Performance by Category**")
        
        fig = viz.create_category_performance_plot(results['param_df'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Best models table
        st.write("**Best Model by Category:**")
        best_models_df = results['param_df'][['category', 'model_name', 'cv_avg_metric']].copy()
        best_models_df.columns = ['Category', 'Best Model', 'WMAPE']
        best_models_df['WMAPE'] = best_models_df['WMAPE'].round(3)
        st.dataframe(best_models_df, hide_index=True)
    
    with tab4:
        st.write("**Forecast Error Analysis**")
        
        fig = viz.create_error_distribution_plot(results['output_df'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Error statistics
        forecast_data = pipeline.get_forecast_vs_actual_data()
        if not forecast_data.empty:
            errors = ((forecast_data['eval_test_preds'] - forecast_data['true_vals']) / 
                     forecast_data['true_vals'] * 100)
            errors = errors[np.isfinite(errors)]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Error %", f"{errors.mean():.1f}%")
            with col2:
                st.metric("Std Error %", f"{errors.std():.1f}%")
            with col3:
                st.metric("Median Error %", f"{errors.median():.1f}%")
    
    # Download section
    st.subheader("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download predictions
        csv = results['output_df'].to_csv(index=False)
        st.download_button(
            label="📄 Download Forecast Data (CSV)",
            data=csv,
            file_name=f"tempo_forecast_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Download model parameters
        param_csv = results['param_df'].to_csv(index=False)
        st.download_button(
            label="⚙️ Download Model Parameters (CSV)",
            data=param_csv,
            file_name=f"tempo_model_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.title("🎯 Stop Flying Blind - Make Confident Decisions with TEMPO")
    st.markdown("**Interactive demonstration of the TEMPO time series forecasting library**")
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    # Data selection
    st.sidebar.subheader("1️⃣ Data Selection")
    
    sample_datasets = load_sample_datasets()
    
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Upload CSV file", "Use sample dataset"]
    )
    
    data = None
    
    if data_source == "Upload CSV file":
        uploaded_file = st.sidebar.file_uploader(
            "Upload your CSV file",
            type=['csv'],
            help="CSV should have columns: date, category, and a numeric target variable (auto-detected)"
        )
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.sidebar.success("✅ File uploaded successfully!")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading file: {e}")
    
    else:  # Sample dataset
        dataset_names = list(sample_datasets.keys())
        selected_dataset = st.sidebar.selectbox("Select sample dataset:", dataset_names)
        
        if selected_dataset:
            dataset_info = sample_datasets[selected_dataset]
            
            if isinstance(dataset_info, dict) and dataset_info.get('type') == 'synthetic':
                # Generate synthetic data
                with st.sidebar:
                    with st.spinner("Generating synthetic data..."):
                        generator = SyntheticDataGenerator()
                        data = generator.generate_demo_dataset(dataset_info['scenario'])
                st.sidebar.success("✅ Synthetic data generated!")
            else:
                # Load existing dataset
                data = dataset_info
                st.sidebar.success("✅ Sample dataset loaded!")
    
    # Configuration options (only show if data is loaded)
    if data is not None:
        st.sidebar.subheader("2️⃣ Forecasting Configuration")
        
        # Column mapping
        available_columns = data.columns.tolist()
        
        date_col = st.sidebar.selectbox("Date column:", available_columns, 
                                       index=available_columns.index('date') if 'date' in available_columns else 0)
        
        category_col = st.sidebar.selectbox("Category column:", available_columns,
                                          index=available_columns.index('category') if 'category' in available_columns else 1)
        
        target_col = st.sidebar.selectbox("Target variable:", available_columns,
                                        index=available_columns.index('n_rented') if 'n_rented' in available_columns else 2)
        
        # Forecasting parameters
        test_periods = st.sidebar.slider("Test periods (holdout):", 1, 30, 6,
                                       help="Number of periods to hold out for testing forecast accuracy")
        
        n_trials = st.sidebar.slider("Optuna trials per model:", 5, 50, 10,
                                    help="More trials = better optimization but longer runtime")
        
        # Model selection
        available_models = ['prophet', 'xgboost', 'expsmooth', 'lightgbm', 'knn']
        selected_models = st.sidebar.multiselect(
            "Select models to use:", 
            available_models,
            default=['prophet', 'xgboost', 'expsmooth'],
            help="Fewer models = faster processing"
        )
        
        config = {
            'date_col': date_col,
            'category_col': category_col,
            'target_col': target_col,
            'test_periods': test_periods,
            'n_trials': n_trials,
            'selected_models': selected_models if selected_models else ['prophet']
        }
        
        # Run forecasting button
        run_button = st.sidebar.button("🚀 Run Forecasting", type="primary", use_container_width=True)
        
    # Main content area
    if data is not None:
        display_data_preview(data)
        
        # Validate data
        pipeline = DemoPipeline(target_col=config['target_col'], 
                              date_col=config['date_col'], 
                              category_col=config['category_col'])
        
        is_valid, errors = pipeline.validate_data(data.copy())
        
        if not is_valid:
            st.error("❌ Data validation failed:")
            for error in errors:
                st.error(f"• {error}")
        else:
            st.success("✅ Data validation passed!")
            
            if run_button and not st.session_state.processing:
                st.session_state.processing = True
                
                with st.spinner("Running forecasting pipeline..."):
                    try:
                        pipeline, results = run_forecasting_pipeline(data, config)
                        st.session_state.pipeline = pipeline
                        st.session_state.results = results
                        st.session_state.processing = False
                        st.rerun()
                        
                    except Exception as e:
                        st.session_state.processing = False
                        st.error(f"❌ Forecasting failed: {str(e)}")
                        st.error("**Error details:**")
                        st.code(traceback.format_exc())
    
    # Display results if available
    if st.session_state.results is not None:
        display_results(st.session_state.pipeline, st.session_state.results)
    
    elif data is None:
        # Landing page
        st.markdown("""
       

Every day, your business makes critical decisions based on forecasts. What if those predictions could be more accurate?

### 💼 Transform Your Operations:
- 📦 **Inventory Management**: Eliminate costly stockouts while reducing excess inventory carrying costs
- 💰 **Budget Planning**: Create reliable financial forecasts that support strategic investments
- 🚛 **Supply Chain**: Anticipate disruptions before they impact your customers
- 👥 **Workforce Planning**: Right-size teams based on predicted demand patterns
- 📈 **Growth Strategy**: Make expansion decisions backed by data, not guesswork

### 🏆 The TEMPO Advantage:
- **Proven Results**: Organizations reduce forecast errors by up to 25% in their first quarter
- **Risk Reduction**: Identify potential problems weeks or months in advance
- **Cost Control**: Optimize resource allocation across your entire operation
- **Competitive Edge**: React faster to market changes than competitors using outdated methods

### 🚀 See Your Data Come Alive:
1. **Upload your historical business data** - sales, costs, demand, whatever drives your decisions
2. **Watch multiple AI models compete** to find the best predictions for your specific business
3. **Get actionable insights immediately** - no waiting for IT or external consultants
4. **Export results** directly into your existing planning tools

### 💡 Ready to Transform Your Forecasting?
This demo shows real capabilities with real data. See how TEMPO could impact your bottom line.
        """)
        
        # Show sample data formats
        with st.expander("📝 Sample Data Format"):
            sample_data = pd.DataFrame({
                'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
                'category': ['Product A', 'Product A', 'Product B'], 
                'n_rented': [100, 105, 80]
            })
            st.dataframe(sample_data)


if __name__ == "__main__":
    main()