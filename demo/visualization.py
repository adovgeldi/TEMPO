"""
Enhanced Visualization Components for TEMPO Demo Application

This module provides interactive and static visualizations using Plotly and Matplotlib
for displaying forecasting results in an appealing and informative way.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")


class DemoVisualizer:
    """Enhanced visualizations for TEMPO demo application"""
    
    def __init__(self):
        """Initialize the visualizer with default styling"""
        self.colors = {
            'actual': '#1f77b4',
            'forecast': '#ff7f0e', 
            'test': '#d62728',
            'train': '#2ca02c',
            'background': '#f8f9fa',
            'grid': '#e1e5e9'
        }
        
    def create_forecast_comparison_plot(self, 
                                      data: pd.DataFrame,
                                      category: str = None,
                                      title: str = None,
                                      last_n_periods: int = 18) -> go.Figure:
        """
        Create an interactive time series plot comparing forecasts to actuals
        
        Args:
            data: DataFrame with columns: date, category, true_vals, eval_test_preds, forecast
            category: Specific category to plot (None for all)
            title: Custom title for the plot
            last_n_periods: Number of recent periods to show (default 18)
            
        Returns:
            Plotly figure object
        """
        if category:
            plot_data = data[data['category'] == category].copy()
            default_title = f"Forecast vs Actual: {category}"
        else:
            plot_data = data.copy()
            default_title = "Forecast vs Actual (All Categories)"
        
        title = title or default_title
        
        plot_data['date'] = pd.to_datetime(plot_data['date'])
        plot_data = plot_data.sort_values('date')
        
        # Filter to show only the last N unique dates for better visibility
        unique_dates = sorted(plot_data['date'].unique())
        if len(unique_dates) > last_n_periods:
            last_n_dates = unique_dates[-last_n_periods:]
            plot_data = plot_data[plot_data['date'].isin(last_n_dates)].copy()
        
        fig = go.Figure()
        
        if category:
            # Single category plot with multiple models
            # First, add actual values (only once, deduplicated by date)
            actual_data = plot_data[~plot_data['true_vals'].isna()].drop_duplicates(subset=['date'])
            if not actual_data.empty:
                fig.add_trace(go.Scatter(
                    x=actual_data['date'],
                    y=actual_data['true_vals'],
                    mode='lines+markers',
                    name='Actual',
                    line=dict(color=self.colors['actual'], width=2),
                    marker=dict(size=4)
                ))
            
            # Add predictions for each model
            models = plot_data['model_name'].unique()
            model_colors = px.colors.qualitative.Set1[:len(models)]
            
            for i, model in enumerate(models):
                model_data = plot_data[plot_data['model_name'] == model]
                
                # Add test predictions for this model
                test_data = model_data[~model_data['eval_test_preds'].isna()]
                if not test_data.empty:
                    fig.add_trace(go.Scatter(
                        x=test_data['date'],
                        y=test_data['eval_test_preds'],
                        mode='lines+markers',
                        name=f'{model} - Test Forecast',
                        line=dict(color=model_colors[i % len(model_colors)], width=2, dash='dash'),
                        marker=dict(size=4, symbol='diamond')
                    ))
                
                # Add future forecasts for this model if available
                forecast_data = model_data[~model_data['forecast'].isna() & model_data['true_vals'].isna()]
                if not forecast_data.empty:
                    fig.add_trace(go.Scatter(
                        x=forecast_data['date'],
                        y=forecast_data['forecast'],
                        mode='lines+markers',
                        name=f'{model} - Future Forecast',
                        line=dict(color=model_colors[i % len(model_colors)], width=2),
                        marker=dict(size=4, symbol='circle-open')
                    ))
        else:
            # Multiple categories
            categories = plot_data['category'].unique()
            colors = px.colors.qualitative.Set1[:len(categories)]
            
            for i, cat in enumerate(categories):
                cat_data = plot_data[plot_data['category'] == cat]
                
                # Actual values
                fig.add_trace(go.Scatter(
                    x=cat_data['date'],
                    y=cat_data['true_vals'],
                    mode='lines',
                    name=f'{cat} (Actual)',
                    line=dict(color=colors[i % len(colors)], width=1.5)
                ))
                
                # Test predictions
                test_data = cat_data[~cat_data['eval_test_preds'].isna()]
                if not test_data.empty:
                    fig.add_trace(go.Scatter(
                        x=test_data['date'],
                        y=test_data['eval_test_preds'],
                        mode='lines',
                        name=f'{cat} (Test Forecast)',
                        line=dict(color=colors[i % len(colors)], width=1.5, dash='dash')
                    ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            showlegend=True
        )
        
        return fig
    
    def create_model_performance_plot(self, 
                                    comparison_data: pd.DataFrame,
                                    x_metric: str = 'wmape',
                                    y_metric: str = 'mae',
                                    title: str = None) -> go.Figure:
        """
        Create an interactive scatter plot of model performance
        
        Args:
            comparison_data: DataFrame with model performance metrics
            x_metric: Metric for x-axis
            y_metric: Metric for y-axis 
            title: Custom title
            
        Returns:
            Plotly figure object
        """
        title = title or f"Model Performance: {y_metric.upper()} vs {x_metric.upper()}"
        
        fig = px.scatter(
            comparison_data,
            x=x_metric,
            y=y_metric,
            color='model_type',
            size='wmape',
            hover_data=['category'],
            title=title,
            template='plotly_white'
        )
        
        fig.update_traces(marker=dict(
            size=comparison_data[x_metric] * 10,  # Scale for visibility
            opacity=0.7,
            line=dict(width=1, color='white')
        ))
        
        fig.update_layout(
            xaxis_title=x_metric.upper(),
            yaxis_title=y_metric.upper(),
            height=500,
            showlegend=True
        )
        
        return fig
    
    def create_metrics_summary_table(self, summary_metrics: Dict[str, Any]) -> go.Figure:
        """
        Create a summary table of key metrics
        
        Args:
            summary_metrics: Dictionary with summary statistics
            
        Returns:
            Plotly table figure
        """
        # Prepare table data
        metrics_data = [
            ['Total Categories', summary_metrics.get('total_categories', 'N/A')],
            ['Models Used', ', '.join(summary_metrics.get('models_used', []))],
            ['Average WMAPE', f"{summary_metrics.get('avg_wmape', 0):.3f}"],
            ['Best WMAPE', f"{summary_metrics.get('best_wmape', 0):.3f}"],
            ['Worst WMAPE', f"{summary_metrics.get('worst_wmape', 0):.3f}"]
        ]
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Metric', 'Value'],
                fill_color='lightblue',
                align='center',
                font=dict(size=14, color='white'),
                height=40
            ),
            cells=dict(
                values=[[row[0] for row in metrics_data], 
                       [row[1] for row in metrics_data]],
                fill_color='white',
                align='left',
                font=dict(size=12),
                height=35
            )
        )])
        
        fig.update_layout(
            title="Forecasting Summary",
            height=300,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        return fig
    
    def create_category_performance_plot(self, param_df: pd.DataFrame) -> go.Figure:
        """
        Create a bar plot showing performance by category
        
        Args:
            param_df: DataFrame with category performance data
            
        Returns:
            Plotly figure object
        """
        fig = px.bar(
            param_df.sort_values('cv_avg_metric'),
            x='category',
            y='cv_avg_metric',
            color='model_name',
            title='Forecast Performance by Category',
            labels={'cv_avg_metric': 'WMAPE', 'category': 'Category'},
            template='plotly_white'
        )
        
        fig.update_layout(
            xaxis_title='Category',
            yaxis_title='WMAPE (Lower is Better)',
            height=400,
            showlegend=True
        )
        
        return fig
    
    def create_time_series_subplots(self, data: pd.DataFrame, max_categories: int = 6, last_n_periods: int = 18) -> go.Figure:
        """
        Create subplot grid showing individual time series for each category
        
        Args:
            data: Output data from forecasting pipeline
            max_categories: Maximum number of categories to display
            last_n_periods: Number of recent periods to show per category
            
        Returns:
            Plotly figure with subplots
        """
        categories = data['category'].unique()[:max_categories]
        n_categories = len(categories)
        
        # Calculate subplot grid
        n_cols = min(2, n_categories)
        n_rows = (n_categories + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=[f"{cat}" for cat in categories],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        for i, category in enumerate(categories):
            row = (i // n_cols) + 1
            col = (i % n_cols) + 1
            
            cat_data = data[data['category'] == category].copy()
            cat_data['date'] = pd.to_datetime(cat_data['date'])
            cat_data = cat_data.sort_values('date')
            
            # Filter to show only the last N unique dates
            unique_dates = sorted(cat_data['date'].unique())
            if len(unique_dates) > last_n_periods:
                last_n_dates = unique_dates[-last_n_periods:]
                cat_data = cat_data[cat_data['date'].isin(last_n_dates)].copy()
            
            # Actual values (deduplicated)
            actual_data = cat_data[~cat_data['true_vals'].isna()].drop_duplicates(subset=['date'])
            if not actual_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=actual_data['date'],
                        y=actual_data['true_vals'],
                        mode='lines',
                        name='Actual',
                        line=dict(color=self.colors['actual'], width=1.5),
                        showlegend=(i == 0)
                    ),
                    row=row, col=col
                )
            
            # Test predictions for each model
            models = cat_data['model_name'].unique()
            model_colors = px.colors.qualitative.Set1[:len(models)]
            
            for j, model in enumerate(models):
                model_data = cat_data[cat_data['model_name'] == model]
                test_data = model_data[~model_data['eval_test_preds'].isna()]
                
                if not test_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=test_data['date'],
                            y=test_data['eval_test_preds'],
                            mode='lines',
                            name=f'{model} - Test',
                            line=dict(color=model_colors[j % len(model_colors)], width=1.5, dash='dash'),
                            showlegend=(i == 0)  # Only show legend for first category
                        ),
                        row=row, col=col
                    )
            
            # Future forecasts
            forecast_data = cat_data[~cat_data['forecast'].isna() & cat_data['true_vals'].isna()]
            if not forecast_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=forecast_data['date'],
                        y=forecast_data['forecast'],
                        mode='lines',
                        name='Future Forecast',
                        line=dict(color=self.colors['forecast'], width=1.5),
                        showlegend=(i == 0)
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title='Time Series Forecasts by Category',
            height=200 * n_rows + 100,
            template='plotly_white',
            showlegend=True
        )
        
        return fig
    
    def create_error_distribution_plot(self, data: pd.DataFrame) -> go.Figure:
        """
        Create histogram showing distribution of forecast errors
        
        Args:
            data: DataFrame with actual and predicted values
            
        Returns:
            Plotly figure object
        """
        # Calculate errors where both actual and predicted are available
        test_data = data[~data['true_vals'].isna() & ~data['eval_test_preds'].isna()].copy()
        
        if test_data.empty:
            # Create empty plot
            fig = go.Figure()
            fig.add_annotation(
                text="No test data available for error analysis",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title="Forecast Error Distribution",
                height=400
            )
            return fig
        
        # Calculate percentage errors
        test_data['error_pct'] = ((test_data['eval_test_preds'] - test_data['true_vals']) / 
                                 test_data['true_vals'] * 100)
        
        # Remove infinite values
        test_data = test_data[np.isfinite(test_data['error_pct'])]
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=test_data['error_pct'],
            nbinsx=30,
            name='Forecast Errors',
            marker=dict(
                color=self.colors['forecast'],
                opacity=0.7,
                line=dict(color='white', width=1)
            )
        ))
        
        # Add vertical line at zero
        fig.add_vline(
            x=0, 
            line_dash="dash", 
            line_color="red",
            annotation_text="Perfect Forecast"
        )
        
        fig.update_layout(
            title='Distribution of Forecast Errors (%)',
            xaxis_title='Forecast Error (%)',
            yaxis_title='Frequency',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_matplotlib_summary_plot(summary_metrics: Dict[str, Any]) -> plt.Figure:
        """
        Create a matplotlib summary plot for static display
        
        Args:
            summary_metrics: Summary metrics dictionary
            
        Returns:
            Matplotlib figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('TEMPO Forecasting Results Summary', fontsize=16, fontweight='bold')
        
        # Model performance
        model_perf = summary_metrics.get('model_performance', {})
        if model_perf:
            models = list(model_perf.keys())
            wmapes = [model_perf[m]['avg_wmape'] for m in models]
            
            bars = ax1.bar(models, wmapes, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(models)])
            ax1.set_title('Average WMAPE by Model')
            ax1.set_ylabel('WMAPE')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        # Summary statistics
        stats_labels = ['Total Categories', 'Avg WMAPE', 'Best WMAPE', 'Worst WMAPE']
        stats_values = [
            summary_metrics.get('total_categories', 0),
            summary_metrics.get('avg_wmape', 0),
            summary_metrics.get('best_wmape', 0),
            summary_metrics.get('worst_wmape', 0)
        ]
        
        ax2.axis('off')
        table_data = []
        for label, value in zip(stats_labels, stats_values):
            if isinstance(value, float):
                table_data.append([label, f"{value:.3f}"])
            else:
                table_data.append([label, str(value)])
        
        table = ax2.table(cellText=table_data,
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax2.set_title('Key Metrics')
        
        # Models used
        models_used = summary_metrics.get('models_used', [])
        if models_used:
            ax3.pie([1] * len(models_used), labels=models_used, autopct='',
                   colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(models_used)])
            ax3.set_title('Models Used')
        
        # Performance range
        if summary_metrics.get('best_wmape') and summary_metrics.get('worst_wmape'):
            wmape_range = [summary_metrics['best_wmape'], summary_metrics['worst_wmape']]
            ax4.bar(['Best', 'Worst'], wmape_range, color=['green', 'red'], alpha=0.7)
            ax4.set_title('WMAPE Range')
            ax4.set_ylabel('WMAPE')
        
        plt.tight_layout()
        return fig


# Convenience functions for quick plotting
def quick_forecast_plot(data: pd.DataFrame, category: str = None, last_n_periods: int = 18) -> go.Figure:
    """Quick function to create a forecast comparison plot"""
    viz = DemoVisualizer()
    return viz.create_forecast_comparison_plot(data, category, last_n_periods=last_n_periods)


def quick_performance_plot(comparison_data: pd.DataFrame) -> go.Figure:
    """Quick function to create a model performance plot"""
    viz = DemoVisualizer()
    return viz.create_model_performance_plot(comparison_data)


if __name__ == "__main__":
    # Demo usage
    print("Demo visualization module loaded successfully")
    print("Available functions:")
    print("- DemoVisualizer class with methods for interactive plots")
    print("- quick_forecast_plot() for quick time series plots")
    print("- quick_performance_plot() for quick model comparison plots")