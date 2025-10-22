"""
Synthetic Data Generator for TEMPO Demo Application

This module creates realistic time series data for demonstrating the TEMPO forecasting library.
It generates various scenarios with different patterns and characteristics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import random


class SyntheticDataGenerator:
    """Generate synthetic time series data with various realistic patterns"""
    
    def __init__(self, seed: int = 42):
        """Initialize the generator with a random seed for reproducibility"""
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_scenario(self, scenario: str, **kwargs) -> pd.DataFrame:
        """
        Generate data for a specific scenario
        
        Args:
            scenario: One of 'retail_sales', 'equipment_rental', 'saas_metrics', 'energy_consumption'
            **kwargs: Scenario-specific parameters
            
        Returns:
            DataFrame with columns: date, category, target_value
        """
        scenarios = {
            'retail_sales': self._generate_retail_sales,
            'equipment_rental': self._generate_equipment_rental, 
            'saas_metrics': self._generate_saas_metrics,
            'energy_consumption': self._generate_energy_consumption
        }
        
        if scenario not in scenarios:
            raise ValueError(f"Unknown scenario: {scenario}. Available: {list(scenarios.keys())}")
            
        return scenarios[scenario](**kwargs)
    
    def _generate_retail_sales(self, 
                              start_date: str = "2020-01-01",
                              end_date: str = "2024-12-31", 
                              categories: List[str] = None,
                              base_sales_range: Tuple[int, int] = (50, 500)) -> pd.DataFrame:
        """Generate retail sales data with seasonal patterns and promotions"""
        
        if categories is None:
            categories = ["Electronics", "Clothing", "Home & Garden", "Sports", "Books"]
            
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        data = []
        
        for category in categories:
            base_sales = np.random.randint(*base_sales_range)
            
            for date in date_range:
                # Base trend with slight growth
                days_from_start = (date - date_range[0]).days
                trend = 1 + (days_from_start / len(date_range)) * 0.2  # 20% growth over period
                
                # Seasonal patterns
                seasonal = self._get_seasonal_multiplier(date, category)
                
                # Weekly patterns (higher on weekends for some categories)
                weekly = self._get_weekly_multiplier(date, category)
                
                # Random noise
                noise = np.random.normal(1, 0.15)
                
                # Special events (holidays, promotions)
                event_multiplier = self._get_event_multiplier(date, category)
                
                sales = base_sales * trend * seasonal * weekly * noise * event_multiplier
                sales = max(0, int(sales))  # Ensure non-negative
                
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'category': category,
                    'n_rented': sales  # Using same column name as TEMPO examples
                })
        
        return pd.DataFrame(data)
    
    def _generate_equipment_rental(self,
                                 start_date: str = "2020-01-01", 
                                 end_date: str = "2024-12-31",
                                 categories: List[str] = None,
                                 base_rental_range: Tuple[int, int] = (20, 200)) -> pd.DataFrame:
        """Generate equipment rental data with business day patterns"""
        
        if categories is None:
            categories = ["Construction Tools", "Party Equipment", "Lawn & Garden", "Moving Equipment", "Professional Audio"]
            
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        data = []
        
        for category in categories:
            base_rental = np.random.randint(*base_rental_range)
            
            for date in date_range:
                # Business day patterns (most equipment rented on business days)
                if date.weekday() < 5:  # Monday-Friday
                    weekly_multiplier = 1.2
                elif date.weekday() == 5:  # Saturday
                    weekly_multiplier = 0.8
                else:  # Sunday
                    weekly_multiplier = 0.4
                    
                # Seasonal patterns (higher in spring/summer for most equipment)
                month = date.month
                if category in ["Construction Tools", "Lawn & Garden"]:
                    seasonal = 1.3 if month in [4, 5, 6, 7, 8] else 0.8
                elif category == "Party Equipment":
                    seasonal = 1.4 if month in [5, 6, 7, 8, 12] else 0.9
                else:
                    seasonal = 1.0
                    
                # Random noise
                noise = np.random.normal(1, 0.2)
                
                rentals = base_rental * weekly_multiplier * seasonal * noise
                rentals = max(0, int(rentals))
                
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'category': category,
                    'n_rented': rentals
                })
                
        return pd.DataFrame(data)
    
    def _generate_saas_metrics(self,
                              start_date: str = "2020-01-01",
                              end_date: str = "2024-12-31", 
                              categories: List[str] = None,
                              base_usage_range: Tuple[int, int] = (100, 1000)) -> pd.DataFrame:
        """Generate SaaS usage metrics with growth trends"""
        
        if categories is None:
            categories = ["API Calls", "File Storage", "User Sessions", "Data Processing", "Email Sending"]
            
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        data = []
        
        for category in categories:
            base_usage = np.random.randint(*base_usage_range)
            
            for date in date_range:
                # Growth trend (SaaS typically grows over time)
                days_from_start = (date - date_range[0]).days
                growth_rate = np.random.uniform(0.3, 0.8)  # 30-80% growth
                trend = 1 + (days_from_start / len(date_range)) * growth_rate
                
                # Business day patterns (higher usage on weekdays)
                if date.weekday() < 5:
                    weekly_multiplier = 1.1
                else:
                    weekly_multiplier = 0.7
                    
                # Monthly patterns (often lower at month-end due to budget cycles)
                if date.day > 25:
                    monthly_multiplier = 0.9
                else:
                    monthly_multiplier = 1.0
                    
                # Random noise
                noise = np.random.normal(1, 0.25)
                
                usage = base_usage * trend * weekly_multiplier * monthly_multiplier * noise
                usage = max(0, int(usage))
                
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'category': category,
                    'n_rented': usage
                })
                
        return pd.DataFrame(data)
    
    def _generate_energy_consumption(self,
                                   start_date: str = "2020-01-01",
                                   end_date: str = "2024-12-31",
                                   categories: List[str] = None, 
                                   base_consumption_range: Tuple[int, int] = (200, 800)) -> pd.DataFrame:
        """Generate energy consumption data with seasonal and daily patterns"""
        
        if categories is None:
            categories = ["Residential", "Commercial", "Industrial", "Data Center", "Municipal"]
            
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        data = []
        
        for category in categories:
            base_consumption = np.random.randint(*base_consumption_range)
            
            for date in date_range:
                # Seasonal patterns (higher in summer/winter for HVAC)
                month = date.month
                if category == "Residential":
                    seasonal = 1.4 if month in [12, 1, 2, 6, 7, 8] else 0.9
                elif category == "Commercial": 
                    seasonal = 1.2 if month in [6, 7, 8] else 0.95
                else:
                    seasonal = 1.0
                    
                # Weekly patterns (lower on weekends for commercial)
                if category in ["Commercial", "Industrial"] and date.weekday() >= 5:
                    weekly_multiplier = 0.7
                else:
                    weekly_multiplier = 1.0
                    
                # Random noise
                noise = np.random.normal(1, 0.15)
                
                consumption = base_consumption * seasonal * weekly_multiplier * noise
                consumption = max(0, int(consumption))
                
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'category': category,
                    'n_rented': consumption
                })
                
        return pd.DataFrame(data)
    
    def _get_seasonal_multiplier(self, date: pd.Timestamp, category: str) -> float:
        """Get seasonal multiplier based on category and date"""
        month = date.month
        
        seasonal_patterns = {
            "Electronics": {12: 1.8, 11: 1.4, 1: 0.8, 2: 0.7},  # Holiday shopping
            "Clothing": {3: 1.3, 4: 1.4, 9: 1.3, 10: 1.2},     # Season changes
            "Home & Garden": {4: 1.5, 5: 1.6, 6: 1.3, 7: 1.2}, # Spring/summer
            "Sports": {6: 1.3, 7: 1.4, 8: 1.3, 12: 1.2},       # Summer + winter sports
            "Books": {9: 1.3, 1: 1.2, 8: 0.8}                   # Back to school + new year
        }
        
        return seasonal_patterns.get(category, {}).get(month, 1.0)
    
    def _get_weekly_multiplier(self, date: pd.Timestamp, category: str) -> float:
        """Get weekly pattern multiplier"""
        weekday = date.weekday()  # 0=Monday, 6=Sunday
        
        if category in ["Electronics", "Clothing"]:
            # Higher on weekends
            return 1.3 if weekday >= 5 else 1.0
        else:
            # Fairly consistent
            return 1.0
    
    def _get_event_multiplier(self, date: pd.Timestamp, category: str) -> float:
        """Get special event multiplier (holidays, promotions)"""
        month, day = date.month, date.day
        
        # Black Friday (last Friday of November)
        if month == 11 and date.weekday() == 4 and day >= 22:
            return 2.5 if category == "Electronics" else 1.8
            
        # Christmas week
        if month == 12 and day >= 20:
            return 1.6
            
        # Valentine's Day
        if month == 2 and day == 14:
            return 1.4 if category == "Electronics" else 1.0
            
        # Random promotional events (5% chance on any day)
        if np.random.random() < 0.05:
            return np.random.uniform(1.2, 1.6)
            
        return 1.0
    
    def get_available_scenarios(self) -> Dict[str, str]:
        """Return available scenarios with descriptions"""
        return {
            'retail_sales': 'Retail sales with seasonal patterns, holidays, and promotions',
            'equipment_rental': 'Equipment rental with business day patterns and seasonal trends', 
            'saas_metrics': 'SaaS usage metrics with growth trends and business cycles',
            'energy_consumption': 'Energy consumption with seasonal and daily usage patterns'
        }
    
    def generate_demo_dataset(self, scenario: str = 'retail_sales', **kwargs) -> pd.DataFrame:
        """Generate a demo dataset with default parameters optimized for demonstration"""
        default_params = {
            'retail_sales': {
                'start_date': '2021-01-01',
                'end_date': '2024-12-31', 
                'categories': ['Product A', 'Product B', 'Product C'],
                'base_sales_range': (100, 300)
            },
            'equipment_rental': {
                'start_date': '2021-01-01',
                'end_date': '2024-12-31',
                'categories': ['Equipment A', 'Equipment B', 'Equipment C'],
                'base_rental_range': (50, 150)
            },
            'saas_metrics': {
                'start_date': '2021-01-01', 
                'end_date': '2024-12-31',
                'categories': ['Service A', 'Service B', 'Service C'],
                'base_usage_range': (200, 500)
            },
            'energy_consumption': {
                'start_date': '2021-01-01',
                'end_date': '2024-12-31', 
                'categories': ['Zone A', 'Zone B', 'Zone C'],
                'base_consumption_range': (300, 600)
            }
        }
        
        # Merge default params with user provided params
        params = default_params.get(scenario, {})
        params.update(kwargs)
        
        return self.generate_scenario(scenario, **params)


if __name__ == "__main__":
    # Demo usage
    generator = SyntheticDataGenerator()
    
    # Generate sample data
    retail_data = generator.generate_demo_dataset('retail_sales')
    print("Generated retail sales data:")
    print(retail_data.head(10))
    print(f"Shape: {retail_data.shape}")
    print(f"Date range: {retail_data['date'].min()} to {retail_data['date'].max()}")
    print(f"Categories: {retail_data['category'].unique()}")