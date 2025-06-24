"""
German Energy Load Analysis
This script provides a comprehensive analysis of German electricity load data from 2015 to 2020.
We'll explore various patterns and trends in energy consumption across different time periods and conditions.
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# Set plot style
plt.style.use('default')  # Changed from 'seaborn' to 'default'
sns.set_theme()  # This will set seaborn's default theme

# Custom color palette
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'tertiary': '#2ca02c',
    'background': '#f8f9fa',
    'text': '#2c3e50',
    'success': '#2ecc71',
    'warning': '#f1c40f',
    'danger': '#e74c3c'
}

def load_and_preprocess_data():
    """Load and preprocess the energy load data."""
    print("Loading and preprocessing data...")
    
    try:
        # Load the data
        url = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"
        df = pd.read_csv(url, index_col=0, parse_dates=True)
        df = df[['DE_load_actual_entsoe_transparency']].dropna()
        df.rename(columns={'DE_load_actual_entsoe_transparency': 'load_MW'}, inplace=True)
        df.index = pd.to_datetime(df.index)
        
        print("Data Shape:", df.shape)
        print("\nFirst few rows of the data:")
        print(df.head())
        
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def add_features(df):
    """Add time-based features to the dataset."""
    print("\nAdding time-based features...")
    
    try:
        # Add time-based features
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['quarter'] = df.index.quarter
        df['season'] = df['month'].map({12:'Winter', 1:'Winter', 2:'Winter',
                                       3:'Spring', 4:'Spring', 5:'Spring',
                                       6:'Summer', 7:'Summer', 8:'Summer',
                                       9:'Autumn', 10:'Autumn', 11:'Autumn'})
        df['date'] = df.index.date
        df['day_name'] = df.index.day_name()
        
        # Calculate daily statistics
        daily_stats = df.groupby('date').agg({
            'load_MW': ['mean', 'max', 'min', 'std'],
            'hour': 'count'
        }).reset_index()
        daily_stats.columns = ['date', 'mean_load', 'max_load', 'min_load', 'std_load', 'hour_count']
        daily_stats['date'] = pd.to_datetime(daily_stats['date'])
        daily_stats['year'] = daily_stats['date'].dt.year
        daily_stats['month'] = daily_stats['date'].dt.month
        daily_stats['season'] = daily_stats['month'].map({12:'Winter', 1:'Winter', 2:'Winter',
                                                         3:'Spring', 4:'Spring', 5:'Spring',
                                                         6:'Summer', 7:'Summer', 8:'Summer',
                                                         9:'Autumn', 10:'Autumn', 11:'Autumn'})
        
        print("\nDaily Statistics Shape:", daily_stats.shape)
        print("\nFirst few rows of daily statistics:")
        print(daily_stats.head())
        
        return df, daily_stats
    except Exception as e:
        print(f"Error adding features: {str(e)}")
        raise

def basic_statistics(df):
    """Calculate and display basic statistics."""
    print("\nCalculating basic statistics...")
    
    try:
        # Calculate basic statistics
        stats = df['load_MW'].describe()
        print("\nBasic Statistics:")
        print(stats)
        
        # Plot distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df, x='load_MW', bins=50)
        plt.title('Distribution of Load Values')
        plt.xlabel('Load (MW)')
        plt.savefig('output/load_distribution.png')
        plt.close()
    except Exception as e:
        print(f"Error in basic statistics: {str(e)}")
        raise

def time_series_analysis(df):
    """Perform time series analysis."""
    print("\nPerforming time series analysis...")
    
    try:
        # Plot time series
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['load_MW'],
                                mode='lines',
                                name='Load'))
        fig.update_layout(title='German Electricity Load Over Time',
                         xaxis_title='Date',
                         yaxis_title='Load (MW)',
                         template='plotly_white')
        fig.write_html('output/time_series.html')
    except Exception as e:
        print(f"Error in time series analysis: {str(e)}")
        raise

def daily_patterns_analysis(df):
    """Analyze daily load patterns."""
    print("\nAnalyzing daily patterns...")
    
    try:
        # Calculate daily patterns
        daily_pattern = df.groupby('hour')['load_MW'].agg(['mean', 'std']).reset_index()
        
        # Plot daily patterns with confidence intervals
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_pattern['hour'],
            y=daily_pattern['mean'],
            mode='lines',
            name='Average Load',
            line=dict(color=COLORS['primary'], width=3)
        ))
        fig.add_trace(go.Scatter(
            x=daily_pattern['hour'],
            y=daily_pattern['mean'] + daily_pattern['std'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=daily_pattern['hour'],
            y=daily_pattern['mean'] - daily_pattern['std'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(31, 119, 180, 0.2)',
            name='Standard Deviation'
        ))
        fig.update_layout(
            title='Daily Load Pattern with Variability',
            xaxis_title='Hour of Day',
            yaxis_title='Load (MW)',
            template='plotly_white'
        )
        fig.write_html('output/daily_patterns.html')
    except Exception as e:
        print(f"Error in daily patterns analysis: {str(e)}")
        raise

def weekly_patterns_analysis(df):
    """Analyze weekly load patterns."""
    print("\nAnalyzing weekly patterns...")
    
    try:
        # Calculate weekly patterns
        weekly_pattern = df.groupby(['dayofweek', 'is_weekend'])['load_MW'].mean().reset_index()
        
        # Plot weekly patterns
        fig = px.bar(weekly_pattern, x='dayofweek', y='load_MW', color='is_weekend',
                     title='Weekly Load Pattern (Weekend vs Weekday)',
                     labels={'dayofweek': 'Day of Week', 'load_MW': 'Average Load (MW)',
                            'is_weekend': 'Weekend'},
                     color_discrete_sequence=[COLORS['primary'], COLORS['secondary']])
        fig.update_layout(
            xaxis=dict(ticktext=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                      tickvals=[0, 1, 2, 3, 4, 5, 6]),
            template='plotly_white'
        )
        fig.write_html('output/weekly_patterns.html')
    except Exception as e:
        print(f"Error in weekly patterns analysis: {str(e)}")
        raise

def seasonal_analysis(df):
    """Analyze seasonal patterns."""
    print("\nAnalyzing seasonal patterns...")
    
    try:
        # Calculate seasonal patterns
        seasonal_pattern = df.groupby(['season', 'hour'])['load_MW'].mean().reset_index()
        
        # Plot seasonal patterns
        fig = px.line(seasonal_pattern, x='hour', y='load_MW', color='season',
                      title='Seasonal Daily Load Patterns',
                      labels={'hour': 'Hour of Day', 'load_MW': 'Average Load (MW)',
                             'season': 'Season'})
        fig.update_layout(template='plotly_white')
        fig.write_html('output/seasonal_patterns.html')
        
        # Monthly heatmap
        monthly_heatmap = df.pivot_table(
            values='load_MW',
            index='hour',
            columns='month',
            aggfunc='mean'
        )
        
        fig = px.imshow(monthly_heatmap,
                        title='Load Distribution by Hour and Month',
                        labels=dict(x='Month', y='Hour', color='Load (MW)'),
                        color_continuous_scale='Viridis',
                        width=5000,
                        height=1000)
        fig.update_layout(template='plotly_white')
        fig.write_html('output/monthly_heatmap.html')
    except Exception as e:
        print(f"Error in seasonal analysis: {str(e)}")
        raise

def performance_analysis(daily_stats):
    """Analyze seasonal performance."""
    print("\nAnalyzing seasonal performance...")
    
    try:
        # Calculate seasonal performance
        seasonal_performance = []
        for year in daily_stats['year'].unique():
            for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
                season_data = daily_stats[
                    (daily_stats['year'] == year) & 
                    (daily_stats['season'] == season)
                ]
                
                if not season_data.empty:
                    best_day = season_data.loc[season_data['mean_load'].idxmax()]
                    worst_day = season_data.loc[season_data['mean_load'].idxmin()]
                    
                    seasonal_performance.append({
                        'Year': year,
                        'Season': season,
                        'Best Day': best_day['date'].strftime('%Y-%m-%d'),
                        'Best Day Load': best_day['mean_load'],
                        'Worst Day': worst_day['date'].strftime('%Y-%m-%d'),
                        'Worst Day Load': worst_day['mean_load'],
                        'Seasonal Average': season_data['mean_load'].mean(),
                        'Seasonal Std': season_data['mean_load'].std()
                    })
        
        performance_df = pd.DataFrame(seasonal_performance)
        print("\nSeasonal Performance Summary:")
        print(performance_df)
        
        # Plot seasonal performance trends
        fig = go.Figure()
        
        for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
            season_data = performance_df[performance_df['Season'] == season]
            fig.add_trace(go.Scatter(
                x=season_data['Year'],
                y=season_data['Seasonal Average'],
                name=season,
                mode='lines+markers',
                error_y=dict(
                    type='data',
                    array=season_data['Seasonal Std'],
                    visible=True
                )
            ))
        
        fig.update_layout(
            title='Seasonal Performance Trends with Variability',
            xaxis_title='Year',
            yaxis_title='Average Load (MW)',
            template='plotly_white',
            showlegend=True
        )
        fig.write_html('output/seasonal_performance.html')
        
        return performance_df
    except Exception as e:
        print(f"Error in performance analysis: {str(e)}")
        raise

def generate_insights(df, performance_df):
    """Generate key insights and recommendations."""
    print("\nGenerating insights and recommendations...")
    
    try:
        # Calculate key metrics
        peak_hours = df.groupby('hour')['load_MW'].mean().nlargest(3)
        off_peak_hours = df.groupby('hour')['load_MW'].mean().nsmallest(3)
        seasonal_means = df.groupby('season')['load_MW'].mean()
        seasonal_std = df.groupby('season')['load_MW'].std()
        
        # Calculate year-over-year growth
        yearly_avg = df.groupby('year')['load_MW'].mean()
        yoy_growth = ((yearly_avg.iloc[-1] - yearly_avg.iloc[0]) / yearly_avg.iloc[0]) * 100
        
        print("\nKey Insights:")
        print(f"1. Overall Growth Rate: {yoy_growth:.1f}%")
        print(f"2. Peak Hours: {', '.join(map(str, peak_hours.index))}")
        print(f"3. Off-Peak Hours: {', '.join(map(str, off_peak_hours.index))}")
        print(f"4. Highest Average Load Season: {seasonal_means.idxmax()}")
        print(f"5. Lowest Average Load Season: {seasonal_means.idxmin()}")
        print(f"6. Most Stable Season: {seasonal_std.idxmin()}")
        print(f"7. Most Variable Season: {seasonal_std.idxmax()}")
        
        print("\nStrategic Recommendations:")
        print("1. Load Management:")
        print("   - Implement demand response programs during peak hours")
        print("   - Consider time-of-use pricing to shift demand")
        print("\n2. Infrastructure Planning:")
        print("   - Focus on capacity expansion during peak seasons")
        print("   - Optimize maintenance during low-demand periods")
        print("\n3. Energy Storage:")
        print("   - Invest in storage solutions for peak demand")
        print("   - Utilize storage during high variability periods")
        print("\n4. Renewable Integration:")
        print("   - Align renewable generation with peak patterns")
        print("   - Develop flexible generation capacity")
    except Exception as e:
        print(f"Error generating insights: {str(e)}")
        raise

def main():
    """Main function to run the analysis."""
    print("Starting German Energy Load Analysis...")
    
    try:
        # Load and preprocess data
        df = load_and_preprocess_data()
        
        # Add features
        df, daily_stats = add_features(df)
        
        # Perform analyses
        basic_statistics(df)
        time_series_analysis(df)
        daily_patterns_analysis(df)
        weekly_patterns_analysis(df)
        seasonal_analysis(df)
        performance_df = performance_analysis(daily_stats)
        
        # Generate insights
        generate_insights(df, performance_df)
        
        print("\nAnalysis complete! Check the 'output' directory for generated visualizations.")
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 