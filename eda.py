import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="German Energy Load Analysis Dashboard",
    page_icon="⚡",
    layout="wide"
)

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

# Title and description
st.title("⚡ German Energy Load Analysis Dashboard")
st.markdown("""
This dashboard provides a comprehensive analysis of German electricity load data from 2015 to 2020.
Explore various patterns and trends in energy consumption across different time periods and conditions.
""")

# Load and preprocess data
@st.cache_data
def load_data():
    url = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"
    df = pd.read_csv(url, index_col=0, parse_dates=True)
    df = df[['DE_load_actual_entsoe_transparency']].dropna()
    df.rename(columns={'DE_load_actual_entsoe_transparency': 'load_MW'}, inplace=True)
    df.index = pd.to_datetime(df.index)
    return df

# Load the data
df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=2015,
    max_value=2020,
    value=(2015, 2020)
)

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

# Filter data based on year range
mask = (df['year'] >= year_range[0]) & (df['year'] <= year_range[1])
filtered_df = df[mask]
filtered_daily_stats = daily_stats[(daily_stats['year'] >= year_range[0]) & 
                                 (daily_stats['year'] <= year_range[1])]

# Create tabs for different analyses
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Executive Summary", "Time Patterns", "Seasonal Analysis", 
    "Statistical Analysis", "Business Insights", "Performance Analysis"
])

with tab1:
    st.header("Executive Summary")
    
    # Key Performance Indicators
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_load = filtered_df['load_MW'].mean()
        st.metric("Average Load", f"{avg_load:,.0f} MW")
    with col2:
        max_load = filtered_df['load_MW'].max()
        st.metric("Peak Load", f"{max_load:,.0f} MW")
    with col3:
        min_load = filtered_df['load_MW'].min()
        st.metric("Minimum Load", f"{min_load:,.0f} MW")
    with col4:
        load_std = filtered_df['load_MW'].std()
        st.metric("Load Variability", f"{load_std:,.0f} MW")
    
    # Year-over-Year Growth
    yearly_avg = df.groupby('year')['load_MW'].mean()
    yoy_growth = ((yearly_avg.iloc[-1] - yearly_avg.iloc[0]) / yearly_avg.iloc[0]) * 100
    
    st.subheader("Year-over-Year Growth Analysis")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=yearly_avg.index,
        y=yearly_avg.values,
        marker_color=COLORS['primary'],
        name='Average Load'
    ))
    fig.update_layout(
        title='Annual Average Load Trend',
        xaxis_title='Year',
        yaxis_title='Average Load (MW)',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(f"""
    📈 **Key Growth Metrics:**
    - Overall growth rate: {yoy_growth:.1f}%
    - Highest annual average: {yearly_avg.max():,.0f} MW ({yearly_avg.idxmax()})
    - Lowest annual average: {yearly_avg.min():,.0f} MW ({yearly_avg.idxmin()})
    """)

with tab2:
    st.header("Time-based Patterns")
    
    # Daily patterns with confidence intervals
    st.subheader("Daily Load Patterns")
    daily_stats = filtered_df.groupby('hour')['load_MW'].agg(['mean', 'std']).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_stats['hour'],
        y=daily_stats['mean'],
        mode='lines',
        name='Average Load',
        line=dict(color=COLORS['primary'], width=3)
    ))
    fig.add_trace(go.Scatter(
        x=daily_stats['hour'],
        y=daily_stats['mean'] + daily_stats['std'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=daily_stats['hour'],
        y=daily_stats['mean'] - daily_stats['std'],
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
    st.plotly_chart(fig, use_container_width=True)
    
    # Weekly patterns with weekend highlight
    st.subheader("Weekly Load Patterns")
    weekly_pattern = filtered_df.groupby(['dayofweek', 'is_weekend'])['load_MW'].mean().reset_index()
    
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
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Seasonal Analysis")
    
    # Seasonal patterns
    st.subheader("Seasonal Load Patterns")
    seasonal_pattern = filtered_df.groupby(['season', 'hour'])['load_MW'].mean().reset_index()
    
    fig = px.line(seasonal_pattern, x='hour', y='load_MW', color='season',
                  title='Seasonal Daily Load Patterns',
                  labels={'hour': 'Hour of Day', 'load_MW': 'Average Load (MW)',
                         'season': 'Season'})
    fig.update_layout(template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly heatmap
    st.subheader("Monthly Load Heatmap")
    monthly_heatmap = filtered_df.pivot_table(
        values='load_MW',
        index='hour',
        columns='month',
        aggfunc='mean'
    )
    
    fig = px.imshow(monthly_heatmap,
                    title='Load Distribution by Hour and Month',
                    labels=dict(x='Month', y='Hour', color='Load (MW)'),
                    color_continuous_scale='Viridis',
                    width=5000,  # Increased width
                    height=1000)  # Increased height
    fig.update_layout(template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Statistical Analysis")
    
    # Distribution analysis with KDE
    st.subheader("Load Distribution Analysis")
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=filtered_df['load_MW'],
        name='Load Distribution',
        nbinsx=50,
        marker_color=COLORS['primary']
    ))
    fig.update_layout(
        title='Load Distribution with Kernel Density Estimation',
        xaxis_title='Load (MW)',
        yaxis_title='Frequency',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Box plot by season
    st.subheader("Seasonal Load Distribution")
    fig = px.box(filtered_df, x='season', y='load_MW',
                 title='Load Distribution by Season',
                 color='season',
                 labels={'season': 'Season', 'load_MW': 'Load (MW)'})
    fig.update_layout(template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.header("Business Insights")
    
    # Peak load analysis
    peak_hours = filtered_df.groupby('hour')['load_MW'].mean().nlargest(3)
    off_peak_hours = filtered_df.groupby('hour')['load_MW'].mean().nsmallest(3)
    
    st.subheader("Peak Load Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Peak Hours", f"{', '.join(map(str, peak_hours.index))}")
        st.metric("Average Peak Load", f"{peak_hours.mean():,.0f} MW")
    with col2:
        st.metric("Off-Peak Hours", f"{', '.join(map(str, off_peak_hours.index))}")
        st.metric("Average Off-Peak Load", f"{off_peak_hours.mean():,.0f} MW")
    
    # Load variability analysis
    st.subheader("Load Variability Analysis")
    hourly_variability = filtered_df.groupby('hour')['load_MW'].std()
    max_variability_hour = hourly_variability.idxmax()
    
    st.info(f"""
    🔍 **Key Insights:**
    1. **Peak Demand Hours:**
       - Highest demand occurs during hours {', '.join(map(str, peak_hours.index))}
       - Average peak load: {peak_hours.mean():,.0f} MW
    
    2. **Off-Peak Hours:**
       - Lowest demand occurs during hours {', '.join(map(str, off_peak_hours.index))}
       - Average off-peak load: {off_peak_hours.mean():,.0f} MW
    
    3. **Load Variability:**
       - Highest variability occurs at hour {max_variability_hour}
       - Standard deviation: {hourly_variability[max_variability_hour]:,.0f} MW
    
    4. **Seasonal Patterns:**
       - Highest average load: {filtered_df.groupby('season')['load_MW'].mean().idxmax()}
       - Lowest average load: {filtered_df.groupby('season')['load_MW'].mean().idxmin()}
    """)
    
    # Recommendations
    st.subheader("Strategic Recommendations")
    st.markdown("""
    1. **Load Management:**
       - Implement demand response programs during peak hours
       - Consider time-of-use pricing to shift demand to off-peak hours
    
    2. **Infrastructure Planning:**
       - Focus on capacity expansion during identified peak hours
       - Optimize maintenance schedules during off-peak periods
    
    3. **Energy Storage:**
       - Invest in energy storage solutions to manage peak demand
       - Utilize storage during high variability periods
    
    4. **Renewable Integration:**
       - Align renewable generation with peak demand patterns
       - Develop flexible generation capacity for high variability periods
    """)

with tab6:
    st.header("Performance Analysis")
    
    # Seasonal Performance Analysis
    st.subheader("Seasonal Performance Analysis")
    
    # Calculate best and worst days for each season and year
    seasonal_performance = []
    for year in filtered_daily_stats['year'].unique():
        for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
            season_data = filtered_daily_stats[
                (filtered_daily_stats['year'] == year) & 
                (filtered_daily_stats['season'] == season)
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
    
    # Display performance metrics
    st.dataframe(
        performance_df.style.background_gradient(cmap='RdYlGn', subset=['Best Day Load', 'Worst Day Load'])
    )
    
    # Visualize seasonal performance
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
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance Metrics
    st.subheader("Key Performance Metrics")
    
    # Calculate year-over-year growth for each season
    seasonal_growth = []
    for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
        season_data = performance_df[performance_df['Season'] == season]
        if len(season_data) > 1:
            growth = ((season_data['Seasonal Average'].iloc[-1] - 
                      season_data['Seasonal Average'].iloc[0]) / 
                     season_data['Seasonal Average'].iloc[0] * 100)
            seasonal_growth.append({
                'Season': season,
                'Growth Rate': growth
            })
    
    growth_df = pd.DataFrame(seasonal_growth)
    
    # Display growth metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Best Performing Season",
            growth_df.loc[growth_df['Growth Rate'].idxmax(), 'Season'],
            f"{growth_df['Growth Rate'].max():.1f}% growth"
        )
    with col2:
        st.metric(
            "Most Stable Season",
            filtered_df.groupby('season')['load_MW'].std().idxmin(),
            f"Std: {filtered_df.groupby('season')['load_MW'].std().min():,.0f} MW"
        )
    
    # Performance Insights
    st.subheader("Performance Insights")
    st.markdown(f"""
    ### 📊 Key Performance Indicators
    
    1. **Seasonal Trends:**
       - {growth_df.loc[growth_df['Growth Rate'].idxmax(), 'Season']} shows the highest growth rate at {growth_df['Growth Rate'].max():.1f}%
       - {growth_df.loc[growth_df['Growth Rate'].idxmin(), 'Season']} shows the lowest growth rate at {growth_df['Growth Rate'].min():.1f}%
    
    2. **Load Stability:**
       - Most stable season: {filtered_df.groupby('season')['load_MW'].std().idxmin()}
       - Most variable season: {filtered_df.groupby('season')['load_MW'].std().idxmax()}
    
    3. **Peak Performance:**
       - Highest single-day load: {performance_df['Best Day Load'].max():,.0f} MW
       - Lowest single-day load: {performance_df['Worst Day Load'].min():,.0f} MW
    
    4. **Seasonal Patterns:**
       - Highest seasonal average: {performance_df.groupby('Season')['Seasonal Average'].mean().idxmax()}
       - Lowest seasonal average: {performance_df.groupby('Season')['Seasonal Average'].mean().idxmin()}
    """)
    
    # Decision Support
    st.subheader("Data-Driven Decision Support")
    st.markdown("""
    ### 🎯 Strategic Recommendations Based on Performance Analysis
    
    1. **Capacity Planning:**
       - Focus on capacity expansion during {performance_df.groupby('Season')['Seasonal Average'].mean().idxmax()}
       - Optimize maintenance during {performance_df.groupby('Season')['Seasonal Average'].mean().idxmin()}
    
    2. **Resource Allocation:**
       - Allocate more resources during {filtered_df.groupby('season')['load_MW'].std().idxmax()}
       - Optimize resource utilization during {filtered_df.groupby('season')['load_MW'].std().idxmin()}
    
    3. **Risk Management:**
       - Implement additional monitoring during highest variability periods
       - Develop contingency plans for extreme load conditions
    
    4. **Growth Strategy:**
       - Focus on growth opportunities in {growth_df.loc[growth_df['Growth Rate'].idxmax(), 'Season']}
       - Address challenges in {growth_df.loc[growth_df['Growth Rate'].idxmin(), 'Season']}
    """)

# Footer
st.markdown("---")
st.markdown("Data source: Open Power System Data") 