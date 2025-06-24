# ⚡ German Energy Load Forecasting

A comprehensive machine learning project that analyzes and forecasts German electricity load data from 2015-2020 using advanced deep learning models. This project provides insights into energy consumption patterns and delivers accurate 24-hour load predictions for energy grid management.

## 🎯 Project Overview

This project addresses the critical challenge of energy load forecasting in Germany's power grid by:
- **Analyzing 5+ years** of hourly electricity consumption data (2015-2020)
- **Training multiple deep learning models** (LSTM, GRU, Conv1D) for load prediction
- **Providing interactive dashboards** for data exploration and model comparison
- **Generating actionable insights** for energy infrastructure planning

## 📁 Key Features & Files

### Core Analysis Files
- **`energy_analysis.py`** - Comprehensive data analysis script that generates:
  - Daily, weekly, and seasonal load patterns
  - Interactive HTML visualizations for time series analysis
  - Statistical insights and business recommendations
  - Performance metrics across different time periods

- **`energy_forecasting.ipynb`** - Main Jupyter notebook containing:
  - Complete data preprocessing pipeline
  - Feature engineering (time-based, seasonal, holiday features)
  - Model training for LSTM, GRU, and Conv1D architectures
  - Model comparison and evaluation metrics
  - Training time analysis and performance benchmarking

### Interactive Dashboards
- **`eda.py`** - Streamlit dashboard for exploratory data analysis featuring:
  - Executive summary with key performance indicators
  - Interactive time pattern analysis
  - Seasonal performance trends
  - Business insights and recommendations
  - Year-over-year growth analysis

- **`dashboard.py`** - Model comparison dashboard displaying:
  - Performance metrics across all models
  - Prediction vs actual plots
  - Residual analysis visualizations
  - Real-time model evaluation results

### Model Management
- **`model_evaluation.py`** - Automated model evaluation script that:
  - Loads trained models and generates predictions
  - Calculates comprehensive metrics (MSE, RMSE, MAE, R²)
  - Creates prediction and residual visualizations
  - Saves comparison results to CSV format

### Output & Results
- **`output/`** - Generated visualizations and results:
  - Interactive HTML plots for time series, daily patterns, seasonal analysis
  - Model prediction plots and residual analysis
  - Performance metrics CSV with detailed model comparisons
  - Statistical distribution plots

- **`model_checkpoints/`** - Saved trained models:
  - `best_LSTM.h5` - Best performing LSTM model
  - `best_GRU.h5` - Best performing GRU model  
  - `best_Conv1D.h5` - Best performing Conv1D model

## 🚀 Installation & Usage

### Prerequisites
- Python 3.8+
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Energy_forecasting
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the analysis pipeline**
   ```bash
   # Run comprehensive data analysis
   python energy_analysis.py
   
   # Train and evaluate models
   jupyter notebook energy_forecasting.ipynb
   
   # Evaluate model performance
   python model_evaluation.py
   ```

4. **Launch interactive dashboards**
   ```bash
   # Launch EDA dashboard
   streamlit run eda.py
   
   # Launch model comparison dashboard
   streamlit run dashboard.py
   ```

## 📊 Results & Impact

### Key Findings

1. **Model Performance**: GRU achieved the best performance with:
   - **R² Score: 94.2%** (excellent predictive accuracy)
   - **RMSE: 0.052** (low prediction error)
   - **MAE: 0.037** (minimal absolute error)

2. **Energy Consumption Patterns**:
   - **Peak Hours**: 18:00-20:00 (evening demand surge)
   - **Off-Peak Hours**: 02:00-04:00 (lowest consumption)
   - **Weekend Effect**: 15-20% lower consumption compared to weekdays
   - **Seasonal Variation**: Winter shows highest average load, Summer lowest

3. **Infrastructure Insights**:
   - **Growth Trend**: Consistent year-over-year growth in energy demand
   - **Variability**: Highest load variability during winter months
   - **Holiday Impact**: Significant reduction in consumption during holidays

### Business Value

- **Grid Management**: Accurate 24-hour load predictions enable optimal power generation scheduling
- **Cost Optimization**: Demand response programs can be targeted during peak hours
- **Infrastructure Planning**: Seasonal patterns inform capacity expansion decisions
- **Renewable Integration**: Load patterns guide renewable energy deployment strategies

## 🛠 Tech Stack

### Languages & Frameworks
- **Python 3.8+** - Primary programming language
- **Jupyter Notebook** - Interactive development and analysis
- **Streamlit** - Interactive web dashboards

### Data Science & ML Libraries
- **TensorFlow/Keras** - Deep learning model development
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Scikit-learn** - Machine learning utilities and metrics

### Visualization & Analysis
- **Matplotlib** - Static plotting and visualization
- **Seaborn** - Statistical data visualization
- **Plotly** - Interactive HTML visualizations
- **Holidays** - Holiday calendar integration

### Model Architectures
- **LSTM** - Long Short-Term Memory networks for sequence modeling
- **GRU** - Gated Recurrent Units for efficient RNN implementation
- **Conv1D** - 1D Convolutional Neural Networks for time series

## Model Performance Comparison

| Model | MSE | RMSE | MAE | R² Score | Training Time |
|-------|-----|------|-----|----------|---------------|
| **GRU** | 0.0027 | 0.052 | 0.037 | **94.2%** | Fastest |
| **LSTM** | 0.0028 | 0.053 | 0.036 | 94.0% | Medium |
| **Conv1D** | 0.0051 | 0.071 | 0.054 | 89.0% | Fastest |

## Configuration

The project uses several configuration options:
- **Data Source**: Open Power System Data API
- **Time Window**: 168 hours (7 days) input → 24 hours output
- **Train/Val/Test Split**: 70%/15%/15% chronological split
- **Feature Engineering**: Time-based, seasonal, and holiday features
- **Model Checkpointing**: Best models saved based on validation loss

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation for any changes



##  Acknowledgments

- **Data Source**: [Open Power System Data](https://data.open-power-system-data.org/) for providing the German electricity load dataset
- **Research Community**: Contributions from the energy forecasting and time series analysis communities
- **Open Source Tools**: TensorFlow, Pandas, Streamlit, and other open-source libraries that made this project possible



---

**Note**: This project is designed for educational and research purposes. For production deployment, additional considerations for data security, model monitoring, and system reliability should be implemented. 
