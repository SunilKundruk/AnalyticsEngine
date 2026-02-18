# 🤖 AI Data Analysis Platform

A comprehensive **Streamlit-based web application** for intelligent data analysis, visualization, and machine learning with voice capabilities.

## 📋 Features

### 📊 Data Analysis & Visualization
- **Smart Data Cleaning**: Automatic handling of missing values, outliers, and data inconsistencies
- **Exploratory Data Analysis (EDA)**: Generate comprehensive statistical summaries
- **Interactive Visualizations**: Plotly and Matplotlib charts for better insights
- **Data Profiling**: Detailed column statistics and data type analysis

### 🤖 Machine Learning
- **Classification Models**: Random Forest, Logistic Regression, SVM
- **Regression Models**: Random Forest Regressor, Linear Regression
- **Clustering**: DBSCAN and K-Means clustering
- **Anomaly Detection**: Isolation Forest for outlier detection
- **Model Evaluation**: Comprehensive metrics (accuracy, MSE, classification reports)

### 🔊 Voice Features
- **Speech Recognition**: Convert voice input to text
- **Text-to-Speech**: Generate audio output
- **Voice Commands**: Control features through voice

### 📈 Time Series Analysis
- **ARIMA Forecasting**: Statistical time series predictions
- **Seasonal Decomposition**: Understand temporal patterns
- **Trend Analysis**: Visualize time-based trends

### 💾 Data Management
- **CSV Upload**: Support for CSV file uploads
- **Data Export**: Download processed and predictions as CSV
- **Session State Management**: Persistent data across sessions

---

## 🚀 Installation

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Step 1: Clone the Repository
```bash
git clone https://github.com/SunilKundruk/Information-Retrival.git
cd Information-Retrival
```

### Step 2: Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 (Optional): Install Voice Features
For full voice functionality:
```bash
pip install pyaudio speechrecognition pyttsx3
```

### Step 5 (Optional): Install Time Series Features
```bash
pip install statsmodels
```

---

## 📖 Usage

### Run the Application
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your default browser.

### Basic Workflow
1. **Upload Data**: Click "Upload CSV" to load your dataset
2. **Explore Data**: View data overview, statistics, and visualizations
3. **Clean Data**: Use smart cleaning to handle missing values and outliers
4. **Build Models**: Train machine learning models on your data
5. **Make Predictions**: Generate predictions on new data
6. **Export Results**: Download your processed data and predictions

---

## 📁 Project Structure

```
Information-Retrival/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── LICENSE               # Project license
├── README.md             # This file
├── setup.py              # Package setup configuration
├── template.py           # Project template generator
├── test.py               # Test suite
├── src/
│   ├── __init__.py       # Package initializer
│   └── helper.py         # Utility functions & classes
├── research/
│   └── trials.ipynb      # Jupyter notebook for experimentation
└── .gitignore            # Git ignore rules
```

---

## 📦 Dependencies

### Core Dependencies
- **streamlit** - Web app framework
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **plotly** - Interactive visualizations
- **seaborn** - Statistical visualizations
- **matplotlib** - Plotting library
- **scikit-learn** - Machine learning

### Optional Dependencies
- **pyaudio** - Audio processing
- **speechrecognition** - Speech-to-text
- **pyttsx3** - Text-to-speech
- **statsmodels** - Time series forecasting

See [requirements.txt](requirements.txt) for full list.

---

## 🎯 Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app**:
   ```bash
   streamlit run app.py
   ```

3. **Upload your CSV file** from the sidebar

4. **Explore and analyze** your data with built-in tools

---

## 🎨 UI Features

- **Dark Theme**: Easy on the eyes with dark backgrounds
- **Responsive Design**: Works on desktop and tablet
- **Interactive Charts**: Hover for details, zoom, pan
- **Real-time Updates**: Instant feedback on data changes
- **Progress Indicators**: Visual feedback during processing

---

## 🐛 Troubleshooting

### Voice Features Not Working
```bash
pip install pyaudio speechrecognition pyttsx3
```

### Time Series Features Not Available
```bash
pip install statsmodels
```

### Port Already in Use
```bash
streamlit run app.py --server.port 8502
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Sunil Kundruk**
- GitHub: [@SunilKundruk](https://github.com/SunilKundruk)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue.

---

## 📝 Version History

- **v1.0.0** - Initial release
  - Core data analysis features
  - Machine learning models
  - Data visualization tools
  - Voice capabilities

---

**Last Updated**: February 18, 2026