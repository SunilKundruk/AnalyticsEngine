import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')
import io
import base64
import json
import pickle
from datetime import datetime, timedelta
import time
from sklearn.cluster import DBSCAN
import re

# Only import voice-related libraries if available
VOICE_AVAILABLE = False
try:
    import speech_recognition as sr
    import pyttsx3
    VOICE_AVAILABLE = True
except ImportError:
    st.warning("Voice functionality disabled. Install 'pip install pyaudio speechrecognition pyttsx3' for voice features.")

# Only import time series libraries if available
TIMESERIES_AVAILABLE = False
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    TIMESERIES_AVAILABLE = True
except ImportError:
    st.warning("Time series forecasting disabled. Install 'pip install statsmodels' for forecasting features.")

# Page configuration
st.set_page_config(
    page_title="AI Data Analysis Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI - Dark theme with light text
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f1f2e 0%, #16213e 100%);
        padding: 2rem;
        border-radius: 10px;
        color: #e0e0e0;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        border-top: 3px solid #0f3460;
    }
    .feature-card {
        background: linear-gradient(135deg, #2a2a3e 0%, #1f1f2e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #00d4ff;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease;
        color: #e0e0e0;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        border-left: 4px solid #00ffff;
    }
    .chat-message {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border-left: 3px solid #00d4ff;
        animation: fadeIn 0.3s ease;
        color: #e0e0e0;
    }
    .user-message {
        background: linear-gradient(135deg, #3d2e4e 0%, #2a1f3d 100%);
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        text-align: right;
        border-right: 3px solid #9c27b0;
        animation: fadeIn 0.3s ease;
        color: #e0e0e0;
    }
    .metric-card {
        background: linear-gradient(135deg, #2a2a3e 0%, #1f1f2e 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #00d4ff;
        color: #00ffff;
        font-weight: bold;
    }
    .success-message {
        background: linear-gradient(135deg, #1e3d20 0%, #2d5a2d 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
        color: #81c784;
    }
    .error-message {
        background: linear-gradient(135deg, #3d1e1e 0%, #5a2d2d 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #f44336;
        color: #ef5350;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stTab {
        font-size: 16px;
        font-weight: 500;
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

class DataCleaner:
    """Smart data cleaning utilities with enhanced features"""
    
    @staticmethod
    def auto_clean(df):
        """Comprehensive one-click auto-cleaning"""
        cleaned_df = df.copy()
        cleaning_report = []
        
        # Handle missing values intelligently
        for col in cleaned_df.columns:
            missing_count = cleaned_df[col].isnull().sum()
            if missing_count > 0:
                if cleaned_df[col].dtype in ['int64', 'float64']:
                    # Use median for numeric columns
                    fill_value = cleaned_df[col].median()
                    cleaned_df[col].fillna(fill_value, inplace=True)
                    cleaning_report.append(f"Filled {missing_count} missing values in '{col}' with median ({fill_value:.2f})")
                else:
                    # Use mode for categorical columns
                    if not cleaned_df[col].mode().empty:
                        fill_value = cleaned_df[col].mode()[0]
                        cleaned_df[col].fillna(fill_value, inplace=True)
                        cleaning_report.append(f"Filled {missing_count} missing values in '{col}' with mode ('{fill_value}')")
                    else:
                        cleaned_df[col].fillna('Unknown', inplace=True)
                        cleaning_report.append(f"Filled {missing_count} missing values in '{col}' with 'Unknown'")
        
        # Remove duplicates
        initial_rows = len(cleaned_df)
        cleaned_df.drop_duplicates(inplace=True)
        duplicates_removed = initial_rows - len(cleaned_df)
        if duplicates_removed > 0:
            cleaning_report.append(f"Removed {duplicates_removed} duplicate rows")
        
        # Handle outliers using IQR method
        for col in cleaned_df.select_dtypes(include=[np.number]).columns:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:  # Avoid division by zero
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_count = ((cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)).sum()
                if outliers_count > 0:
                    cleaned_df[col] = cleaned_df[col].clip(lower_bound, upper_bound)
                    cleaning_report.append(f"Capped {outliers_count} outliers in '{col}'")
        
        # Convert string numbers to numeric
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            try:
                # Try to convert to numeric
                numeric_series = pd.to_numeric(cleaned_df[col], errors='coerce')
                if not numeric_series.isnull().all():
                    non_null_ratio = numeric_series.notnull().sum() / len(numeric_series)
                    if non_null_ratio > 0.8:  # If 80% can be converted to numeric
                        cleaned_df[col] = numeric_series
                        cleaning_report.append(f"Converted '{col}' to numeric type")
            except:
                pass
        
        return cleaned_df, cleaning_report
    
    @staticmethod
    def detect_anomalies(df, contamination=0.1):
        """Enhanced anomaly detection with multiple methods"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return None, "No numeric columns for anomaly detection"
        
        try:
            # Isolation Forest method
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            anomalies = iso_forest.fit_predict(df[numeric_cols])
            
            anomaly_df = df.copy()
            anomaly_df['anomaly_score'] = anomalies
            anomaly_df['anomaly_method'] = 'Isolation Forest'
            
            anomaly_count = (anomalies == -1).sum()
            
            return anomaly_df[anomaly_df['anomaly_score'] == -1], f"Detected {anomaly_count} anomalies using Isolation Forest"
        except Exception as e:
            return None, f"Error in anomaly detection: {str(e)}"

class PredictionEngine:
    """Enhanced prediction and forecasting engine"""
    
    def __init__(self, df):
        self.df = df
        self.models = {}
        self.scalers = {}
        
    def forecast_timeseries(self, target_col, periods=30):
        """Enhanced time series forecasting"""
        if not TIMESERIES_AVAILABLE:
            return None, "Time series libraries not available. Install statsmodels."
        
        try:
            if target_col not in self.df.columns:
                return None, f"Column '{target_col}' not found"
            
            # Ensure we have enough data
            if len(self.df) < 10:
                return None, "Insufficient data for forecasting (minimum 10 points required)"
            
            data = self.df[target_col].dropna()
            
            if len(data) < 10:
                return None, "Insufficient non-null data for forecasting"
            
            # Try different ARIMA orders
            best_aic = float('inf')
            best_model = None
            best_order = None
            
            # Grid search for best ARIMA parameters
            for p in range(0, 3):
                for d in range(0, 2):
                    for q in range(0, 3):
                        try:
                            model = ARIMA(data, order=(p, d, q))
                            fitted_model = model.fit()
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_model = fitted_model
                                best_order = (p, d, q)
                        except:
                            continue
            
            if best_model is None:
                return None, "Could not fit ARIMA model"
            
            # Generate forecast
            forecast = best_model.forecast(steps=periods)
            conf_int = best_model.get_forecast(steps=periods).conf_int()
            
            # Create forecast dataframe
            forecast_dates = pd.date_range(
                start=len(data), periods=periods, freq='D'
            )
            
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Forecast': forecast,
                'Lower_CI': conf_int.iloc[:, 0] if len(conf_int.columns) > 0 else forecast * 0.95,
                'Upper_CI': conf_int.iloc[:, 1] if len(conf_int.columns) > 1 else forecast * 1.05
            })
            
            return {
                'historical': data,
                'forecast': forecast_df,
                'model': best_model,
                'order': best_order,
                'aic': best_aic
            }, f"Successfully generated forecast using ARIMA{best_order}"
        except Exception as e:
            return None, f"Forecasting error: {str(e)}"
    
    def predict_classification(self, target_col, test_size=0.2):
        """Enhanced classification/regression prediction"""
        try:
            if target_col not in self.df.columns:
                return None, f"Column '{target_col}' not found"
            
            # Prepare data
            X = self.df.drop(target_col, axis=1)
            y = self.df[target_col]
            
            # Remove non-numeric columns that can't be easily encoded
            categorical_cols = X.select_dtypes(include=['object']).columns
            
            # Handle categorical variables
            for col in categorical_cols:
                unique_vals = X[col].nunique()
                if unique_vals > 50:  # Too many unique values, drop the column
                    X = X.drop(col, axis=1)
                else:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
            
            # Handle target variable if categorical
            target_is_categorical = False
            if y.dtype == 'object':
                le_target = LabelEncoder()
                y = le_target.fit_transform(y.astype(str))
                target_is_categorical = True
            
            # Check if we have enough data
            if len(X) < 10:
                return None, "Insufficient data for model training (minimum 10 samples required)"
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=min(test_size, 0.3), random_state=42, stratify=y if target_is_categorical else None
            )
            
            # Determine if it's classification or regression
            unique_targets = len(np.unique(y))
            is_classification = unique_targets <= 10 or target_is_categorical
            
            # Train model
            if is_classification:
                model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                score = accuracy_score(y_test, predictions)
                metric = "Accuracy"
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                score = mean_squared_error(y_test, predictions)
                metric = "MSE"
            
            self.models[f'model_{target_col}'] = model
            
            return {
                'model': model,
                'predictions': predictions,
                'actual': y_test,
                'score': score,
                'metric': metric,
                'feature_importance': dict(zip(X.columns, model.feature_importances_)),
                'is_classification': is_classification
            }, f"Successfully trained {'classification' if is_classification else 'regression'} model"
        except Exception as e:
            return None, f"Prediction error: {str(e)}"

class VoiceInterface:
    """Voice command interface with error handling"""
    
    def __init__(self):
        if VOICE_AVAILABLE:
            try:
                self.recognizer = sr.Recognizer()
                self.microphone = sr.Microphone()
                self.tts_engine = pyttsx3.init()
                self.available = True
            except Exception as e:
                self.available = False
                st.error(f"Voice interface initialization failed: {str(e)}")
        else:
            self.available = False
        
    def listen_for_command(self):
        """Listen for voice command with better error handling"""
        if not self.available:
            return "Voice interface not available"
        
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
            command = self.recognizer.recognize_google(audio)
            return command.lower()
        except sr.UnknownValueError:
            return "Could not understand audio. Please try again."
        except sr.RequestError:
            return "Could not connect to speech recognition service"
        except sr.WaitTimeoutError:
            return "No audio detected. Please try again."
        except Exception as e:
            return f"Voice recognition error: {str(e)}"
    
    def speak_response(self, text):
        """Convert text to speech with error handling"""
        if not self.available:
            return
        
        try:
            # Limit text length for TTS
            if len(text) > 200:
                text = text[:200] + "..."
            
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            st.error(f"Text-to-speech error: {str(e)}")

class ConversationalAI:
    """Enhanced conversational interface"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
    
    def generate_insights(self, df, query):
        """Generate comprehensive AI insights about the data"""
        try:
            # Enhanced statistical summary
            summary = df.describe()
            shape = df.shape
            missing_values = df.isnull().sum()
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            # Advanced pattern matching for queries
            query_lower = query.lower()
            
            if any(word in query_lower for word in ['summary', 'describe', 'overview', 'tell me about']):
                insights = []
                insights.append(f"📊 Your dataset contains {shape[0]:,} rows and {shape[1]} columns.")
                insights.append(f"📈 Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:5])}")
                if len(numeric_cols) > 5:
                    insights.append(f"   ... and {len(numeric_cols)-5} more")
                insights.append(f"📝 Categorical columns ({len(categorical_cols)}): {', '.join(categorical_cols[:3])}")
                if len(categorical_cols) > 3:
                    insights.append(f"   ... and {len(categorical_cols)-3} more")
                
                total_missing = missing_values.sum()
                if total_missing > 0:
                    insights.append(f"⚠️ Total missing values: {total_missing:,} ({total_missing/df.size*100:.1f}% of all data)")
                else:
                    insights.append("✅ No missing values detected!")
                
                return "\n".join(insights)
            
            elif any(word in query_lower for word in ['missing', 'null', 'nan', 'empty']):
                if missing_values.sum() == 0:
                    return "✅ Great news! Your dataset has no missing values."
                else:
                    missing_cols = missing_values[missing_values > 0]
                    insights = [f"📋 Missing values analysis:"]
                    for col, count in missing_cols.head(10).items():
                        percentage = (count / len(df)) * 100
                        insights.append(f"  • {col}: {count:,} missing ({percentage:.1f}%)")
                    return "\n".join(insights)
            
            elif any(word in query_lower for word in ['correlation', 'relationship', 'related']):
                if len(numeric_cols) < 2:
                    return "❌ Need at least 2 numeric columns for correlation analysis."
                else:
                    corr_matrix = df[numeric_cols].corr()
                    # Find highest correlations
                    high_corr = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_val = corr_matrix.iloc[i, j]
                            if abs(corr_val) > 0.5:
                                high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
                    
                    if high_corr:
                        insights = ["🔗 Strong correlations found:"]
                        for col1, col2, corr in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True)[:5]:
                            insights.append(f"  • {col1} ↔ {col2}: {corr:.3f}")
                        return "\n".join(insights)
                    else:
                        return "📊 No strong correlations (>0.5) found between numeric columns."
            
            elif any(word in query_lower for word in ['unique', 'distinct', 'values']):
                insights = ["🎯 Unique values per column:"]
                for col in df.columns[:10]:
                    unique_count = df[col].nunique()
                    total_count = len(df[col].dropna())
                    percentage = (unique_count / total_count) * 100 if total_count > 0 else 0
                    insights.append(f"  • {col}: {unique_count:,} unique ({percentage:.1f}%)")
                return "\n".join(insights)
            
            elif any(word in query_lower for word in ['distribution', 'spread', 'range']):
                if len(numeric_cols) == 0:
                    return "❌ No numeric columns available for distribution analysis."
                insights = ["📈 Distribution summary for numeric columns:"]
                for col in numeric_cols[:5]:
                    col_data = df[col].dropna()
                    insights.append(f"  • {col}: Range [{col_data.min():.2f}, {col_data.max():.2f}], Mean: {col_data.mean():.2f}, Std: {col_data.std():.2f}")
                return "\n".join(insights)
            
            else:
                # General insights
                insights = []
                insights.append(f"🤖 AI Analysis of your dataset:")
                insights.append(f"📊 Dataset size: {shape[0]:,} rows × {shape[1]} columns")
                
                if len(numeric_cols) > 0:
                    insights.append(f"📈 {len(numeric_cols)} numeric columns available for analysis")
                
                if missing_values.sum() > 0:
                    insights.append(f"⚠️ {missing_values.sum():,} missing values need attention")
                
                insights.append("💡 Try asking about: summaries, correlations, missing values, or distributions")
                
                return "\n".join(insights)
        except Exception as e:
            return f"🤖 I encountered an error analyzing your data: {str(e)}"

def create_download_link(df, filename, file_format="csv"):
    """Create download link for dataframe with better formatting"""
    try:
        if file_format == "csv":
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv" style="text-decoration: none; background: #4CAF50; color: white; padding: 10px 15px; border-radius: 5px; display: inline-block; margin: 5px;">📥 Download CSV</a>'
        elif file_format == "excel":
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Data', index=False)
            excel_data = output.getvalue()
            b64 = base64.b64encode(excel_data).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx" style="text-decoration: none; background: #2196F3; color: white; padding: 10px 15px; border-radius: 5px; display: inline-block; margin: 5px;">📊 Download Excel</a>'
        
        return href
    except Exception as e:
        return f"<p>Error creating download link: {str(e)}</p>"

def create_visualization(df, chart_type, x_col, y_col=None):
    """Create various types of visualizations"""
    try:
        if chart_type == "histogram" and x_col in df.columns:
            fig = px.histogram(df, x=x_col, title=f"Distribution of {x_col}")
            return fig
        elif chart_type == "scatter" and x_col in df.columns and y_col in df.columns:
            fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
            return fig
        elif chart_type == "box" and x_col in df.columns:
            fig = px.box(df, y=x_col, title=f"Box Plot of {x_col}")
            return fig
        elif chart_type == "correlation" and len(df.select_dtypes(include=[np.number]).columns) > 1:
            corr_matrix = df.select_dtypes(include=[np.number]).corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Matrix")
            return fig
        else:
            return None
    except Exception as e:
        st.error(f"Visualization error: {str(e)}")
        return None

def main():
    # Header with enhanced styling
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Data Analysis Platform</h1>
        <p>Advanced Analytics • Voice Control • Predictive Modeling • Smart Insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'cleaned_df' not in st.session_state:
        st.session_state.cleaned_df = None
    if 'cleaning_report' not in st.session_state:
        st.session_state.cleaning_report = []
    if 'voice_interface' not in st.session_state and VOICE_AVAILABLE:
        st.session_state.voice_interface = VoiceInterface()
    if 'ai_assistant' not in st.session_state:
        st.session_state.ai_assistant = ConversationalAI()
    
    # Sidebar with enhanced features
    with st.sidebar:
        st.markdown("## 🎯 Control Panel")
        
        # File upload section
        st.markdown("### 📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Choose your dataset", 
            type=['csv', 'xlsx', 'json'],
            help="Supported formats: CSV, Excel, JSON"
        )
        
        if uploaded_file:
            try:
                with st.spinner("Loading dataset..."):
                    if uploaded_file.name.endswith('.csv'):
                        st.session_state.df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith('.xlsx'):
                        st.session_state.df = pd.read_excel(uploaded_file)
                    elif uploaded_file.name.endswith('.json'):
                        st.session_state.df = pd.read_json(uploaded_file)
                
                st.markdown(f"""
                <div class="success-message">
                    ✅ Successfully loaded<br>
                    📊 {st.session_state.df.shape[0]:,} rows × {st.session_state.df.shape[1]} columns
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"""
                <div class="error-message">
                    ❌ Error loading file:<br>
                    {str(e)}
                </div>
                """, unsafe_allow_html=True)
        
        # Quick actions
        if st.session_state.df is not None:
            st.markdown("### 🚀 Quick Actions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🧹 Clean Data", use_container_width=True):
                    with st.spinner("Cleaning data..."):
                        cleaner = DataCleaner()
                        cleaned_df, cleaning_report = cleaner.auto_clean(st.session_state.df)
                        st.session_state.cleaned_df = cleaned_df
                        st.session_state.cleaning_report = cleaning_report
                        st.success("✅ Data cleaned!")
                        st.rerun()
            
            with col2:
                if st.button("🔍 Find Anomalies", use_container_width=True):
                    with st.spinner("Detecting anomalies..."):
                        cleaner = DataCleaner()
                        anomaly_results, message = cleaner.detect_anomalies(st.session_state.df)
                        if anomaly_results is not None:
                            st.session_state.anomaly_results = anomaly_results
                            st.success(f"✅ {message}")
                        else:
                            st.error(f"❌ {message}")
            
            # Dataset info
            st.markdown("### 📈 Dataset Info")
            df_to_show = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", f"{df_to_show.shape[0]:,}")
                st.metric("Missing", f"{df_to_show.isnull().sum().sum():,}")
            with col2:
                st.metric("Columns", df_to_show.shape[1])
                st.metric("Memory", f"{df_to_show.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Main content area
    if st.session_state.df is not None:
        # Create enhanced tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Explorer", "🤖 AI Assistant", "📈 Analytics", "🎨 Visualizations", "⬇️ Export"])
        
        with tab1:
            st.markdown("## 📊 Data Explorer")
            
            # Show cleaning report if available
            if st.session_state.cleaning_report:
                with st.expander("🧹 Data Cleaning Report", expanded=False):
                    for report_item in st.session_state.cleaning_report:
                        st.write(f"• {report_item}")
            # Data overview section
            df_to_display = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 📋 Dataset Preview")
                st.dataframe(
                    df_to_display.head(100), 
                    use_container_width=True,
                    height=400
                )
            
            with col2:
                st.markdown("### 📊 Quick Stats")
                
                # Basic statistics
                stats_data = {
                    "Metric": ["Total Rows", "Total Columns", "Missing Values", "Duplicate Rows", "Memory Usage"],
                    "Value": [
                        f"{df_to_display.shape[0]:,}",
                        f"{df_to_display.shape[1]}",
                        f"{df_to_display.isnull().sum().sum():,}",
                        f"{df_to_display.duplicated().sum():,}",
                        f"{df_to_display.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
                    ]
                }
                
                for metric, value in zip(stats_data["Metric"], stats_data["Value"]):
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>{metric}</strong><br>
                        <span style="font-size: 1.2em; color: #1f77b4;">{value}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Column information
            st.markdown("### 🔍 Column Analysis")
            
            col_info = []
            for col in df_to_display.columns:
                col_data = df_to_display[col]
                col_info.append({
                    "Column": col,
                    "Type": str(col_data.dtype),
                    "Non-Null": f"{col_data.notnull().sum():,}",
                    "Null": f"{col_data.isnull().sum():,}",
                    "Unique": f"{col_data.nunique():,}",
                    "Sample Values": ", ".join([str(x) for x in col_data.dropna().unique()[:3]])
                })
            
            col_info_df = pd.DataFrame(col_info)
            st.dataframe(col_info_df, use_container_width=True)
        
        with tab2:
            st.markdown("## 🤖 AI Assistant")
            
            # Voice control section
            if VOICE_AVAILABLE and hasattr(st.session_state, 'voice_interface') and st.session_state.voice_interface.available:
                st.markdown("### 🎤 Voice Control")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🎤 Start Voice Command", use_container_width=True):
                        with st.spinner("Listening... Please speak your question"):
                            voice_command = st.session_state.voice_interface.listen_for_command()
                            if voice_command and not voice_command.startswith("Could not") and not voice_command.startswith("No audio"):
                                st.session_state.messages.append({"role": "user", "content": voice_command, "source": "voice"})
                                # Generate response
                                response = st.session_state.ai_assistant.generate_insights(df_to_display, voice_command)
                                st.session_state.messages.append({"role": "assistant", "content": response})
                                st.session_state.voice_interface.speak_response(response[:200])  # Limit speech length
                                st.rerun()
                            else:
                                st.error(voice_command)
                
                with col2:
                    if st.button("🔇 Stop Voice", use_container_width=True):
                        pass
            
            # Chat interface
            st.markdown("### 💬 Chat with Your Data")
            
            # Display chat history
            for message in st.session_state.messages:
                if message["role"] == "user":
                    source_icon = "🎤" if message.get("source") == "voice" else "👤"
                    st.markdown(f"""
                    <div class="user-message">
                        {source_icon} <strong>You:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message">
                        🤖 <strong>AI Assistant:</strong><br>{message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Chat input
            user_input = st.text_input(
                "Ask me anything about your data:",
                placeholder="e.g., 'Tell me about missing values' or 'Show me correlations'",
                key="chat_input"
            )
            
            if st.button("Send 📤") and user_input:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Generate AI response
                with st.spinner("AI is analyzing..."):
                    response = st.session_state.ai_assistant.generate_insights(df_to_display, user_input)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                
                st.rerun()
            
            # Suggested questions
            st.markdown("### 💡 Suggested Questions")
            suggested_questions = [
                "Give me a summary of this dataset",
                "What columns have missing values?",
                "Show me correlations between variables",
                "Tell me about data distributions",
                "Find unique values in each column"
            ]
            
            cols = st.columns(2)
            for i, question in enumerate(suggested_questions):
                with cols[i % 2]:
                    if st.button(f"💭 {question}", key=f"suggestion_{i}", use_container_width=True):
                        st.session_state.messages.append({"role": "user", "content": question})
                        response = st.session_state.ai_assistant.generate_insights(df_to_display, question)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.rerun()
        
        with tab3:
            st.markdown("## 📈 Advanced Analytics")
            
            # Prediction section
            st.markdown("### 🎯 Predictive Modeling")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔮 Classification/Regression")
                target_column = st.selectbox(
                    "Select target column for prediction:",
                    options=df_to_display.columns,
                    key="prediction_target"
                )
                
                if st.button("🚀 Train Model", key="train_model"):
                    if target_column:
                        with st.spinner("Training model..."):
                            engine = PredictionEngine(df_to_display)
                            result, message = engine.predict_classification(target_column)
                            
                            if result:
                                st.success(f"✅ {message}")
                                
                                # Display results
                                st.markdown("**Model Performance:**")
                                st.metric(result['metric'], f"{result['score']:.4f}")
                                
                                # Feature importance
                                if result['feature_importance']:
                                    st.markdown("**Feature Importance:**")
                                    importance_df = pd.DataFrame([
                                        {"Feature": k, "Importance": v} 
                                        for k, v in sorted(result['feature_importance'].items(), 
                                                         key=lambda x: x[1], reverse=True)[:10]
                                    ])
                                    st.bar_chart(importance_df.set_index('Feature'))
                            else:
                                st.error(f"❌ {message}")
            
            with col2:
                st.markdown("#### 📊 Time Series Forecasting")
                if TIMESERIES_AVAILABLE:
                    numeric_cols = df_to_display.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        forecast_column = st.selectbox(
                            "Select column for forecasting:",
                            options=numeric_cols,
                            key="forecast_target"
                        )
                        
                        forecast_periods = st.slider("Forecast periods:", 10, 100, 30)
                        
                        if st.button("📈 Generate Forecast", key="generate_forecast"):
                            with st.spinner("Generating forecast..."):
                                engine = PredictionEngine(df_to_display)
                                result, message = engine.forecast_timeseries(forecast_column, forecast_periods)
                                
                                if result:
                                    st.success(f"✅ {message}")
                                    
                                    # Plot forecast
                                    fig = go.Figure()
                                    
                                    # Historical data
                                    fig.add_trace(go.Scatter(
                                        y=result['historical'].values,
                                        mode='lines',
                                        name='Historical',
                                        line=dict(color='blue')
                                    ))
                                    
                                    # Forecast
                                    forecast_start = len(result['historical'])
                                    fig.add_trace(go.Scatter(
                                        x=list(range(forecast_start, forecast_start + len(result['forecast']))),
                                        y=result['forecast']['Forecast'].values,
                                        mode='lines',
                                        name='Forecast',
                                        line=dict(color='red', dash='dash')
                                    ))
                                    
                                    fig.update_layout(
                                        title=f"Time Series Forecast for {forecast_column}",
                                        xaxis_title="Time Period",
                                        yaxis_title="Value"
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.error(f"❌ {message}")
                    else:
                        st.info("No numeric columns available for forecasting")
                else:
                    st.info("Install statsmodels for time series forecasting features")
            
            # Anomaly detection results
            if hasattr(st.session_state, 'anomaly_results') and st.session_state.anomaly_results is not None:
                st.markdown("### 🚨 Anomaly Detection Results")
                st.dataframe(st.session_state.anomaly_results, use_container_width=True)
        
        with tab4:
            st.markdown("## 🎨 Data Visualizations")
            
            # Visualization controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                chart_type = st.selectbox(
                    "Chart Type:",
                    ["histogram", "scatter", "box", "correlation", "line", "bar"],
                    key="chart_type"
                )
            
            with col2:
                x_column = st.selectbox(
                    "X-axis Column:",
                    options=df_to_display.columns,
                    key="x_column"
                )
            
            with col3:
                y_column = st.selectbox(
                    "Y-axis Column:",
                    options=["None"] + list(df_to_display.columns),
                    key="y_column"
                )
            
            # Generate visualization
            if st.button("📊 Generate Chart", use_container_width=True):
                y_col = None if y_column == "None" else y_column
                fig = create_visualization(df_to_display, chart_type, x_column, y_col)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Could not generate chart with selected parameters")
            
            # Pre-built visualizations
            st.markdown("### 🎯 Quick Visualizations")
            
            viz_cols = st.columns(3)
            
            with viz_cols[0]:
                if st.button("📊 Data Distribution", use_container_width=True):
                    numeric_cols = df_to_display.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        fig = make_subplots(
                            rows=min(2, len(numeric_cols)), 
                            cols=2,
                            subplot_titles=[f"Distribution of {col}" for col in numeric_cols[:4]]
                        )
                        
                        for i, col in enumerate(numeric_cols[:4]):
                            row = (i // 2) + 1
                            col_pos = (i % 2) + 1
                            fig.add_trace(
                                go.Histogram(x=df_to_display[col], name=col, showlegend=False),
                                row=row, col=col_pos
                            )
                        
                        fig.update_layout(height=600, title_text="Data Distributions")
                        st.plotly_chart(fig, use_container_width=True)
            
            with viz_cols[1]:
                if st.button("🔗 Correlation Heatmap", use_container_width=True):
                    numeric_cols = df_to_display.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 1:
                        corr_matrix = df_to_display[numeric_cols].corr()
                        fig = px.imshow(
                            corr_matrix, 
                            text_auto=True, 
                            aspect="auto",
                            title="Correlation Matrix",
                            color_continuous_scale="RdBu"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            with viz_cols[2]:
                if st.button("📈 Missing Data Pattern", use_container_width=True):
                    missing_data = df_to_display.isnull().sum()
                    if missing_data.sum() > 0:
                        fig = px.bar(
                            x=missing_data.index,
                            y=missing_data.values,
                            title="Missing Values by Column"
                        )
                        fig.update_xaxis(tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.success("✅ No missing data to visualize!")
        
        with tab5:
            st.markdown("## ⬇️ Export & Download")
            
            # Export options
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📁 Download Options")
                
                export_df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df
                filename = st.text_input("Filename:", value="processed_data")
                
                # Download buttons
                csv_link = create_download_link(export_df, filename, "csv")
                excel_link = create_download_link(export_df, filename, "excel")
                
                st.markdown(csv_link, unsafe_allow_html=True)
                st.markdown(excel_link, unsafe_allow_html=True)
                
                # Export specific columns
                st.markdown("#### 🎯 Export Selected Columns")
                selected_columns = st.multiselect(
                    "Choose columns to export:",
                    options=export_df.columns,
                    default=list(export_df.columns[:5])
                )
                
                if selected_columns and st.button("📤 Export Selected"):
                    selected_df = export_df[selected_columns]
                    selected_link = create_download_link(selected_df, f"{filename}_selected", "csv")
                    st.markdown(selected_link, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 📊 Export Summary Report")
                
                if st.button("📋 Generate Report", use_container_width=True):
                    report_data = {
                        "Dataset Summary": {
                            "Total Rows": export_df.shape[0],
                            "Total Columns": export_df.shape[1],
                            "Missing Values": export_df.isnull().sum().sum(),
                            "Memory Usage (MB)": round(export_df.memory_usage(deep=True).sum() / 1024**2, 2)
                        },
                        "Column Information": export_df.dtypes.to_dict(),
                        "Missing Values by Column": export_df.isnull().sum().to_dict(),
                        "Unique Values by Column": {col: export_df[col].nunique() for col in export_df.columns}
                    }
                    
                    # Convert to JSON for download
                    json_str = json.dumps(report_data, indent=2, default=str)
                    b64 = base64.b64encode(json_str.encode()).decode()
                    href = f'<a href="data:application/json;base64,{b64}" download="data_report.json" style="text-decoration: none; background: #FF9800; color: white; padding: 10px 15px; border-radius: 5px; display: inline-block; margin: 5px;">📋 Download Report</a>'
                    st.markdown(href, unsafe_allow_html=True)
                
                # Cleaning report export
                if st.session_state.cleaning_report:
                    st.markdown("#### 🧹 Cleaning Report")
                    cleaning_text = "\n".join([f"• {item}" for item in st.session_state.cleaning_report])
                    
                    if st.button("📄 Export Cleaning Report"):
                        b64 = base64.b64encode(cleaning_text.encode()).decode()
                        href = f'<a href="data:text/plain;base64,{b64}" download="cleaning_report.txt" style="text-decoration: none; background: #9C27B0; color: white; padding: 10px 15px; border-radius: 5px; display: inline-block; margin: 5px;">📄 Download Cleaning Report</a>'
                        st.markdown(href, unsafe_allow_html=True)
    
    else:
        # Welcome screen when no data is loaded
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 15px; margin: 2rem 0;">
            <h2 style="color: black;">🚀 Welcome to AI Data Analysis Platform</h2>
            <p style="font-size: 1.2em; color: #666; margin: 1rem 0;">
                Upload your dataset to get started with advanced analytics
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature showcase
        features_col1, features_col2, features_col3 = st.columns(3)
        
        with features_col1:
            st.markdown("""
            <div class="feature-card">
                <h3>🤖 AI-Powered Insights</h3>
                <p>Get intelligent analysis and recommendations from your data with natural language queries.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with features_col2:
            st.markdown("""
            <div class="feature-card">
                <h3>🧹 Smart Data Cleaning</h3>
                <p>Automatic data cleaning with missing value handling, outlier detection, and duplicate removal.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with features_col3:
            st.markdown("""
            <div class="feature-card">
                <h3>📈 Predictive Analytics</h3>
                <p>Build machine learning models and generate forecasts with just a few clicks.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional features
        features_col4, features_col5, features_col6 = st.columns(3)
        
        with features_col4:
            st.markdown("""
            <div class="feature-card">
                <h3>🎤 Voice Control</h3>
                <p>Interact with your data using voice commands for hands-free analysis.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with features_col5:
            st.markdown("""
            <div class="feature-card">
                <h3>🎨 Rich Visualizations</h3>
                <p>Create beautiful, interactive charts and graphs to visualize your insights.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with features_col6:
            st.markdown("""
            <div class="feature-card">
                <h3>⬇️ Easy Export</h3>
                <p>Export your processed data and analysis reports in multiple formats.</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()