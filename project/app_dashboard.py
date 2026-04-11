import streamlit as st
import yfinance as yf
import pandas as pd
import joblib
import plotly.graph_objects as go
import os
import numpy as np
from datetime import datetime, timedelta

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Stock Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# -------------------- CUSTOM CSS INJECTION --------------------
def load_css():
    """Load custom CSS for premium UI"""
    css_path = 'assets/style.css'
    if os.path.exists(css_path):
        with open(css_path, 'r', encoding='utf-8') as f:
            css_content = f.read()
            # Add cache-busting comment
            css_content = f"/* Cache-busted: {datetime.now().isoformat()} */\n" + css_content
            st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)
    else:
        # Fallback CSS if file not found
        st.markdown("""
        <style>
        .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        </style>
        """, unsafe_allow_html=True)

load_css()

# -------------------- THEME TOGGLE --------------------
def init_theme():
    """Initialize theme state"""
    if 'theme' not in st.session_state:
        st.session_state.theme = 'dark'  # Default to neon dark theme
    return st.session_state.theme

def toggle_theme():
    """Toggle between light and dark theme"""
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'

# Initialize theme
current_theme = init_theme()

# Define theme variables for use in UI elements
theme_icon = "🌙" if current_theme == 'light' else "☀️"
theme_label = "Dark Mode" if current_theme == 'light' else "Light Mode"

# Apply theme
theme_attr = 'data-theme="dark"' if current_theme == 'dark' else ''
st.markdown(f'<div {theme_attr}>', unsafe_allow_html=True)

# Debug: Show current theme
st.sidebar.write(f"Current theme: {current_theme}")
st.sidebar.write(f"Theme attr: {theme_attr}")

# -------------------- PREMIUM HEADER --------------------
st.markdown("""
<div class="premium-header">
    <div class="floating-icon">📈</div>
    <h1 class="premium-title">AI-Powered Stock Prediction</h1>
    <p class="premium-subtitle">Real-time Multi-Stock Price Prediction & Visualization</p>
</div>
""", unsafe_allow_html=True)

# -------------------- FLOATING THEME TOGGLE --------------------
# Add floating theme toggle button
st.markdown(f"""
<div class="theme-toggle" onclick="{{}}">
    <span>{theme_icon}</span>
    <span>{theme_label}</span>
</div>
<script>
    // Add click handler for theme toggle
    document.addEventListener('DOMContentLoaded', function() {{
        const themeToggle = document.querySelector('.theme-toggle');
        if (themeToggle) {{
            themeToggle.addEventListener('click', function() {{
                // Trigger Streamlit rerun by simulating button click
                const streamlitButton = document.querySelector('[data-testid="stButton"] button');
                if (streamlitButton) {{
                    streamlitButton.click();
                }}
            }});
        }}
    }});
</script>
""", unsafe_allow_html=True)

# -------------------- THEME TOGGLE IN SIDEBAR --------------------
st.sidebar.markdown("---")
if st.sidebar.button(f"{theme_icon} {theme_label}", key="theme_toggle", use_container_width=True, help="Toggle between light and dark theme"):
    toggle_theme()
    st.rerun()

# -------------------- PRO MODE TOGGLE --------------------
pro_mode = st.sidebar.checkbox("✨ Pro Dashboard Mode", value=True, help="Enable premium visual effects")

# -------------------- SIDEBAR WITH GLASSMORPHISM --------------------
st.sidebar.markdown(f"""
<div style="
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 20px;
    margin-bottom: 20px;
    border: 1px solid var(--glass-border);
    box-shadow: var(--shadow-soft);
">
    <h2 style="text-align: center; margin: 0; color: var(--text-primary);">🔍 Select Options</h2>
</div>
""", unsafe_allow_html=True)

stocks = ['AXISBANK.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'RELIANCE.NS', 'SBIN.NS', 'TCS.NS']
ticker = st.sidebar.selectbox("📊 Choose a Stock", stocks, key="stock_select")
period = st.sidebar.selectbox("📅 Historical Time Period", ["1mo", "3mo", "6mo", "1y", "2y"], key="period_select")

# Model type selection
model_type = st.sidebar.selectbox(
    "🤖 Choose Model Type", 
    ["Linear Regression", "Random Forest", "Compare Both"], 
    key="model_type_select",
    help="Select between Linear Regression, Random Forest, or compare both models"
)

# Prediction period options
prediction_options = {
    "1 Day": 1,
    "1 Week (7 Days)": 7,
    "2 Weeks (14 Days)": 14,
    "1 Month (30 Days)": 30,
    "3 Months (90 Days)": 90,
    "6 Months (180 Days)": 180,
    "1 Year (365 Days)": 365
}
prediction_period = st.sidebar.selectbox("🔮 Predict Future", list(prediction_options.keys()), key="pred_select")
days_to_predict = prediction_options[prediction_period]

# -------------------- MODEL COMPARISON FUNCTION --------------------
def display_model_comparison(data, all_predictions, ticker, prediction_period, pro_mode, current_theme):
    """Display comparison between multiple models"""
    
    # Create comparison chart
    fig = go.Figure()
    
    # Plot historical data
    fig.add_trace(go.Scatter(
        x=data["Date"], 
        y=data["Close"], 
        mode="lines", 
        name="Historical Close Price",
        line=dict(
            color='#667eea',
            width=3,
            shape='spline'
        ),
        fill='tonexty' if pro_mode else None,
        fillcolor='rgba(102, 126, 234, 0.1)' if pro_mode else None
    ))
    
    # Plot predictions for each model
    colors = {'Linear Regression': '#f093fb', 'Random Forest': '#00ff88'}
    
    for model_name, predictions in all_predictions.items():
        if not predictions.empty:
            fig.add_trace(go.Scatter(
                x=predictions["Date"], 
                y=predictions["Close"], 
                mode="lines", 
                name=f"{model_name} Prediction",
                line=dict(
                    color=colors.get(model_name, '#ff6b6b'),
                    width=3,
                    dash='dash',
                    shape='spline'
                ),
                fill='tonexty' if pro_mode else None,
                fillcolor='rgba(255, 107, 107, 0.15)' if pro_mode else None
            ))
    
    # Add vertical line to separate historical from predicted
    last_historical_date = pd.to_datetime(data["Date"].iloc[-1])
    if isinstance(last_historical_date, pd.Timestamp):
        last_date_dt = last_historical_date.to_pydatetime()
    else:
        last_date_dt = pd.to_datetime(last_historical_date).to_pydatetime()
    
    fig.add_shape(
        type="line",
        x0=last_date_dt,
        x1=last_date_dt,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="#667eea", width=2, dash="dot")
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{ticker} Stock Price - Model Comparison ({prediction_period})",
            font=dict(size=24, family="Inter", color="#1a1a2e" if current_theme == 'light' else "#ffffff"),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(
                text="Date",
                font=dict(size=14, family="Inter", color="#4a5568" if current_theme == 'light' else "#a0aec0")
            ),
            gridcolor="rgba(102, 126, 234, 0.1)",
            showgrid=True
        ),
        yaxis=dict(
            title=dict(
                text="Price (₹)",
                font=dict(size=14, family="Inter", color="#4a5568" if current_theme == 'light' else "#a0aec0")
            ),
            gridcolor="rgba(102, 126, 234, 0.1)",
            showgrid=True
        ),
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)" if current_theme == 'light' else "rgba(0, 0, 0, 0.5)",
            bordercolor="rgba(102, 126, 234, 0.3)",
            borderwidth=1,
            font=dict(family="Inter", size=12)
        ),
        plot_bgcolor="rgba(255, 255, 255, 0.05)" if current_theme == 'dark' else "rgba(255, 255, 255, 0.8)",
        paper_bgcolor="rgba(255, 255, 255, 0)" if current_theme == 'dark' else "rgba(255, 255, 255, 0)",
        margin=dict(l=60, r=40, t=80, b=60)
    )
    
    # Display chart
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display comparison metrics
    st.markdown("""
    <div style="margin: 32px 0;">
        <h3 class="section-header" style="margin-bottom: 24px;">
            <span class="floating-emoji">📊</span>
            <span>Model Comparison Metrics</span>
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    current_price = float(data.iloc[-1]["Close"])
    
    # Create comparison table
    comparison_data = []
    for model_name, predictions in all_predictions.items():
        if not predictions.empty:
            next_day_pred = float(predictions.iloc[0]["Close"])
            final_day_pred = float(predictions.iloc[-1]["Close"])
            avg_pred = float(predictions["Close"].mean())
            
            comparison_data.append({
                'Model': model_name,
                'Current Price': f"₹{current_price:.2f}",
                'Next Day Prediction': f"₹{next_day_pred:.2f}",
                'Final Day Prediction': f"₹{final_day_pred:.2f}",
                'Average Prediction': f"₹{avg_pred:.2f}",
                'Next Day Change %': f"{((next_day_pred - current_price) / current_price * 100):+.2f}%",
                'Total Change %': f"{((final_day_pred - current_price) / current_price * 100):+.2f}%"
            })
    
    # Display comparison table
    if comparison_data:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# -------------------- PREDICTION FUNCTION --------------------
def generate_future_predictions(model, last_row, historical_data, days_to_predict):
    """
    Generate multi-day predictions using the trained model.
    For each future day, estimates Open/High/Low/Volume based on previous predictions and historical patterns.
    """
    predictions = []
    current_date = pd.to_datetime(last_row["Date"])
    
    # Calculate historical statistics for estimation
    historical_volatility = historical_data["Close"].pct_change().std() if len(historical_data) > 1 else 0.02
    avg_volume = historical_data["Volume"].mean() if len(historical_data) > 0 else last_row["Volume"]
    avg_high_low_range = ((historical_data["High"] - historical_data["Low"]) / historical_data["Close"]).mean() if len(historical_data) > 0 else 0.03
    avg_open_close_diff = ((historical_data["Open"] - historical_data["Close"].shift(1)) / historical_data["Close"].shift(1)).mean() if len(historical_data) > 1 else 0.0
    
    # Use the last row as starting point
    prev_close = float(last_row["Close"])
    prev_open = float(last_row["Open"])
    
    # Get recent trend (last 5 days)
    if len(historical_data) >= 5:
        recent_trend = (historical_data["Close"].iloc[-1] - historical_data["Close"].iloc[-5]) / historical_data["Close"].iloc[-5]
    else:
        recent_trend = 0.0
    
    for day in range(days_to_predict):
        # Estimate next day's open price
        gap = prev_close * (avg_open_close_diff * 0.8 + recent_trend * 0.2 / days_to_predict)
        next_open = prev_close + gap
        
        # Estimate high and low based on typical daily range
        daily_range_pct = avg_high_low_range * 0.9
        next_high = next_open * (1 + daily_range_pct / 2)
        next_low = next_open * (1 - daily_range_pct / 2)
        
        # Volume - use average with slight variation based on day of week
        day_of_week = (current_date + timedelta(days=1)).weekday()
        volume_multiplier = 1.0 if day_of_week < 5 else 0.7
        next_volume = avg_volume * volume_multiplier
        
        # Create feature vector
        X_next = pd.DataFrame({
            'Open': [next_open],
            'High': [next_high],
            'Low': [next_low],
            'Volume': [next_volume]
        })
        
        # Predict next day's close with comprehensive error handling
        try:
            # Validate model before prediction
            if model is None:
                raise ValueError("Model is None")
            
            pred_close = model.predict(X_next)
            # Handle numpy array prediction properly
            if hasattr(pred_close, 'item'):
                next_close = float(pred_close.item())
            elif hasattr(pred_close, '__iter__') and len(pred_close) > 0:
                next_close = float(pred_close[0])
            else:
                next_close = float(pred_close)
        except Exception as pred_error:
            # Fallback to simple trend-based prediction if model fails
            trend = historical_data["Close"].pct_change().mean() if len(historical_data) > 1 else 0.0
            next_close = prev_close * (1 + trend * 0.1)  # Conservative prediction
            st.warning(f"⚠️ Model prediction failed, using trend-based fallback: {str(pred_error)}")
        
        # Fix unrealistic Random Forest predictions
        if hasattr(model, 'n_estimators'):  # This is a Random Forest model
            # Check if prediction is unrealistic (more than 50% different from current price)
            if next_close < prev_close * 0.5 or next_close > prev_close * 2.0:
                # Use a simple trend-based prediction instead
                trend = historical_data["Close"].pct_change().mean() if len(historical_data) > 1 else 0.0
                next_close = prev_close * (1 + trend * 0.1)  # Conservative prediction
        
        # Update for next iteration
        prev_close = next_close
        prev_open = next_open
        
        # Store prediction
        current_date += timedelta(days=1)
        while current_date.weekday() >= 5:  # Skip weekends
            current_date += timedelta(days=1)
            
        predictions.append({
            'Date': current_date,
            'Close': next_close,
            'Open': next_open,
            'High': next_high,
            'Low': next_low,
            'Volume': next_volume
        })
    
    return pd.DataFrame(predictions)

# -------------------- FETCH LIVE DATA --------------------
st.markdown(f"""
<div class="section-header">
    <span class="floating-emoji"></span>
    <span>Historical & Predicted Data for {ticker}</span>
</div>
""", unsafe_allow_html=True)

# Function to generate fallback sample data
def generate_fallback_data(ticker_symbol, days=30):
    """Generate realistic sample data when yfinance fails"""
    import numpy as np
    from datetime import datetime, timedelta
    
    # Base prices for different stocks
    base_prices = {
        'HDFCBANK': 1600,
        'ICICIBANK': 1000,
        'INFY': 1500,
        'RELIANCE': 2500,
        'AXISBANK': 800
    }
    
    stock_name = ticker_symbol.replace('.NS', '')
    base_price = base_prices.get(stock_name, 1000)
    
    dates = [datetime.now() - timedelta(days=i) for i in range(days, 0, -1)]
    
    # Generate realistic price movements
    np.random.seed(42)  # For reproducible data
    prices = []
    current_price = base_price
    
    for i in range(days):
        # Random walk with trend
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        current_price = current_price * (1 + change)
        
        # Generate OHLC
        high = current_price * (1 + abs(np.random.normal(0, 0.01)))
        low = current_price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = low + (high - low) * np.random.random()
        close_price = current_price
        volume = int(np.random.normal(1000000, 200000))
        
        prices.append({
            'Date': dates[i],
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close_price, 2),
            'Volume': max(volume, 100000)
        })
    
    return pd.DataFrame(prices)

# Add a button to trigger data loading
if st.sidebar.button("Load Data & Predictions", key="load_data", use_container_width=True):
    with st.spinner("Fetching live market data..."):
        try:
            data = yf.download(ticker, period=period, interval="1d", auto_adjust=False)
            data.reset_index(inplace=True)
            
            # Check if data was actually fetched
            if data.empty or len(data) == 0:
                st.info(f"""
                **Live Data Unavailable** - Using realistic sample data for {ticker}
                
                *Note: This is a demonstration dashboard with simulated historical data and predictions. 
                The models and predictions are for educational purposes only.*
                """)
                data = generate_fallback_data(ticker)
            
            # Store data in session state
            st.session_state.data = data
            st.session_state.data_loaded = True
            st.success(f"Data loaded successfully for {ticker}!")
            st.rerun()
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            st.warning("⚠️ Using sample data for demonstration.")
            data = generate_fallback_data(ticker)
            st.session_state.data = data
            st.session_state.data_loaded = True
            st.rerun()
else:
    # Try to load data from session state or fetch automatically
    if 'data_loaded' not in st.session_state:
        with st.spinner("Fetching live market data..."):
            try:
                data = yf.download(ticker, period=period, interval="1d", auto_adjust=False)
                data.reset_index(inplace=True)
                
                # Check if data was actually fetched
                if data.empty or len(data) == 0:
                    st.warning(f"⚠️ Unable to fetch live data for {ticker}. Using sample data for demonstration.")
                    data = generate_fallback_data(ticker)
                
                st.session_state.data = data
                st.session_state.data_loaded = True
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                st.warning("⚠️ Using sample data for demonstration.")
                data = generate_fallback_data(ticker)
                st.session_state.data = data
                st.session_state.data_loaded = True
    else:
        data = st.session_state.data

# Handle MultiIndex columns from yfinance
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

# Display data in glass card
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.dataframe(data.tail(), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# -------------------- VISUALIZATION --------------------
fig = go.Figure()

# Plot historical data with gradient
fig.add_trace(go.Scatter(
    x=data["Date"], 
    y=data["Close"], 
    mode="lines", 
    name="Historical Close Price",
    line=dict(
        color='#667eea',
        width=3,
        shape='spline'
    ),
    fill='tonexty' if pro_mode else None,
    fillcolor='rgba(102, 126, 234, 0.1)' if pro_mode else None
))

# -------------------- PREDICTION --------------------
st.markdown(f"""
<div class="section-header">
    <span class="floating-emoji">🔮</span>
    <span>Future Price Predictions - {prediction_period}</span>
</div>
""", unsafe_allow_html=True)

try:
    stock_name = ticker.replace(".NS", "")
    
    if model_type == "Compare Both":
        # Load both models for comparison
        lr_model_filename = f"{stock_name}_NS_model.joblib"
        rf_model_filename = f"{stock_name}_NS_RF_model.joblib"
        lr_model_path = f"models/{lr_model_filename}"
        rf_model_path = f"models/{rf_model_filename}"
        
        models_loaded = []
        model_names = []
        
        # Load Linear Regression
        if os.path.exists(lr_model_path):
            try:
                lr_model = joblib.load(lr_model_path)
                models_loaded.append(("Linear Regression", lr_model))
                model_names.append("Linear Regression")
                st.success(f"✅ Loaded Linear Regression model for {stock_name}")
            except Exception as e:
                st.error(f"❌ Error loading Linear Regression model: {str(e)}")
        else:
            st.warning(f"⚠️ Linear Regression model not found: {lr_model_filename}")
        
        # Load Random Forest
        if os.path.exists(rf_model_path):
            try:
                rf_model = joblib.load(rf_model_path)
                models_loaded.append(("Random Forest", rf_model))
                model_names.append("Random Forest")
                st.success(f"✅ Loaded Random Forest model for {stock_name}")
            except Exception as e:
                st.error(f"❌ Error loading Random Forest model: {str(e)}")
        else:
            st.warning(f"⚠️ Random Forest model not found: {rf_model_filename}")
        
        if not models_loaded:
            st.error(f"❌ No models could be loaded for {ticker}")
            st.info("Available models:")
            st.info(f"Linear Regression: {', '.join([f.replace('_model.joblib', '') for f in os.listdir('models') if f.endswith('_model.joblib')])}")
            st.info(f"Random Forest: {', '.join([f.replace('_RF_model.joblib', '') for f in os.listdir('models') if f.endswith('_RF_model.joblib')])}")
        else:
            st.success(f"🎯 Ready for comparison: {', '.join(model_names)} for {stock_name}")
            
            # Check if required columns exist
            required_cols = ["Open", "High", "Low", "Volume"]
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                st.error(f"Missing required data columns: {missing_cols}")
            else:
                # Generate predictions for both models
                last_row = data.iloc[-1]
                all_predictions = {}
                
                with st.spinner(f"Generating predictions for {days_to_predict} days ahead..."):
                    for model_name, model in models_loaded:
                        try:
                            st.write(f"Debug: Attempting prediction for {model_name}")
                            st.write(f"Debug: Last row data: {last_row[['Open', 'High', 'Low', 'Volume']].to_dict()}")
                            predictions = generate_future_predictions(model, last_row, data, days_to_predict)
                            all_predictions[model_name] = predictions
                            st.success(f"Generated predictions for {model_name}")
                            st.write(f"Debug: Predictions shape: {predictions.shape}")
                        except Exception as e:
                            st.error(f"Error generating predictions for {model_name}: {str(e)}")
                            import traceback
                            st.error(f"Full error: {traceback.format_exc()}")
                
                # Display comparison if we have predictions
                if all_predictions:
                    display_model_comparison(data, all_predictions, ticker, prediction_period, pro_mode, current_theme)
                else:
                    st.error("❌ No predictions generated")
    
    else:
        # Single model mode
        if model_type == "Random Forest":
            model_filename = f"{stock_name}_NS_RF_model.joblib"
            model_display_name = f"Random Forest - {stock_name}"
        else:  # Linear Regression (default)
            model_filename = f"{stock_name}_NS_model.joblib"
            model_display_name = f"Linear Regression - {stock_name}"
        
        model_path = f"models/{model_filename}"
        
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found for {ticker}")
            st.info(f"🔍 Tried: {model_filename}")
            st.info(f"🤖 Model Type: {model_type}")
            # List available models
            lr_models = [f.replace("_model.joblib", "") for f in os.listdir("models") if f.endswith("_model.joblib")]
            rf_models = [f.replace("_RF_model.joblib", "") for f in os.listdir("models") if f.endswith("_RF_model.joblib")]
            st.info(f"Available Linear Regression models: {', '.join(lr_models)}")
            st.info(f"Available Random Forest models: {', '.join(rf_models)}")
        else:
            # Initialize last_row outside try block to ensure it's always defined
            last_row = data.iloc[-1] if len(data) > 0 else None
            future_predictions = pd.DataFrame()  # Initialize as empty dataframe
            
            try:
                model = joblib.load(model_path)
                
                # Validate model object
                if model is None:
                    raise ValueError("Model loaded as None")
                
                # Display model info
                st.success(f"Loaded {model_display_name}")
                
                # Check if required columns exist
                required_cols = ["Open", "High", "Low", "Volume"]
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    st.error(f"Missing required data columns: {missing_cols}")
                else:
                    # Generate future predictions
                    with st.spinner(f"Generating predictions for {days_to_predict} days ahead..."):
                        future_predictions = generate_future_predictions(model, last_row, data, days_to_predict)
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.info(f"Model file: {model_filename}")
            
            # Plot future predictions on the chart
            if not future_predictions.empty:
                fig.add_trace(go.Scatter(
                    x=future_predictions["Date"], 
                    y=future_predictions["Close"], 
                    mode="lines", 
                    name=f"Predicted Price ({prediction_period})",
                    line=dict(
                        color='#f093fb',
                        width=3,
                        dash='dash',
                        shape='spline'
                    ),
                    fill='tonexty' if pro_mode else None,
                    fillcolor='rgba(240, 147, 251, 0.15)' if pro_mode else None
                ))
                
                # Add a vertical line to separate historical from predicted
                last_historical_date = pd.to_datetime(data["Date"].iloc[-1])
                if isinstance(last_historical_date, pd.Timestamp):
                    last_date_dt = last_historical_date.to_pydatetime()
                else:
                    last_date_dt = pd.to_datetime(last_historical_date).to_pydatetime()
                
                fig.add_shape(
                    type="line",
                    x0=last_date_dt,
                    x1=last_date_dt,
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(color="#667eea", width=2, dash="dot")
                )
                fig.add_annotation(
                    x=last_date_dt,
                    y=0.95,
                    yref="paper",
                    text="<b>Today</b>",
                    showarrow=False,
                    font=dict(color="#667eea", size=12, family="Inter"),
                    bgcolor="rgba(255, 255, 255, 0.9)",
                    bordercolor="#667eea",
                    borderwidth=2,
                    borderpad=8,
                    xshift=10
                )
            
            # Update chart layout with premium styling
            fig.update_layout(
                title=dict(
                    text=f"{ticker} Stock Price - Historical & Predicted ({model_type})",
                    font=dict(size=24, family="Inter", color="#1a1a2e" if current_theme == 'light' else "#ffffff"),
                    x=0.5,
                    xanchor='center'
                ),
                xaxis=dict(
                    title=dict(
                        text="Date",
                        font=dict(size=14, family="Inter", color="#4a5568" if current_theme == 'light' else "#a0aec0")
                    ),
                    gridcolor="rgba(102, 126, 234, 0.1)",
                    showgrid=True
                ),
                yaxis=dict(
                    title=dict(
                        text="Price (₹)",
                        font=dict(size=14, family="Inter", color="#4a5568" if current_theme == 'light' else "#a0aec0")
                    ),
                    gridcolor="rgba(102, 126, 234, 0.1)",
                    showgrid=True
                ),
                hovermode='x unified',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255, 255, 255, 0.8)" if current_theme == 'light' else "rgba(0, 0, 0, 0.5)",
                    bordercolor="rgba(102, 126, 234, 0.3)",
                    borderwidth=1,
                    font=dict(family="Inter", size=12)
                ),
                plot_bgcolor="rgba(255, 255, 255, 0.05)" if current_theme == 'dark' else "rgba(255, 255, 255, 0.8)",
                paper_bgcolor="rgba(255, 255, 255, 0)" if current_theme == 'dark' else "rgba(255, 255, 255, 0)",
                margin=dict(l=60, r=40, t=80, b=60)
            )
            
            # Display chart in glass container
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display prediction metrics in premium cards
            if last_row is not None and "Close" in last_row:
                current_price = float(last_row["Close"])
            else:
                st.error("Unable to get current price - no data available")
                current_price = 0.0
            
            if not future_predictions.empty:
                # Next day prediction
                next_day_pred = future_predictions.iloc[0]["Close"]
                next_day_change = next_day_pred - current_price
                next_day_change_pct = (next_day_change / current_price) * 100
                
                # Final day prediction
                final_day_pred = future_predictions.iloc[-1]["Close"]
                final_day_change = final_day_pred - current_price
                final_day_change_pct = (final_day_change / current_price) * 100
                
                # Average predicted price
                avg_predicted = future_predictions["Close"].mean()
                
                # Premium metric cards
                st.markdown("""
                <div style="margin: 32px 0;">
                    <h3 class="section-header" style="margin-bottom: 24px;">
                        <span class="floating-emoji">💎</span>
                        <span>Key Metrics</span>
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    current_price_str = f"₹{current_price:.2f}"
                    st.markdown(f'''
                    <div class="metric-card">
                        <div style="text-align: center; padding: 8px;">
                            <div style="font-size: 0.75rem; font-weight: 500; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;">Current Price</div>
                            <div style="font-size: 2rem; font-weight: 700; color: var(--text-primary);">{current_price_str}</div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col2:
                    delta_color2 = "#00ff00" if next_day_change_pct >= 0 else "#ff0000"
                    delta_symbol2 = "↑" if next_day_change_pct >= 0 else "↓"
                    next_day_pred_str = f"₹{next_day_pred:.2f}"
                    delta_str2 = f"{delta_symbol2} {next_day_change_pct:+.2f}%"
                    st.markdown(f'''
                    <div class="metric-card">
                        <div style="text-align: center; padding: 8px;">
                            <div style="font-size: 0.75rem; font-weight: 500; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;">Next Day Prediction</div>
                            <div style="font-size: 2rem; font-weight: 700; color: var(--text-primary); margin-bottom: 4px;">{next_day_pred_str}</div>
                            <div style="font-size: 0.9rem; color: {delta_color2}; font-weight: 600;">{delta_str2}</div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col3:
                    delta_color3 = "#00ff00" if final_day_change_pct >= 0 else "#ff0000"
                    delta_symbol3 = "↑" if final_day_change_pct >= 0 else "↓"
                    final_day_pred_str = f"₹{final_day_pred:.2f}"
                    delta_str3 = f"{delta_symbol3} {final_day_change_pct:+.2f}%"
                    prediction_period_lower = prediction_period.lower()
                    st.markdown(f'''
                    <div class="metric-card">
                        <div style="text-align: center; padding: 8px;">
                            <div style="font-size: 0.75rem; font-weight: 500; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;">Prediction for {prediction_period_lower}</div>
                            <div style="font-size: 2rem; font-weight: 700; color: var(--text-primary); margin-bottom: 4px;">{final_day_pred_str}</div>
                            <div style="font-size: 0.9rem; color: {delta_color3}; font-weight: 600;">{delta_str3}</div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col4:
                    avg_predicted_str = f"₹{avg_predicted:.2f}"
                    st.markdown(f'''
                    <div class="metric-card">
                        <div style="text-align: center; padding: 8px;">
                            <div style="font-size: 0.75rem; font-weight: 500; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;">Average Predicted Price</div>
                            <div style="font-size: 2rem; font-weight: 700; color: var(--text-primary);">{avg_predicted_str}</div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Display prediction table in glass card
                st.markdown("""
                <div class="section-header">
                    <span class="floating-emoji">📅</span>
                    <span>Detailed Predictions</span>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                display_predictions = future_predictions[['Date', 'Close', 'Open', 'High', 'Low']].copy()
                display_predictions['Date'] = display_predictions['Date'].dt.strftime('%Y-%m-%d')
                display_predictions.columns = ['Date', 'Predicted Close', 'Predicted Open', 'Predicted High', 'Predicted Low']
                display_predictions = display_predictions.round(2)
                st.dataframe(display_predictions, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Prediction trend analysis in premium cards
                st.markdown("""
                <div class="section-header">
                    <span class="floating-emoji">📊</span>
                    <span>Prediction Trend Analysis</span>
                </div>
                """, unsafe_allow_html=True)
                
                trend_col1, trend_col2 = st.columns(2)
                
                with trend_col1:
                    max_pred = future_predictions["Close"].max()
                    max_pred_date = future_predictions.loc[future_predictions["Close"].idxmax(), "Date"]
                    st.markdown(f'''
                    <div class="prediction-card">
                        <div style="text-align: center; padding: 8px;">
                            <div style="font-size: 0.75rem; font-weight: 500; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;">Highest Predicted Price</div>
                            <div style="font-size: 2rem; font-weight: 700; color: var(--text-primary); margin-bottom: 4px;">₹{max_pred:.2f}</div>
                            <div style="font-size: 0.85rem; color: var(--text-secondary); font-weight: 500;">On {max_pred_date.strftime('%Y-%m-%d')}</div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with trend_col2:
                    min_pred = future_predictions["Close"].min()
                    min_pred_date = future_predictions.loc[future_predictions["Close"].idxmin(), "Date"]
                    st.markdown(f'''
                    <div class="prediction-card">
                        <div style="text-align: center; padding: 8px;">
                            <div style="font-size: 0.75rem; font-weight: 500; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;">Lowest Predicted Price</div>
                            <div style="font-size: 2rem; font-weight: 700; color: var(--text-primary); margin-bottom: 4px;">₹{min_pred:.2f}</div>
                            <div style="font-size: 0.85rem; color: var(--text-secondary); font-weight: 500;">On {min_pred_date.strftime('%Y-%m-%d')}</div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                
except FileNotFoundError as e:
    st.error(f"⚠️ Model file not found: {e}")
    fig.update_layout(title=f"{ticker} Stock Price Trend", xaxis_title="Date", yaxis_title="Price (₹)")
    st.plotly_chart(fig, use_container_width=True)
except KeyError as e:
    st.error(f"⚠️ Missing data column: {e}")
    fig.update_layout(title=f"{ticker} Stock Price Trend", xaxis_title="Date", yaxis_title="Price (₹)")
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"⚠️ Error making prediction: {str(e)}")
    st.exception(e)
    fig.update_layout(title=f"{ticker} Stock Price Trend", xaxis_title="Date", yaxis_title="Price (₹)")
    st.plotly_chart(fig, use_container_width=True)

# Close theme div
st.markdown('</div>', unsafe_allow_html=True)
