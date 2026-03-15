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
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
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
        st.session_state.theme = 'light'
    return st.session_state.theme

def toggle_theme():
    """Toggle between light and dark theme"""
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'

# Initialize theme
current_theme = init_theme()

# Apply theme
theme_attr = 'data-theme="dark"' if current_theme == 'dark' else ''
st.markdown(f'<div {theme_attr}>', unsafe_allow_html=True)

# -------------------- PREMIUM HEADER --------------------
st.markdown("""
<div class="premium-header">
    <div class="floating-icon">📈</div>
    <h1 class="premium-title">AI-Powered Stock Prediction</h1>
    <p class="premium-subtitle">Real-time Multi-Stock Price Prediction & Visualization</p>
</div>
""", unsafe_allow_html=True)

# -------------------- THEME TOGGLE IN SIDEBAR --------------------
st.sidebar.markdown("---")
theme_icon = "🌙" if current_theme == 'light' else "☀️"
theme_label = "Dark Mode" if current_theme == 'light' else "Light Mode"
if st.sidebar.button(f"{theme_icon} {theme_label}", key="theme_toggle", use_container_width=True):
    toggle_theme()
    st.rerun()

# -------------------- PRO MODE TOGGLE --------------------
pro_mode = st.sidebar.checkbox("✨ Pro Dashboard Mode", value=True, help="Enable premium visual effects")

# -------------------- SIDEBAR WITH GLASSMORPHISM --------------------
st.sidebar.markdown("""
<div style="
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 20px;
    margin-bottom: 20px;
    border: 1px solid rgba(255, 255, 255, 0.3);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
">
    <h2 style="text-align: center; margin: 0; color: #fff;">🔍 Select Options</h2>
</div>
""", unsafe_allow_html=True)

stocks = ['AXISBANK.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'RELIANCE.NS', 'SBIN.NS', 'TCS.NS']
ticker = st.sidebar.selectbox("📊 Choose a Stock", stocks, key="stock_select")
period = st.sidebar.selectbox("📅 Historical Time Period", ["1mo", "3mo", "6mo", "1y", "2y"], key="period_select")

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
        
        # Predict next day's close
        pred_close = model.predict(X_next)
        next_close = float(pred_close[0]) if hasattr(pred_close, '__iter__') else float(pred_close)
        
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
    <span class="floating-emoji">📊</span>
    <span>Historical & Predicted Data for {ticker}</span>
</div>
""", unsafe_allow_html=True)

with st.spinner("🔄 Fetching live market data..."):
    data = yf.download(ticker, period=period, interval="1d", auto_adjust=False)
    data.reset_index(inplace=True)

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
    # Try both naming conventions
    model_filename1 = ticker.replace(".", "_") + "_model.joblib"
    model_filename2 = ticker + "_model.joblib"
    model_path1 = f"models/{model_filename1}"
    model_path2 = f"models/{model_filename2}"
    
    model_path = None
    if os.path.exists(model_path1):
        model_path = model_path1
    elif os.path.exists(model_path2):
        model_path = model_path2
    
    if not model_path:
        st.error(f"⚠️ Model file not found for {ticker}")
        st.info(f"Tried: {model_filename1} and {model_filename2}")
        st.info("Available models: " + ", ".join([f.replace("_model.joblib", "") for f in os.listdir("models") if f.endswith("_model.joblib")]))
    else:
        model = joblib.load(model_path)
        last_row = data.iloc[-1]
        
        # Check if required columns exist
        required_cols = ["Open", "High", "Low", "Volume"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            st.error(f"⚠️ Missing required data columns: {missing_cols}")
        else:
            # Generate future predictions
            with st.spinner(f"🔮 Generating predictions for {days_to_predict} days ahead..."):
                future_predictions = generate_future_predictions(model, last_row, data, days_to_predict)
            
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
                    text=f"{ticker} Stock Price - Historical & Predicted",
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
            current_price = float(last_row["Close"])
            
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
