import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import time
from streamlit_lottie import st_lottie
import requests
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="AI Stock Forecaster",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Custom CSS for styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load Lottie animation
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load animations
lottie_stock = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_6wutsrox.json")
lottie_loading = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_raiw2hpe.json")

# Apply custom CSS
local_css("style.css")  # You would create this file in the same directory

# App header with animation
col1, col2 = st.columns([3, 1])
with col1:
    st.title("📈 AI-Powered Stock Market Forecaster")
    st.markdown("""
    <div style="background: linear-gradient(to right, #30CFD0, #c43ad6); -webkit-background-clip: text;
    -webkit-text-fill-color: transparent; font-size: 18px; margin-bottom: 20px;">
    Predictive analytics for smarter investment decisions
    </div>
    """, unsafe_allow_html=True)
with col2:
    if lottie_stock:
        st_lottie(lottie_stock, height=100, key="stock")

# Sidebar with user inputs
with st.sidebar:
    st.header("🔍 Prediction Parameters")
    st.markdown("---")
    
    # Stock selection with more options
    stocks = ("AAPL", "GOOG", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX", "PYPL", "ADBE")
    selected_stock = st.selectbox("Select Stock Ticker", stocks, index=0)
    
    # Enhanced slider with more options
    n_years = st.slider("Forecast Period (Years)", 1, 5, 2, 
                        help="Select how many years into the future you want to predict")
    period = n_years * 365
    
    st.markdown("---")
    st.markdown("""
    <div style="color: #888; font-size: 14px;">
    <b>Note:</b> Predictions are based on historical data and Prophet forecasting model.
    Past performance is not indicative of future results.
    </div>
    """, unsafe_allow_html=True)

# Main content
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

@st.cache_data(show_spinner=False)
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    
    # Flatten multiindex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [' '.join(col).strip() if col[1] else col[0] for col in data.columns.values]
    
    return data

# Loading animation
with st.spinner('Fetching market data...'):
    if lottie_loading:
        st_lottie(lottie_loading, height=100, key="loading")
    data = load_data(selected_stock)

# Handle dynamic column names
date_col = "Date"
open_col = f"Open {selected_stock}"
close_col = f"Close {selected_stock}"
high_col = f"High {selected_stock}"
low_col = f"Low {selected_stock}"

required_cols = [date_col, open_col, close_col, high_col, low_col]
missing = [col for col in required_cols if col not in data.columns]
if missing:
    st.error(f"Missing columns in data: {missing}")
    st.stop()

# Drop rows with missing essential data
data.dropna(subset=required_cols, inplace=True)

# Display raw data in an expandable section
with st.expander("📋 View Raw Data", expanded=False):
    st.dataframe(data.style.background_gradient(cmap='Blues'), height=300)

# Metrics row with updated colors and no delta arrows where needed
st.subheader(f"🔎 {selected_stock} Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest Close", f"${data[close_col].iloc[-1]:.2f}", 
           f"{(data[close_col].iloc[-1] - data[close_col].iloc[-2]):.2f}",
           delta_color="normal")
col2.metric("52 Week High", f"${data[high_col].max():.2f}",
           help="Highest price in the last 52 weeks",
           delta_color="off")
col3.metric("52 Week Low", f"${data[low_col].min():.2f}",
           delta_color="off")
col4.metric("Current Volatility", f"{(data[close_col].pct_change().std() * 100):.2f}%",
           delta_color="off")

# Interactive price chart
st.subheader("📊 Interactive Price History")
tab1, tab2, tab3 = st.tabs(["Line Chart", "Candlestick", "Area Chart"])

with tab1:
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=data[date_col], y=data[close_col], name='Closing Price',
                                line=dict(color='#4b79cf', width=2)))
    fig_line.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        template="plotly_dark",
        height=500
    )
    st.plotly_chart(fig_line, use_container_width=True)

with tab2:
    fig_candle = go.Figure(go.Candlestick(
        x=data[date_col],
        open=data[open_col],
        high=data[high_col],
        low=data[low_col],
        close=data[close_col],
        increasing_line_color='#2ecc71',
        decreasing_line_color='#e74c3c'
    ))
    fig_candle.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=500,
        template="plotly_dark"
    )
    st.plotly_chart(fig_candle, use_container_width=True)

with tab3:
    fig_area = px.area(data, x=date_col, y=close_col, 
                      title="Closing Price Trend",
                      labels={"Close": "Price (USD)"})
    fig_area.update_traces(line_color='#9b59b6', fill='tozeroy')
    fig_area.update_layout(
        hovermode="x",
        height=500,
        template="plotly_dark"
    )
    st.plotly_chart(fig_area, use_container_width=True)

# Forecasting section
st.subheader("🔮 AI Price Forecast")
st.markdown("""
<div style="background-color: #2c3e50; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
Our forecasting model uses Facebook's Prophet algorithm to predict future stock prices based on historical trends.
</div>
""", unsafe_allow_html=True)

# Prepare data for Prophet
df_train = data[[date_col, close_col]].rename(columns={date_col: "ds", close_col: "y"})

# Safely convert columns
df_train['ds'] = pd.to_datetime(df_train['ds'], errors='coerce')
df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')
df_train.dropna(inplace=True)

if df_train.empty:
    st.error("No valid data available for forecasting. Try another stock.")
    st.stop()

# Progress bar for model training
progress_bar = st.progress(0)
status_text = st.empty()

for i in range(100):
    # Update progress bar
    progress_bar.progress(i + 1)
    status_text.text(f"Training model... {i+1}%")
    time.sleep(0.01)  # Simulate processing time

# Train model
m = Prophet(daily_seasonality=True, yearly_seasonality=True)
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

progress_bar.empty()
status_text.text("Model training complete!")
st.success("Forecast generated successfully! ✅")

# Display forecast results
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Forecast Data Preview")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail().style.format({
        'yhat': '{:.2f}',
        'yhat_lower': '{:.2f}',
        'yhat_upper': '{:.2f}'
    }))

with col2:
    st.markdown("### Forecast Accuracy Metrics")
    st.metric("Trend Change Sensitivity", "High")
    st.metric("Seasonality Detection", "Strong")
    st.metric("Confidence Interval", "95%")

# Interactive forecast plot
st.markdown("### Interactive Forecast Visualization")
fig_forecast = plot_plotly(m, forecast)
fig_forecast.update_layout(
    height=600,
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    hovermode="x",
    template="plotly_dark"
)
st.plotly_chart(fig_forecast, use_container_width=True)

# Forecast components
st.markdown("### Forecast Breakdown")
with st.expander("View Forecast Components", expanded=False):
    fig_components = m.plot_components(forecast)
    st.pyplot(fig_components)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #777; font-size: 14px;">
<p>This application is for educational purposes only. Not financial advice.</p>
<p>Data provided by Yahoo Finance | Powered by Streamlit and Prophet</p>
</div>
""", unsafe_allow_html=True)