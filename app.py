import streamlit as st
import pandas as pd
import numpy as np
import torch
import timesfm
import plotly.graph_objects as go
from datetime import datetime, timedelta
from stock_data_loader import get_stock_data, prepare_for_timesfm, search_stock_tickers

import glob

# Page Config
st.set_page_config(page_title="TimesFM Stock Predictor", layout="wide")

def get_latest_scan_results():
    """Find the most recent full market scan CSV."""
    files = glob.glob("full_market_scan_*.csv")
    if not files:
        return None
    # Sort by filename (which includes timestamp)
    latest_file = sorted(files)[-1]
    try:
        df = pd.read_csv(latest_file)
        return df, latest_file
    except Exception:
        return None, None

# Cache the model to avoid reloading on every interaction
@st.cache_resource
def load_model():
    print("Loading TimesFM model...")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    return model

def main():
    st.title("📈 TimesFM Stock Trend Predictor")
    st.markdown("""
    This app uses **Google Research's TimesFM 2.5** (Time Series Foundation Model) to predict future stock price trends.
    """)

    # Sidebar Settings
    st.sidebar.header("Search & Settings")
    
    # --- Screener Results Section ---
    scan_data, scan_file = get_latest_scan_results()
    selected_from_screener = None
    
    if scan_data is not None:
        st.sidebar.subheader("🔥 Top Screener Candidates")
        st.sidebar.info(f"Loaded results from: {scan_file}")
        
        # Sort by Growth, Confidence, or Name
        sort_by = st.sidebar.radio("Sort candidates by:", ["Growth %", "Confidence Score", "Name (A-Z)"], horizontal=True)
        
        if sort_by == "Name (A-Z)":
            scan_data_sorted = scan_data.sort_values(by="Ticker", ascending=True)
        else:
            scan_data_sorted = scan_data.sort_values(by=sort_by, ascending=False)
        
        # Create display labels
        scan_data_sorted['label'] = scan_data_sorted.apply(
            lambda x: f"{x['Ticker']} (+{x['Growth %']}% | Conf: {x['Confidence Score']})", axis=1
        )
        
        selected_cand = st.sidebar.selectbox(
            "Quick Select (Top 10% Growth):",
            options=["None"] + scan_data_sorted['label'].tolist()
        )
        
        if selected_cand != "None":
            selected_ticker = selected_cand.split(" ")[0]
            selected_from_screener = scan_data_sorted[scan_data_sorted['Ticker'] == selected_ticker].iloc[0]

    st.sidebar.divider()
    
    # Search Ticker
    default_ticker = selected_from_screener['Ticker'] if selected_from_screener is not None else "NVDA"
    search_query = st.sidebar.text_input("Search Stock (e.g., '2330' or 'NVDA')", value=default_ticker)
    
    suggestions = search_stock_tickers(search_query)
    
    if suggestions:
        selected_option = st.sidebar.selectbox(
            "Select exact ticker from results:",
            options=suggestions,
            format_func=lambda x: x['display']
        )
        ticker = selected_option['symbol']
    else:
        st.sidebar.warning("No suggestions found. Using raw input.")
        ticker = search_query.upper()

    st.sidebar.divider()
    forecast_horizon = st.sidebar.slider("Forecast Horizon (Days)", min_value=1, max_value=30, value=14)
    history_period = st.sidebar.selectbox("Historical Context", options=["6mo", "1y", "2y", "5y"], index=1)
    
    if st.sidebar.button("Run Prediction"):
        if selected_from_screener is not None:
            st.success(f"### Screener Insight for {ticker}")
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Predicted Growth", f"+{selected_from_screener['Growth %']}%")
            sc2.metric("Confidence Score", f"{selected_from_screener['Confidence Score']}/100")
            sc3.metric("Safety Check", selected_from_screener['Safety Check'])
            st.divider()

        with st.spinner(f"Fetching data and predicting for {ticker}..."):
            # 1. Fetch Data
            df = get_stock_data(ticker, period=history_period, interval="1d")
            
            if df is not None:
                # 2. Run Model
                model = load_model()
                
                # Compile config
                model.compile(
                    timesfm.ForecastConfig(
                        max_context=1024,
                        max_horizon=forecast_horizon + 10,
                        normalize_inputs=True,
                        use_continuous_quantile_head=True,
                        force_flip_invariance=True,
                        infer_is_positive=True,
                        fix_quantile_crossing=True,
                    )
                )
                
                context_data = prepare_for_timesfm(df)
                
                # Inference
                point_forecast, quantile_forecast = model.forecast(
                    horizon=forecast_horizon,
                    inputs=[context_data],
                )
                
                points = point_forecast[0]
                quantiles = quantile_forecast[0]
                
                # 3. Process Dates
                last_date = df.index[-1]
                # Filter out timezone info for easier plotting if necessary
                last_date_naive = last_date.replace(tzinfo=None)
                forecast_dates = [last_date_naive + timedelta(days=i+1) for i in range(forecast_horizon)]
                
                # 4. Visualization with Plotly
                fig = go.Figure()

                # Historical Line (last 90 days)
                show_hist_days = 90
                hist_subset = df.iloc[-show_hist_days:]
                fig.add_trace(go.Scatter(
                    x=hist_subset.index, 
                    y=hist_subset['Close'],
                    mode='lines',
                    name='Historical Close',
                    line=dict(color='royalblue', width=2)
                ))

                # Forecast Line
                fig.add_trace(go.Scatter(
                    x=forecast_dates, 
                    y=points,
                    mode='lines+markers',
                    name='TimesFM Forecast',
                    line=dict(color='firebrick', width=3, dash='dash')
                ))

                # Confidence Interval (Quantiles)
                # 10th and 90th quantiles for shaded area
                low_bound = quantiles[:, 1]
                high_bound = quantiles[:, 9]
                
                fig.add_trace(go.Scatter(
                    x=forecast_dates + forecast_dates[::-1],
                    y=list(high_bound) + list(low_bound)[::-1],
                    fill='toself',
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    line=dict(color='rgba(255, 0, 0, 0)'),
                    hoverinfo="skip",
                    showlegend=True,
                    name='80% Confidence Interval'
                ))

                fig.update_layout(
                    title=f"{ticker} Price Forecast (Next {forecast_horizon} Days)",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    template="plotly_white",
                    height=600,
                    hovermode="x unified"
                )

                st.plotly_chart(fig, use_container_width=True)
                
                # Metrics / Table
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Price Metrics")
                    latest_price = df['Close'].iloc[-1]
                    end_forecast = points[-1]
                    change = ((end_forecast - latest_price) / latest_price) * 100
                    
                    st.metric("Latest Price", f"${latest_price:.2f}")
                    st.metric(f"Forecasted Price ({forecast_horizon}d)", f"${end_forecast:.2f}", f"{change:+.2f}%")

                with col2:
                    st.subheader("Forecast Data")
                    forecast_df = pd.DataFrame({
                        "Date": [d.strftime('%Y-%m-%d') for d in forecast_dates],
                        "Predicted Price": points
                    })
                    st.dataframe(forecast_df, height=300)
            else:
                st.error(f"Could not find data for {ticker}. Please check the symbol.")

    else:
        st.info("Enter a stock ticker and click 'Run Prediction' to see the trend.")

if __name__ == "__main__":
    main()
