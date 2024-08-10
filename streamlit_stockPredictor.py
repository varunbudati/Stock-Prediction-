import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

def fetch_stock_data(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        if df.empty:
            st.error(f"No data available for {ticker}. Please check the ticker symbol.")
            return None, None
        return df, stock
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None, None

def calculate_technical_indicators(df):
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
    return df

def predict_future_prices(df, days=30, manual_volatility=None, manual_trend=None):
    X = np.array(range(len(df))).reshape(-1, 1)
    y = df['Close'].values
    model = LinearRegression()
    model.fit(X, y)
    
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
    future_X = np.array(range(len(df), len(df) + days)).reshape(-1, 1)
    future_prices = model.predict(future_X)
    
    if manual_volatility is not None:
        noise = np.random.normal(0, manual_volatility, days)
        future_prices += noise
    
    if manual_trend is not None:
        trend = np.linspace(0, manual_trend, days)
        future_prices += trend
    
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': future_prices
    })
    future_df.set_index('Date', inplace=True)
    return future_df

def plot_stock_data_with_predictions(df, future_df, future_df_manual=None):
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA20'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA50'))
    fig.add_trace(go.Scatter(x=future_df.index, y=future_df['Predicted_Close'], name='Auto Predictions', line=dict(color='red', dash='dash')))
    
    if future_df_manual is not None:
        fig.add_trace(go.Scatter(x=future_df_manual.index, y=future_df_manual['Predicted_Close'], name='Manual Predictions', line=dict(color='green', dash='dot')))
    
    fig.update_layout(title='Stock Price, Moving Averages, and Predictions', xaxis_title='Date', yaxis_title='Price')
    return fig

def calculate_volatility(df):
    return df['Volatility'].iloc[-1]

def predict_price_change(df, days, future_df):
    current_price = df['Close'].iloc[-1]
    predicted_price = future_df['Predicted_Close'].iloc[-1]
    percentage_change = ((predicted_price - current_price) / current_price) * 100
    return percentage_change

def analyze_company_future(stock):
    try:
        info = stock.info
        outlook = "The company's future looks "
        if 'recommendationKey' in info:
            if info['recommendationKey'] in ['buy', 'strongBuy']:
                outlook += "promising. Analysts recommend buying the stock. "
            elif info['recommendationKey'] in ['sell', 'strongSell']:
                outlook += "challenging. Analysts recommend selling the stock. "
            else:
                outlook += "stable. Analysts recommend holding the stock. "
        else:
            outlook += "uncertain based on available data. "
        
        if 'marketCap' in info:
            outlook += f"The company has a market cap of ${info['marketCap']:,}. "
        if 'trailingPE' in info:
            outlook += f"Its P/E ratio is {info['trailingPE']:.2f}. "
        if 'revenueGrowth' in info:
            outlook += f"Its revenue growth is {info['revenueGrowth']:.2%}."
        return outlook
    except Exception as e:
        return f"Unable to analyze company future due to an error: {str(e)}"

def analyze_competition(stock):
    try:
        info = stock.info
        competition = ""
        if 'sector' in info:
            competition += f"The company operates in the {info['sector']} sector. "
        if 'industry' in info:
            competition += f"Specifically, it's in the {info['industry']} industry. "
        
        if 'fullTimeEmployees' in info:
            competition += f"The company has {info['fullTimeEmployees']:,} full-time employees. "
        
        competition += "Competition level is considered "
        if info.get('industryDisruptors', "").lower() == 'yes':
            competition += "high due to industry disruptors. "
        else:
            competition += "moderate. "
        
        return competition
    except Exception as e:
        return f"Unable to analyze competition due to an error: {str(e)}"

def main():
    st.title('Advanced Stock Analysis and Prediction with Manual Input')
    
    ticker = st.sidebar.text_input('Stock Ticker', value='AAPL')
    start_date = st.sidebar.date_input('Start Date', datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input('End Date', datetime.now())
    
    df, stock = fetch_stock_data(ticker, start_date, end_date)
    
    if df is not None and stock is not None:
        df = calculate_technical_indicators(df)
        
        st.subheader('Stock Data')
        st.dataframe(df.tail())
        
        prediction_days = st.slider('Predict for next X days:', 1, 90, 30)
        
        # Manual input section
        st.subheader('Manual Input for Predictions')
        use_manual_input = st.checkbox('Use Manual Input')
        if use_manual_input:
            manual_volatility = st.slider('Manual Volatility (0-1):', 0.0, 1.0, 0.2, 0.01)
            manual_trend = st.slider('Manual Trend (-100 to 100):', -100.0, 100.0, 0.0, 1.0)
        else:
            manual_volatility = None
            manual_trend = None
        
        future_df_auto = predict_future_prices(df, prediction_days)
        future_df_manual = predict_future_prices(df, prediction_days, manual_volatility, manual_trend) if use_manual_input else None
        
        st.plotly_chart(plot_stock_data_with_predictions(df, future_df_auto, future_df_manual))
        
        st.subheader('Future Price Predictions')
        st.write("Automatic Predictions:")
        st.dataframe(future_df_auto)
        if use_manual_input:
            st.write("Manual Input Predictions:")
            st.dataframe(future_df_manual)
        
        st.subheader('Stock Analysis')
        volatility = calculate_volatility(df)
        st.write(f"Current Volatility: {volatility:.2%}")
        
        price_change_auto = predict_price_change(df, prediction_days, future_df_auto)
        st.write(f"Predicted price change (Auto) in the next {prediction_days} days: {price_change_auto:.2f}%")
        
        if use_manual_input:
            price_change_manual = predict_price_change(df, prediction_days, future_df_manual)
            st.write(f"Predicted price change (Manual) in the next {prediction_days} days: {price_change_manual:.2f}%")
        
        st.subheader('Company Future Analysis')
        st.write(analyze_company_future(stock))
        
        st.subheader('Competition Analysis')
        st.write(analyze_competition(stock))
    else:
        st.error("Unable to fetch stock data. Please try again later or check your input.")

if __name__ == '__main__':
    main()