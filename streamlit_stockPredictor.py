import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from scipy.stats import norm

def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    return df, stock

def calculate_technical_indicators(df):
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
    return df

def predict_future_prices(df, days=30):
    X = np.array(range(len(df))).reshape(-1, 1)
    y = df['Close'].values
    model = LinearRegression()
    model.fit(X, y)
    
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
    future_X = np.array(range(len(df), len(df) + days)).reshape(-1, 1)
    future_prices = model.predict(future_X)
    
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': future_prices
    })
    future_df.set_index('Date', inplace=True)
    return future_df

def plot_stock_data_with_predictions(df, future_df):
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA20'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA50'))
    fig.add_trace(go.Scatter(x=future_df.index, y=future_df['Predicted_Close'], name='Predictions', line=dict(color='red', dash='dash')))
    
    fig.update_layout(title='Stock Price, Moving Averages, and Predictions', xaxis_title='Date', yaxis_title='Price')
    return fig

def calculate_volatility(df):
    return df['Volatility'].iloc[-1]

def predict_price_change(df, days):
    current_price = df['Close'].iloc[-1]
    predicted_price = predict_future_prices(df, days)['Predicted_Close'].iloc[-1]
    percentage_change = ((predicted_price - current_price) / current_price) * 100
    return percentage_change

def analyze_company_future(stock):
    info = stock.info
    outlook = "The company's future looks "
    if info['recommendationKey'] in ['buy', 'strongBuy']:
        outlook += "promising. Analysts recommend buying the stock. "
    elif info['recommendationKey'] in ['sell', 'strongSell']:
        outlook += "challenging. Analysts recommend selling the stock. "
    else:
        outlook += "stable. Analysts recommend holding the stock. "
    
    outlook += f"The company has a market cap of ${info['marketCap']:,} and a P/E ratio of {info.get('trailingPE', 'N/A')}. "
    outlook += f"Its revenue growth is {info.get('revenueGrowth', 'N/A')}."
    return outlook

def analyze_competition(stock):
    info = stock.info
    sector = info['sector']
    industry = info['industry']
    competition = f"The company operates in the {sector} sector, specifically in the {industry} industry. "
    
    if 'companyOfficers' in info:
        competition += f"It has {len(info['companyOfficers'])} key executives. "
    
    if 'fullTimeEmployees' in info:
        competition += f"The company has {info['fullTimeEmployees']:,} full-time employees. "
    
    competition += "Competition level is considered "
    if info.get('industryDisruptors', "").lower() == 'yes':
        competition += "high due to industry disruptors. "
    else:
        competition += "moderate. "
    
    return competition

def main():
    st.title('Advanced Stock Analysis and Prediction')
    
    ticker = st.sidebar.text_input('Stock Ticker', value='AAPL')
    start_date = st.sidebar.date_input('Start Date', datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input('End Date', datetime.now())
    
    df, stock = fetch_stock_data(ticker, start_date, end_date)
    df = calculate_technical_indicators(df)
    
    future_df = predict_future_prices(df)
    
    st.subheader('Stock Data')
    st.dataframe(df.tail())
    
    st.plotly_chart(plot_stock_data_with_predictions(df, future_df))
    
    st.subheader('Future Price Predictions')
    st.dataframe(future_df)
    
    st.subheader('Stock Analysis')
    volatility = calculate_volatility(df)
    st.write(f"Current Volatility: {volatility:.2%}")
    
    prediction_days = st.slider('Predict price change for next X days:', 1, 90, 30)
    price_change = predict_price_change(df, prediction_days)
    st.write(f"Predicted price change in the next {prediction_days} days: {price_change:.2f}%")
    
    st.subheader('Company Future Analysis')
    st.write(analyze_company_future(stock))
    
    st.subheader('Competition Analysis')
    st.write(analyze_competition(stock))

if __name__ == '__main__':
    main()