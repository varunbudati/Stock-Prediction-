import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import random

def generate_mock_stock_data(ticker, start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date)
    close_prices = [100]  # Start with a base price of 100
    for _ in range(1, len(date_range)):
        change = random.uniform(-2, 2)  # Random daily change between -2% and 2%
        close_prices.append(close_prices[-1] * (1 + change/100))
    
    data = {
        'Date': date_range,
        'Open': close_prices,
        'High': [price * random.uniform(1, 1.02) for price in close_prices],
        'Low': [price * random.uniform(0.98, 1) for price in close_prices],
        'Close': close_prices,
        'Volume': [random.randint(1000000, 10000000) for _ in range(len(date_range))]
    }
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    return df

def calculate_technical_indicators(df):
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    return df

def predict_future_prices(df, days=30):
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
    
    predictions = []
    current_price = df['Close'].iloc[-1]
    sma20 = df['SMA20'].iloc[-1]
    sma50 = df['SMA50'].iloc[-1]
    
    for _ in range(days):
        if sma20 > sma50:
            prediction = current_price * 1.01  # Predict 1% increase
        else:
            prediction = current_price * 0.99  # Predict 1% decrease
        
        predictions.append(prediction)
        current_price = prediction
        
        # Update SMAs (this is a simplification)
        sma20 = (sma20 * 19 + current_price) / 20
        sma50 = (sma50 * 49 + current_price) / 50
    
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': predictions
    })
    future_df.set_index('Date', inplace=True)
    return future_df

def plot_stock_data_with_predictions(df, future_df):
    fig = go.Figure()
    
    # Plot historical data
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA20'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA50'))
    
    # Plot predictions
    fig.add_trace(go.Scatter(x=future_df.index, y=future_df['Predicted_Close'], name='Predictions', line=dict(color='red', dash='dash')))
    
    fig.update_layout(title='Stock Price, Moving Averages, and Predictions', xaxis_title='Date', yaxis_title='Price')
    return fig

def generate_mock_news(ticker):
    news_templates = [
        f"{ticker} stock sees unexpected surge",
        f"Analysts predict bright future for {ticker}",
        f"{ticker} announces new product line",
        f"Market uncertainty affects {ticker} performance",
        f"{ticker} reports quarterly earnings"
    ]
    return [{'title': random.choice(news_templates)} for _ in range(5)]

def simple_sentiment_analysis(text):
    positive_words = ['surge', 'bright', 'new', 'earnings']
    negative_words = ['uncertainty', 'affects']
    
    words = text.lower().split()
    positive_count = sum(word in positive_words for word in words)
    negative_count = sum(word in negative_words for word in words)
    
    if positive_count > negative_count:
        return 1
    elif negative_count > positive_count:
        return -1
    else:
        return 0

def main():
    st.title('Stock Analysis and Prediction (Mock Data)')
    
    ticker = st.sidebar.text_input('Stock Ticker', value='MOCK')
    start_date = st.sidebar.date_input('Start Date', datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input('End Date', datetime.now())
    
    df = generate_mock_stock_data(ticker, start_date, end_date)
    df = calculate_technical_indicators(df)
    
    future_df = predict_future_prices(df)
    
    st.subheader('Stock Data (Mock)')
    st.dataframe(df.tail())
    
    st.plotly_chart(plot_stock_data_with_predictions(df, future_df))
    
    st.subheader('Future Price Predictions')
    st.dataframe(future_df)
    
    # Simple Sentiment Analysis
    st.subheader('News Sentiment Analysis (Mock)')
    news = generate_mock_news(ticker)
    if news:
        sentiments = [simple_sentiment_analysis(article['title']) for article in news]
        avg_sentiment = sum(sentiments) / len(sentiments)
        st.write(f"Average sentiment score: {avg_sentiment:.2f}")
        st.write("Recent news headlines (Mock):")
        for article in news:
            st.write(f"- {article['title']}")
    else:
        st.write("No recent news found.")

if __name__ == '__main__':
    main()