import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import requests

# Download NLTK data (run this once)
nltk.download('vader_lexicon')

def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def calculate_technical_indicators(df):
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def plot_stock_data(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA20'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA50'))
    fig.update_layout(title='Stock Price and Moving Averages', xaxis_title='Date', yaxis_title='Price')
    return fig

def prepare_data_for_model(df):
    df['Date'] = df.index
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Weekday'] = df['Date'].dt.weekday
    
    features = ['Open', 'High', 'Low', 'Volume', 'SMA20', 'SMA50', 'RSI', 'Day', 'Month', 'Year', 'Weekday']
    X = df[features]
    y = df['Close']
    
    return X, y

def train_random_forest_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_next_day(model, last_data):
    return model.predict(last_data)

def fetch_news(ticker):
    # This is a placeholder. In a real app, you'd use a proper news API
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey=YOUR_API_KEY"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['articles']
    return []

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)['compound']

def main():
    st.title('NVIDIA Stock Analysis and Prediction')
    
    ticker = st.sidebar.text_input('Stock Ticker', value='NVDA')
    start_date = st.sidebar.date_input('Start Date', datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input('End Date', datetime.now())
    
    df = fetch_stock_data(ticker, start_date, end_date)
    df = calculate_technical_indicators(df)
    
    st.subheader('Stock Data')
    st.dataframe(df.tail())
    
    st.plotly_chart(plot_stock_data(df))
    
    st.subheader('Relative Strength Index (RSI)')
    st.line_chart(df['RSI'])
    
    # Random Forest Prediction
    st.subheader('Stock Price Prediction')
    if st.button('Train Model and Predict'):
        X, y = prepare_data_for_model(df)
        model = train_random_forest_model(X, y)
        last_data = X.iloc[-1:].copy()
        last_data['Day'] += 1  # Predict for the next day
        next_day_price = predict_next_day(model, last_data)
        st.write(f"Predicted price for next trading day: ${next_day_price[0]:.2f}")
    
    # News Sentiment Analysis
    st.subheader('News Sentiment Analysis')
    news = fetch_news(ticker)
    if news:
        sentiments = [analyze_sentiment(article['title'] + ' ' + article['description']) for article in news[:5]]
        avg_sentiment = sum(sentiments) / len(sentiments)
        st.write(f"Average sentiment score: {avg_sentiment:.2f}")
        st.write("Recent news headlines:")
        for article in news[:5]:
            st.write(f"- {article['title']}")
    else:
        st.write("No recent news found.")

if __name__ == '__main__':
    main()