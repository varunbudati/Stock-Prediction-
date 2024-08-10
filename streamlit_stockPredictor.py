import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
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

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)

def train_lstm_model(data, seq_length=60, epochs=50, batch_size=32):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = create_sequences(scaled_data, seq_length), scaled_data[seq_length:]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(50, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    
    return model, scaler

def predict_next_day(model, data, scaler, seq_length=60):
    last_sequence = data[-seq_length:].reshape(1, seq_length, 1)
    predicted = model.predict(last_sequence)
    return scaler.inverse_transform(predicted)[0, 0]

def fetch_news(ticker):
    # Find and use a proper news API
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
    
    # LSTM Prediction
    st.subheader('Stock Price Prediction')
    if st.button('Train Model and Predict'):
        model, scaler = train_lstm_model(df['Close'].values)
        next_day_price = predict_next_day(model, df['Close'].values, scaler)
        st.write(f"Predicted price for next trading day: ${next_day_price:.2f}")
    
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