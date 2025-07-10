import streamlit as st
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from sklearn.linear_model import LinearRegression
from bs4 import BeautifulSoup
import requests
import datetime as dt

st.title("📈 AI-Powered Intraday Stock Analyzer")
symbols = st.text_input("Enter stock symbols (comma-separated, e.g., INFY.NS, TCS.NS):", "INFY.NS, TCS.NS")
symbol_list = [s.strip() for s in symbols.split(",")]

from sklearn.linear_model import LinearRegression
import numpy as np

def predict_price(df):
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)

    X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = df['Target'].values.reshape(-1)  # ✅ Flatten to 1D (avoid shape like (548, 1))

    model = LinearRegression()
    model.fit(X, y)

    latest_input = X.iloc[-1].values.reshape(1, -1)  # ✅ Reshape to (1, features)
    prediction = model.predict(latest_input)[0]
    return prediction


def fetch_news(stock):
    url = f"https://www.google.com/search?q={stock}+stock+news+site:moneycontrol.com&tbm=nws"
    headers = {"User-Agent": "Mozilla/5.0"}
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.text, 'html.parser')
    headlines = soup.select('h3')[:5]

    pos_words = ['buy', 'profit', 'upgrade', 'deal', 'growth']
    neg_words = ['loss', 'penalty', 'downgrade', 'fraud', 'decline']
    news_list = []

    for h in headlines:
        text = h.text.lower()
        sentiment = "Neutral"
        if any(w in text for w in pos_words):
            sentiment = "📈 Positive"
        elif any(w in text for w in neg_words):
            sentiment = "📉 Negative"
        news_list.append((h.text, sentiment))
    return news_list

for symbol in symbol_list:
    try:
        st.subheader(f"📊 {symbol}")
        today = dt.date.today()
        start = today - dt.timedelta(days=30)

        df = yf.download(symbol, start=start, end=today, interval="15m")
        df.dropna(inplace=True)

        info = yf.Ticker(symbol).info
        current = info.get('currentPrice')
        high_52 = info.get('fiftyTwoWeekHigh')
        low_52 = info.get('fiftyTwoWeekLow')
        volume = info.get('volume')

        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low']) / 2).cumsum() / df['Volume'].cumsum()

        st.write(f"**Current Price:** ₹{current}")
        st.write(f"**52-Week Range:** ₹{low_52} - ₹{high_52}")
        st.write(f"**Volume:** {volume}")

        latest_rsi = df['RSI'].iloc[-1]
        vwap = df['VWAP'].iloc[-1]
        pred_price = predict_price(df)
        delta = ((pred_price - df['Close'].iloc[-1]) / df['Close'].iloc[-1]) * 100

        st.write(f"**RSI(14):** {latest_rsi:.2f} → {'🔽 Oversold' if latest_rsi < 30 else '🔼 Overbought' if latest_rsi > 70 else '✅ Normal'}")
        st.write(f"**VWAP vs Price:** ₹{vwap:.2f} → {'Above VWAP ✅' if current > vwap else 'Below VWAP ⚠️'}")
        st.write(f"**Predicted Next Price (15-min):** ₹{pred_price:.2f} ({delta:.2f}%)")

        # News
        st.write("📰 **Recent News (Sentiment):**")
        news = fetch_news(symbol)
        for i, (headline, sentiment) in enumerate(news, 1):
            st.markdown(f"{i}. {headline} — *{sentiment}*")

    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
