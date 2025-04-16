#  Stock Prediction App 📈

A machine learning-powered web app for forecasting stock prices (worldwide).

## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://lyns-stockprediction.streamlit.app/)

## GitHub Codespaces

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/app-starter-kit?quickstart=1)

## Project Overview

This project aims to create an app based on an LSTM model,a type of recurrent neural network (RNN), which are well-suited for sequential data such as stock prices, that predicts the stock prices for S&P 500 and European companies. 

This repository contains code for:
- Downloading historical stock data from Yahoo Finance.
- Preprocessing and normalizing the data.
- Building and training LSTM models to forecast stock prices.
- Visualizing the prediction results against actual prices.

##  Features
- Data Acquisition: Automatically fetches historical stock data using yfinance.
- Data Preprocessing: Scales and formats data for time series modeling.
- LSTM Model: Implements an LSTM neural network for regression on stock price sequences.
- Prediction & Visualization: Generates future price predictions and plots them alongside actual data.
- Deployment for production on Streamlit

## Getting Started
![Capture d'écran 2025-04-16 180528](https://github.com/user-attachments/assets/d042dc82-dda1-47d4-be9f-ef25c84fd3de)

### Prerequisites

-   Python 3.6+
-   Libraries: pandas, scikit-learn, yfinance, matplotlib

- Clone the repository:
```
git clone https://github.com/Lynsoo/lyns-stockprediction.git
cd lyns-stockprediction
```

  Install dependencies with:
```
pip install -r requirements.txt
```
Or manually install:
```
pip install numpy pandas matplotlib scikit-learn tensorflow yfinance
```

- 


