import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import math
import yfinance as yf
import lxml

from torch.utils.data import Dataset, DataLoader
from datetime import date, timedelta, datetime

st.title('Stock Prediction App')

st.info('This is a stock prediction app')

# Loading Euronext data with streamlit caching
@st.cache_data
def load_euronext_data():
    euronext = pd.read_csv('eurotickers.csv', sep=';')
    return euronext[['Name', 'Symbol']]

euronext_f = load_euronext_data()

# Loading S&P 500 data with caching
@st.cache_data(ttl=7 * 24 * 60 * 60)
def load_sp500_data():
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    sp500_stocks = table[0]
    tickers = sp500_stocks['Symbol'].tolist()
    companies = sp500_stocks['Security'].tolist()
    dictionary = pd.DataFrame({
        'Name': companies,
        'Symbol': tickers,
    })
    return dictionary

dictionary = load_sp500_data()

@st.cache_data(ttl=7 * 24 * 60 * 60)
def combine_data(euronext_f, dictionary):
    combined_data = pd.concat([dictionary, euronext_f], ignore_index=True)
    return combined_data

combined_data = combine_data(euronext_f, dictionary)

company= st.text_input("Company Name", placeholder ="Enter Company")
df = None

if company:  # Checking if the user has entered a ticker
    try:
        matching_tickers = combined_data[combined_data['Name'].str.contains(company, case=False)]
        if not matching_tickers.empty:
            # Extract the ticker symbol(s) from the matching row(s)
            tickers = matching_tickers['Symbol'].tolist()
            
            # Handle multiple matches if needed
            if len(tickers) > 1:
                st.write(f"Multiple tickers found for '{company}': {tickers}")
                # You can either select one ticker or handle them all
                ticker = tickers[0]  #using the first match
            else:
                ticker = tickers[0]
            # Downloading historical stock data for the entered ticker
            df = yf.download(ticker, period = '10y')
        
            # Downloaded DataFrame Display
            st.write(f"Data for {ticker}:")
            st.dataframe(df)
        
    except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.warning("Please enter a valid company.")

if df is not None and not df.empty:
    training_data_len = math.ceil(len(df)*.8)

    train_data = df[:training_data_len][[('Open', ticker)]]
    test_data = df[training_data_len:][[('Open', ticker)]]
    
    try :
      dataset_test = test_data.Open.values
    except :
      dataset_test = test_data[('Open', ticker)].values
    #1D to 2D
    dataset_test = np.reshape(dataset_test, (-1,1))

    dataset_train = train_data.Open.values
    dataset_train = np.reshape(dataset_train, (-1,1))


    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_train = scaler.fit_transform(dataset_train)
    scaled_test = scaler.transform(dataset_test)
 
    sequence_length = 50 # nb of time steps to look back
    X_train, y_train = [], []
    for i in range(len(scaled_train) - sequence_length):
        X_train.append(scaled_train[i:i+sequence_length])
        y_train.append(scaled_train[i+1:i+sequence_length+1])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    
    sequence_length = 30 #nb of time steps to look back
    X_test, y_test = [], []
    for i in range(len(scaled_test) - sequence_length):
        X_test.append(scaled_test[i:i+sequence_length])
        y_test.append(scaled_test[i+1:i+sequence_length+1])
    X_test, y_test = np.array(X_test), np.array(y_test)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    info = st.write('Please wait a few seconds...')

    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
          super(LSTMModel,self).__init__() #initializes the parent class nn.Module
          self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
          self.linear = nn.Linear(hidden_size, 1)

        def forward(self, x): #defines forward pass of the neural network
          out, _ =self.lstm(x)
          out = self.linear(out)
          return out
    device = torch.device('cpu')
    
    @st.cache_resource(ttl=7 * 24 * 60 * 60)
    def load_model(input_size, hidden_size, num_layers):
        return LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    input_size = 1
    num_layers = 2
    hidden_size = 64
    output_size = 1
   
    model = load_model(input_size, hidden_size, num_layers).to(device)
    model.load_state_dict(torch.load('pretrained_lstm_model.pth'))
    model.eval()

    
    num_forecast_steps = 30

    # converting to NumPy and remove singleton dimensions
    sequence_to_plot = X_test.squeeze().cpu().numpy()

    # use the last 30 data points as the starting point
    historical_data = sequence_to_plot[-1]

    # initializing a list to store the forecasted value
    forecasted_values = []
    
    # using the trained model to forecast the future values
    with torch.no_grad():
        for _ in range(num_forecast_steps*2):
            # converting the historical data to a tensor and add an extra dimension
            historical_data_tensor = torch.as_tensor(historical_data).view(1, -1, 1).float().to(device)

            # prediction for the next time step
            predicted_value = model(historical_data_tensor).cpu().numpy()[0,0]
            forecasted_values.append(predicted_value)

            # updating the historical data with the predicted value
            historical_data = np.roll(historical_data, -1)
            historical_data[-1] = predicted_value

    # Generating future dates
    last_date = test_data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(1), periods=30)
    combined_index = test_data.index.append(future_dates)

    plt.rcParams['figure.figsize']=[14, 4]

    #Test Data
    plt.plot(test_data.index[-100: -30], test_data.Open[-100:-30], label= "test_data", color = "b")
    #reversing the scaling transformation
    original_cases = scaler.inverse_transform(np.expand_dims(sequence_to_plot[-1], axis=0)).flatten()

    #the historical data used as input for forecasting
    plt.plot(test_data.index[-30:], original_cases, label='actual values', color='green')

    #Forecasted Values
    #reversing the scaling transformation
    forecasted_cases = scaler.inverse_transform(np.array(forecasted_values).reshape(-1, 1)).flatten()
    # plotting the forecasted values
    plt.plot(combined_index[-60:], forecasted_cases, label='forecasted values', color='red')

    st.write('Forecasted Values Chart for ' + ticker )

    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Time Series Forecasting')
    plt.grid(True)
    st.pyplot(plt)
