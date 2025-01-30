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

from torch.utils.data import Dataset, DataLoader
from datetime import date, timedelta, datetime

st.title('Stock Prediction App')

st.info('This is a stock prediction app')

ticker = st.text_input("Company's Ticker", placeholder ="Enter Ticker")
df = None

if ticker:  # Check if the user has entered a ticker
    try:
        # Download historical stock data for the entered ticker
        df = yf.download(ticker, period='max')
        
        # Display the downloaded DataFrame
        st.write(f"Data for {ticker}:")
        st.dataframe(df)
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.warning("Please enter a valid ticker symbol.")

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
    st.write(*scaled_test[:5])

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

    input_size = 1
    num_layers = 2
    hidden_size = 64
    output_size = 1

    # defining  the model, loss function, and optimizer
    model = LSTMModel(input_size, hidden_size, num_layers).to(device)

    loss_fn = torch.nn.MSELoss(reduction='mean')

    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 16
    # creating DataLoader for batch training
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    #creating Dataloader for batch testing
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    num_epochs = 50
    train_hist =[]
    test_hist =[]
    #training loop
    for epoch in range(num_epochs):
        total_loss = 0.0

        #training
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)
            loss=loss_fn(predictions, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

         # calculating average training loss and accuracy
        average_loss = total_loss /len(train_loader)
        train_hist.append(average_loss)

        model.eval()
        with torch.no_grad():
           total_test_loss = 0.0

           for batch_X_test, batch_y_test in test_loader:
               batch_X_test, batch_y_test = batch_X_test.to(device), batch_y_test.to(device)
               predictions_test = model(batch_X_test)
               test_loss = loss_fn(predictions_test, batch_y_test)

               total_test_loss += test_loss.item()
           #calculating average test loss and accuracy
           average_test_loss = total_test_loss / len(test_loader)
           test_hist.append(average_test_loss)
        if (epoch+1)%10==0:
          print(f'epoch[{epoch+1}/{num_epochs}] - Training Loss : {average_loss:.4f}, Test Loss : {average_test_loss:.4f}')

    x = np.linspace(1,num_epochs,num_epochs)
    plt.plot(x,train_hist,scalex= True, label="Training loss")
    plt.plot(x, test_hist, label="Test loss")
    plt.legend()
    plt.show()
