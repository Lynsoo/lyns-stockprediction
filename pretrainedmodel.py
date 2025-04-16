import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import pandas as pd

import matplotlib.dates as mdates

import math
import plotly.express as px

from torch.utils.data import Dataset, DataLoader
import yfinance as yf #importing yahoo finance
from datetime import date, timedelta, datetime

euronext = pd.read_csv('/teamspace/studios/this_studio/stockpred/eurotickers.csv', sep=';')
euronext_f = euronext[['Name', 'Symbol']]

# fetching S&P 500 stocks 
table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
sp500_stocks = table[0]

# extracting tickers and company names
tickers = sp500_stocks['Symbol'].tolist()
companies = sp500_stocks['Security'].tolist()

# creating a dataframe
dictionary = pd.DataFrame({
    'Name': companies,
    'Symbol': tickers,
})

# saving to CSV
tickerlist = dictionary.to_dict(orient='records')# fetching S&P 500 stocks 
combined_data= pd.concat([dictionary, euronext_f], ignore_index=True)
print(combined_data)

list = ['NVDA', 'AMZN', 'AAPL', 'AC', 'GM', 'COST', 'DELL', 'FFF', 'ACA', 'RMS','JNJ', 'OR', 'IAM', 'MSFT', 'NFLX',
    'SNAP', 'ORA', 'ORCL', 'MBG', 'AAL', 'DAL', 'TSLA', 'GOOG', 'HPQ', 'MC', 'IBM', 'CSCO', 'INTC', 'KO', 'META', 'AMD', 'AVGO', 'MELI']
for i in list : 
    df = yf.download(i, period = '10y')
    training_data_len = math.ceil(len(df)*.8)
    train_data = df[:training_data_len][[('Open', i)]]
    test_data = df[training_data_len:][[('Open', i)]]
    try :
        dataset_test = test_data.Open.values
    except :
        dataset_test = test_data[('Open', i)].values
    #1D to 2D
    dataset_test = np.reshape(dataset_test, (-1,1))

    dataset_train = train_data.Open.values
    dataset_train = np.reshape(dataset_train, (-1,1))

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))

    # scaling dataset
    scaled_train= scaler.fit_transform(dataset_train)
    scaled_test = scaler.fit_transform(dataset_test)
    sequence_length = 30 # nb of time steps to look back
    X_train, y_train = [], []
    for i in range(len(scaled_train) - sequence_length):
        X_train.append(scaled_train[i:i+sequence_length])
        y_train.append(scaled_train[i+1:i+sequence_length+1])
    X_train, y_train = np.array(X_train), np.array(y_train)

#converting data to Pytorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)    
    sequence_length = 30 #nb of time steps to look back
    X_test, y_test = [], []
    for i in range(len(scaled_test) - sequence_length):
        X_test.append(scaled_test[i:i+sequence_length])
        y_test.append(scaled_test[i+1:i+sequence_length+1])
    X_test, y_test = np.array(X_test), np.array(y_test)

#converting data to Pytorch tensors
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)  
    class LSTMModel(nn.Module):
      #input_size : number of features in input at each time step
      #hidden_size : number of LSTM units
      #num_layers : number of LSTM layers
        def __init__(self, input_size, hidden_size, num_layers):
            super(LSTMModel,self).__init__() #initializes the parent class nn.Module
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.linear = nn.Linear(hidden_size, 1)

        def forward(self, x): #defines forward pass of the neural network
            out, _ =self.lstm(x)
            out = self.linear(out)
            return out

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = 1
    num_layers = 2
    hidden_size = 64
    output_size = 1

# defining  the model + loss function + optimizer
    model = LSTMModel(input_size, hidden_size, num_layers).to(device)

    loss_fn = torch.nn.MSELoss(reduction='mean')

    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 32
    # creating DataLoader for batch training
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    #creating Dataloader for batch testing
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    num_epochs = 100
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

    # validation on test data
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

torch.save(model.state_dict(), 'stock_prediction_model.pth')