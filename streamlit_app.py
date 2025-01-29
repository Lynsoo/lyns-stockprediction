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
df_ticker= [ticker]
df = yf.download( df_ticker, period = 'max')
