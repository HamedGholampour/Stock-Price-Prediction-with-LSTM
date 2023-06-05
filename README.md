# Stock Price Prediction with LSTM

## Description
This project utilizes LSTM (Long Short-Term Memory) neural networks to forecast the upcoming closing prices of a stock by analyzing past data. The Keras library is used for constructing and training the LSTM model. The project acquires historical stock data from Yahoo Finance with the assistance of the yfinance and pandas_datareader libraries. The data is preprocessed and normalized using the MinMaxScaler function from the scikit-learn library. The Sequential model from Keras is employed to construct and train the LSTM model, with different architectures and configurations being investigated. The model creates a new Long Short-Term Memory (LSTM) model with five layers. The model architecture consists of three LSTM layers with decreasing sizes (128 units, 64 units, and 32 units), followed by two dense layers (with 16 units and 1 unit respectively). After each LSTM layer and the two dense layers, a dropout of 0.2 is applied, which helps prevent overfitting.

## Introduction
Stock Price Prediction with LSTM is a project that aims to forecast the closing prices of a given stock based on historical data. By utilizing LSTM neural networks, which are capable of capturing long-term dependencies, the project provides valuable insights for investors and traders in making informed decisions.

## Dependencies
To run this project, the following libraries are required:
- `math`
- `pandas` (imported as `pd`)
- `pandas_datareader` (imported as `pdr`)
- `yfinance` (imported as `yfin`)
- `numpy` (imported as `np`)
- `scikit-learn` (imported as `sklearn`)
- `keras.models` (imported as `Sequential`)
- `keras.layers` (imported as `Dense`, `LSTM`, `Dropout`)
- `matplotlib.pyplot` (imported as `plt`)
- `mplfinance` (imported as `mplf`)
- `keras` (imported as `regularizers`)

To install these dependencies, you can use the following code:

```python
import math
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yfin
yfin.pdr_override()
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import mplfinance as mplf
from keras.layers import Dropout
from keras import regularizers




## Dependencies
The project follows the following steps:

Data Retrieval: Get historical stock data from Yahoo Finance for Shell stock.
Data Preprocessing: Clean the data and extract the desired features.
Data Scaling: Normalize the data using the MinMaxScaler function.
Data Preparation: Split the data into training and testing sets and create sequences of input and output data.
LSTM Model Building: Create and train the LSTM model using three LSTM layers with decreasing sizes and followed by two dense layers.
Prediction: Make predictions on the test data and evaluate the model's performance.
Visualization: Plot the actual and predicted prices to visualize the results.
Please refer to the code implementation for detailed instructions on each step.

