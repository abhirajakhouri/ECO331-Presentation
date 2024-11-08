import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define stock tickers and date range
tickers = ['GOOGL', 'MSFT', 'NVDA', 'BAC', 'AMZN']
start_date = '2009-04-05'
end_date = '2019-04-05'

# Download the data for all tickers
data = yf.download(tickers, start=start_date, end=end_date)
stock_predictions = {}

# Define training and testing periods
train_start = '2009-04-06'
train_end = '2017-04-03'
test_start = '2017-04-04'
test_end = '2019-04-05'

for ticker in tickers:
    # Extract relevant data and calculate features for each stock
    df = pd.DataFrame(index=data['Adj Close'].index)
    df['Close'] = data['Close'][ticker]
    df['H-L'] = data['High'][ticker] - data['Low'][ticker]
    df['O-C'] = data['Close'][ticker] - data['Open'][ticker]
    df['7DAYSMA'] = data['Adj Close'][ticker].rolling(window=7).mean()
    df['14DAYSMA'] = data['Adj Close'][ticker].rolling(window=14).mean()
    df['21DAYSMA'] = data['Adj Close'][ticker].rolling(window=21).mean()
    df['7DAYSSTDDEV'] = data['Adj Close'][ticker].rolling(window=7).std()
    
    # Adding lag features (previous day's closing price)
    df['Lag1'] = df['Close'].shift(1)
    df['Lag2'] = df['Close'].shift(2)
    df.dropna(inplace=True)  # Remove rows with NaN values due to rolling and lag features
    
    # Define the features (X) and target variable (y)
    X = df[['H-L', 'O-C', '7DAYSMA', '14DAYSMA', '21DAYSMA', '7DAYSSTDDEV', 'Lag1', 'Lag2']]
    y = df['Close']
    
    # Split data based on specified date ranges
    X_train = X.loc[train_start:train_end]
    y_train = y.loc[train_start:train_end]
    X_test = X.loc[test_start:test_end]
    y_test = y.loc[test_start:test_end]
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build and compile the ANN model
    model = Sequential()
    model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Output layer for regression

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
    
    # Predict closing prices with the trained model
    y_pred = model.predict(X_test_scaled)
    
    # Store actual and predicted prices for each stock
    stock_predictions[ticker] = {
        'Actual': y_test,
        'Predicted': y_pred.flatten()
    }
    
    # Calculate and print error metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # Convert to percentage
    mbe = np.mean(y_pred.flatten() - y_test)  # Mean Bias Error
    print(f'{ticker} RMSE: {rmse:.2f}')
    print(f'{ticker} MAPE: {mape:.2f}%')
    print(f'{ticker} MBE: {mbe:.2f}')

# Plot the actual vs predicted closing prices for each stock
plt.figure(figsize=(15, 10))
for i, ticker in enumerate(tickers, 1):
    plt.subplot(3, 2, i)
    plt.plot(stock_predictions[ticker]['Actual'].index, stock_predictions[ticker]['Actual'], label='Actual', color='blue')
    plt.plot(stock_predictions[ticker]['Actual'].index, stock_predictions[ticker]['Predicted'], label='Predicted', color='orange')
    plt.title(f'{ticker} Closing Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

plt.tight_layout()
plt.show()
