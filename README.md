# Time-Series-Analysis-Of-Stock-Market
Overview:
This project focuses on time series analysis of stock market data, specifically predicting the closing prices of Apple Inc. (AAPL) stocks over time. Using historical stock data, the project demonstrates how to preprocess data, create moving averages, and implement machine learning models for time series forecasting using Long Short-Term Memory (LSTM) neural networks. The goal is to forecast future stock prices based on past performance.

Project Highlights:
Data Collection: Stock price data is fetched for Apple Inc. from January 1, 2010, to December 31, 2019, using the yfinance API.
Data Preprocessing: The data is cleaned, indexed, and normalized to prepare it for machine learning models. Moving averages (100-day and 200-day) are calculated for trend analysis.
Machine Learning Model: The project employs a deep learning model (LSTM) to predict future stock prices. The model is trained on the historical data (70% of the dataset) and tested on the remaining 30%.
Prediction Visualization: Finally, the project compares the predicted prices with the actual stock prices to evaluate the model's performance.
Project Workflow
1. Data Collection
Using the yfinance package, historical stock data for Apple Inc. (AAPL) is collected from January 2010 to December 2019. The data includes fields like Open, High, Low, Close, Volume, and Adj Close.

python code:
import yfinance as yf

start = '2010-01-01'
end = '2019-12-31'

df = yf.download('AAPL', start=start, end=end)
2. Data Preprocessing:
The raw data is preprocessed to prepare it for analysis:

Resetting Index: The date is used as an index for easy time-based operations.
Dropping Columns: Unnecessary columns (Date and Adj Close) are removed to focus on the Close price.
Rolling Averages: A 100-day and 200-day moving average (MA) are calculated to smooth the data and identify long-term trends.
python code
df = df.reset_index()
df = df.drop(['Date', 'Adj Close'], axis=1)

# Calculate 100-day and 200-day Moving Averages
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
3. Exploratory Data Analysis (EDA)
The Close price is plotted, along with the 100-day and 200-day moving averages, to visually inspect the trends and volatility in the stock prices. This step helps in understanding the overall trend before diving into predictions.

python code:
plt.figure(figsize=(12,6))
plt.plot(df.Close, label='Closing Price')
plt.plot(ma100, 'r', label='100-day MA')
plt.plot(ma200, 'g', label='200-day MA')
plt.legend()
plt.show()
4. Splitting the Data:
The dataset is split into training and testing sets, with 70% of the data used for training the model and the remaining 30% for testing.

python code:
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])
5. Feature Scaling:
To ensure the data is well-suited for LSTM model training, the Close prices are normalized between 0 and 1 using MinMaxScaler to improve convergence speed.

python code:
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

data_training_scaled = scaler.fit_transform(data_training)
6. Preparing Training Data for LSTM
The model uses sequences of 100 time steps to predict the next value. The input (x_train) is created by sliding a 100-time-step window over the normalized data. The output (y_train) is the stock price at the next time step.

python code:
x_train = []
y_train = []
for i in range(100, data_training_scaled.shape[0]):
    x_train.append(data_training_scaled[i-100:i])
    y_train.append(data_training_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
7. LSTM Model Architecture
A sequential LSTM model is built using Keras. It consists of four LSTM layers with dropout to avoid overfitting. The model predicts a single value (the stock price for the next time step) using the dense output layer.

python code:
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

model = Sequential()

model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(120, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1))  # Final prediction layer
8. Model Compilation and Training
The model is compiled using the Adam optimizer and the mean_squared_error loss function. It is trained for 100 epochs on the training data.

python code:
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=100)
9. Testing and Making Predictions
The testing data is scaled and used to make predictions. The predicted prices are then transformed back to the original scale for comparison with the actual prices.

python code:
last_100_days = data_training.tail(100)
final_df = pd.concat([last_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

# Creating test data sequences
x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Making predictions
y_predicted = model.predict(x_test)
10. Results and Visualization
The predicted prices and the actual stock prices are plotted together to visually inspect the model’s performance.

python code:
plt.figure(figsize=(12,6))
plt.plot(y_test, label='Actual Price')
plt.plot(y_predicted, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
Future Improvements:
Model Optimization: Tuning hyperparameters, such as the number of LSTM units, epochs, and batch size, could improve model performance.
Feature Engineering: Incorporating more features (e.g., trading volume, technical indicators) could lead to more accurate predictions.
Different Models: Testing other time series models such as ARIMA, SARIMA, or Prophet for comparison with LSTM.
Conclusion:
This project demonstrates the process of forecasting stock prices using LSTM. Although the model performs reasonably well in capturing trends, further fine-tuning and additional features can enhance the accuracy of the predictions.

Technologies Used:
Python: Core language.
Jupyter Notebook: For code execution.
Libraries:
pandas: Data handling.
numpy: Numerical operations.
matplotlib: Plotting and data visualization.
yfinance: Stock data retrieval.
scikit-learn: Data scaling.
tensorflow and keras: Deep learning framework for LSTM model.
Installation
To run this project, install the necessary dependencies:

bash code
pip install pandas numpy matplotlib yfinance scikit-learn tensorflow keras
How to Use
Clone the Repository:
bash
Copy code
git clone https://github.com/your-username/stock-market-time-series.git
Download the Dataset: Stock data is automatically fetched from yfinance.
Run the Python Script:
bash code
python Time_series_analysis_of_stock_market.py
