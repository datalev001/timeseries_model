import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import prophet
from prophet import Prophet
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose


####################################################
data=pd.read_csv("Electric_Production.csv",parse_dates=['DATE'],index_col='DATE')
monthly_data = data.IPG2211A2N.resample('M').mean().reset_index()
data.IPG2211A2N.resample('M').mean().plot()
plt.show()
monthly_data.shape


# Perform Dickey-Fuller test
result = adfuller(monthly_data['IPG2211A2N'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# Calculate ACF and PACF
lags = 20  # Number of lags to calculate
acf_values = acf(monthly_data['IPG2211A2N'], nlags=lags)
pacf_values = pacf(monthly_data['IPG2211A2N'], nlags=lags, method='ols')

# Plot ACF
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(acf_values)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(monthly_data['IPG2211A2N'])), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(monthly_data['IPG2211A2N'])), linestyle='--', color='gray')
plt.title('Autocorrelation Function')

# Plot PACF
plt.subplot(122)
plt.plot(pacf_values)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(monthly_data['IPG2211A2N'])), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(monthly_data['IPG2211A2N'])), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()


################SARIMA for Electric_Production####################

model_sarima = SARIMAX(train_data, order=(1, 1, 0), seasonal_order=(1, 1, 1, 12))
model_sarima_fit = model_sarima.fit()

# Forecast the last three data points
forecast_sarima = model_sarima_fit.forecast(steps=3)

# Calculate the MAPE between the actual and predicted values
mape_sarima = mean_absolute_percentage_error(test_data, forecast_sarima)

#####################RNN##############################################
tmdata = monthly_data['IPG2211A2N'] #  Electric_Production data
data = tmdata.values.reshape(-1, 1) 

# Decompose to remove the seasonal component
result = seasonal_decompose(tmdata, model='additive', period=12)
deseasonalized = tmdata - result.seasonal

# Normalize the data
scaler = MinMaxScaler(feature_range=(-1, 1))
data_normalized = scaler.fit_transform(deseasonalized.values.reshape(-1, 1))

# Convert data into sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 12
X, y = create_sequences(data_normalized, seq_length)

X_train, X_test = X[:-3], X[-3-seq_length:-seq_length]
y_train, y_test = y[:-3], y[-3:]

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).view(-1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).view(-1)

class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super(SimpleRNN, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.rnn = nn.RNN(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
    
    def forward(self, input_seq):
        rnn_out, _ = self.rnn(input_seq.view(len(input_seq) ,1, -1))
        predictions = self.linear(rnn_out.view(len(input_seq), -1))
        return predictions[-1]

model = SimpleRNN()
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.018)

epochs = 220
for i in range(epochs):
    for seq, labels in zip(X_train, y_train):
        optimizer.zero_grad()
        y_pred = model(seq)
        single_loss = criterion(y_pred, labels.unsqueeze(-1))
        single_loss.backward()
        optimizer.step()
    if i % 10 == 0:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

model.eval()
preds_list = []
with torch.no_grad():
    for i in range(len(X_test)):
        seq = X_test[i].view(-1, 1, 1)  # Reshape to (seq_len, batch_size=1, features=1)
        pred = model(seq)
        preds_list.append(pred.item())
        
# Convert predictions list to a numpy array for inverse scaling
preds_array = np.array(preds_list).reshape(-1, 1)
preds_inverse = scaler.inverse_transform(preds_array)

# Inverse transform the actual test labels
y_test_inverse = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))

# Calculate MAPE
mape = np.mean(np.abs((y_test_inverse - preds_inverse) / y_test_inverse)) * 100
print(f'MAPE: {mape}%')

#########################LSTM#################################################


tmdata = monthly_data['IPG2211A2N'] 

#tmdata = series.sales # shapoo data
data = tmdata.values.reshape(-1, 1) 

# Decompose to remove the seasonal component
result = seasonal_decompose(tmdata, model='additive', period=12)
deseasonalized = tmdata - result.seasonal

# Normalize the data
scaler = MinMaxScaler(feature_range=(-1, 1))
data_normalized = scaler.fit_transform(deseasonalized.values.reshape(-1, 1))

# Convert data into sequences
seq_length = 12
X, y = create_sequences(data_normalized, seq_length)

# Split data into training and testing sets
X_train, X_test = X[:-3], X[-3-seq_length:-seq_length]
y_train, y_test = y[:-3], y[-3:]

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).view(-1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).view(-1)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
    
    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq) ,1, -1))
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

model = LSTMModel()
criterion = nn.MSELoss()

# lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 180
for i in range(epochs):
    for seq, labels in zip(X_train, y_train):
        optimizer.zero_grad()
        y_pred = model(seq)
        single_loss = criterion(y_pred, labels.unsqueeze(-1))
        single_loss.backward()
        optimizer.step()
    if i % 10 == 0:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

model.eval()
preds_list = []
with torch.no_grad():
    for i in range(len(X_test)):
        seq = X_test[i].view(-1, 1, 1)  # Reshape to (seq_len, batch_size=1, features=1)
        pred = model(seq)
        preds_list.append(pred.item())

# Convert predictions list to a numpy array for inverse scaling
preds_array = np.array(preds_list).reshape(-1, 1)
preds_inverse = scaler.inverse_transform(preds_array) + result.seasonal[-len(X_test):].values.reshape(-1, 1)

# Inverse transform the actual test labels
y_test_inverse = scaler.inverse_transform(y_test.numpy().reshape(-1, 1)) + result.seasonal[-len(y_test):].values.reshape(-1, 1)

# Calculate MAPE
mape = np.mean(np.abs((y_test_inverse - preds_inverse) / y_test_inverse)) * 100

print(f'MAPE: {mape}%')

##############################Prophet###############################################
start_date = '2020-01-01'
dates = pd.date_range(start=start_date, periods=len(monthly_data['IPG2211A2N']), freq='M')
df_prophet = pd.DataFrame(data={'ds': dates, 'y': monthly_data['IPG2211A2N'].values})

# Initialize the Prophet model with additional seasonality components
model = Prophet(yearly_seasonality=True, seasonality_prior_scale=0.2)
model.add_seasonality(name='monthly', period=30.5, fourier_order=8)  # Example of adding monthly seasonality

# Fit the model with your DataFrame
model.fit(df_prophet[:-3])  # Exclude the last 3 months for validation

# Create a DataFrame for future predictions including the last 3 months
future = model.make_future_dataframe(periods=3, freq='M')

# Use the model to make predictions
forecast = model.predict(future)

# Focus on the last 3 months for validation
forecast_last_3_months = forecast['yhat'][-3:].values

# Actual values for the last 3 months
actual_last_3_months = df_prophet['y'][-3:].values

# Calculate the MAPE between actual and forecasted values
mape = mean_absolute_percentage_error(actual_last_3_months, forecast_last_3_months)

print(f"Forecasted Values: {forecast_last_3_months}")
print(f"Actual Values: {actual_last_3_months}")
print(f"MAPE: {mape}")

#################Transformer-Based Sequential Forecasting Model####################


result = seasonal_decompose(monthly_data['IPG2211A2N'], model='additive', period=12)

deseasonalized = result.trend + result.resid
deseasonalized.dropna(inplace=True)  # Remove NaN values after decomposition

# Scale the deseasonalized data
scaler = MinMaxScaler(feature_range=(-1, 1))
data_normalized = scaler.fit_transform(deseasonalized.values.reshape(-1, 1))

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 12
X, y = create_sequences(data_normalized, seq_length)

# Split data into training and testing sets
X_train, X_test = X[:-3], X[-3-seq_length:-seq_length]
y_train, y_test = y[:-3], y[-3:]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.encoder = nn.Linear(input_dim, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output

model = TransformerModel(input_dim=1, d_model=64, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.2)
train_data = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
train_loader = DataLoader(train_data, batch_size=16, shuffle=False)

# lr=0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
# 150
for epoch in range(120):
    model.train()
    total_loss = 0
    for batch, (data, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.permute(1, 0, 2)  # Reshape for the transformer [seq_len, batch_size, features]
        output = model(data)
        loss = criterion(output.view(-1), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, Loss: {total_loss / len(train_loader)}')

# Forecasting and reseasonalizing predictions
model.eval()
preds = []
with torch.no_grad():
    for seq in torch.FloatTensor(X_test):
        seq = seq.unsqueeze(1)  # Shape to [seq_len, batch_size=1, features]
        pred = model(seq)
        pred_last = pred[-1, :, :].squeeze().item()
        preds.append(pred_last)

# Inverse transform the predictions and add back the seasonal component
preds_inverse = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
seasonal_component = result.seasonal[-len(preds):].values.reshape(-1, 1)
final_predictions = preds_inverse + seasonal_component

# Evaluate the model
y_test_actual = monthly_data['IPG2211A2N'][-len(preds):].values.reshape(-1, 1)
mape = np.mean(np.abs((y_test_actual - final_predictions) / y_test_actual)) * 100

# MAPE: 4.55%
print(f'MAPE: {mape}%')
