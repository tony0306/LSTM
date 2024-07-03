import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
#%% 下載股票數據

stock_data = yf.download('NVDA', start='2020-01-01', end='2024-06-30')

# 選擇調整收盤價
data = stock_data[['Adj Close']]

#%% 正規化數據

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

#%% 將數據轉換為時間序列
# X = ([[1, 2, 3],[2, 3, 4],[3, 4, 5],]) y = ([[4, 5],[5, 6],[6, 7],])

def create_sequences(data, seq_length, future_days):
    xs, ys = [], []
    for i in range(len(data) - seq_length - future_days + 1):
        x = data[i:(i + seq_length)]  # 輸入數據
        y = data[i + seq_length : i + seq_length + future_days]  # 目標數據(多點預測)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQUENCE_LENGTH = 20  # 過去的天數
FUTURE_DAYS = 10      # 預測未來的天數
X, y_multi = create_sequences(scaled_data, SEQUENCE_LENGTH, FUTURE_DAYS)

#%% 分割訓練集和測試集
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y_multi[:split], y_multi[split:]

# 調整數據形狀以符合LSTM的要求
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

#%% 創建並訓練LSTM模型
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 1)))
model.add(LSTM(units=50))
model.add(Dense(units=25, activation='relu'))  # 添加額外的DNN層
model.add(Dense(units=FUTURE_DAYS))  # 修改輸出層，預測多天
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

#%% 預測未來時間
current_data = scaled_data[-SEQUENCE_LENGTH:].reshape((1, SEQUENCE_LENGTH, 1))
predicted_prices = model.predict(current_data)[0]

# 反向轉換以得到實際預測值
predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))

# 創建未來日期索引
future_dates = pd.date_range(start='2024-07-01', periods=FUTURE_DAYS)
#%%圖像

font = FontProperties(fname='C:\\Windows\\Fonts\\msjh.ttc')
plt.figure(figsize=(14, 7))
plt.plot(stock_data.index, stock_data['Adj Close'], label='歷史股價', color='blue')
plt.plot(future_dates, predicted_prices, label='預測股價', color='orange')
plt.title('使用LSTM預測NVDA股票價格', fontproperties=font)
plt.xlabel('時間', fontproperties=font)
plt.ylabel('股票價格(美金)', fontproperties=font)
plt.legend(prop=font)
plt.show()

