# By tostq <tostq216@163.com>
# Reference to hmmlearn.examples.plot_hmm_stock_analysis.py
# blog: blog.csdn.net/tostq

import datetime
import numpy as np
import pandas as pd
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
from hmm import GaussianHMM
from sklearn.preprocessing import scale

# 導入Yahoo金融資料
r = pd.read_csv('yahoofinance-SPY-20080101-20180101.csv')

dates = r['Date'].values # 日期列
close_v = r['Close'].values # 收盤價
volume = r['Volume'].values[1:] # 交易量


# diff：out[n] = a[n+1] - a[n] 得到價格變化，即一階差分
diff = np.diff(close_v)
dates = dates[1:]
close_v = close_v[1:]


# scale標準化：均值為0和方差為1
# 將價格和交易陣列成輸入資料
X = np.column_stack([scale(diff), scale(volume)])


# 訓練高斯HMM模型，這裡假設隱藏狀態4個
model = GaussianHMM(4,2,20)
model.train(X)

# 預測隱狀態
hidden_states = model.decode(X)

# 列印參數
print("Transition matrix: ", model.transmat_prob)
print("Means and vars of each hidden state")
for i in range(model.n_state):
    print("{0}th hidden state".format(i))
    print("mean = ", model.emit_means[i])
    print("var = ", model.emit_covars[i])
    print()


# 畫圖描述
fig, axs = plt.subplots(model.n_state, sharex=True, sharey=True)
colours = cm.rainbow(np.linspace(0, 1, model.n_state))
for i, (ax, colour) in enumerate(zip(axs, colours)):
    # Use fancy indexing to plot data in each state.
    mask = hidden_states == i
    ax.plot_date(dates[mask], close_v[mask], ".-", c=colour)
    ax.set_title("{0}th hidden state".format(i))

    # Format the ticks.
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_minor_locator(MonthLocator())

    ax.grid(True)

plt.show()

