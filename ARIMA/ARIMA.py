import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#import statsmodels.tsa.api as smt
from statsmodels.tsa.seasonal import seasonal_decompose
#from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings("ignore")

file = r'C:\Users\Admin\Desktop\StockTSMC.csv'      
data = pd.read_csv(file, index_col="Date")
df = pd.DataFrame(data)


arima_data = df["ClosingPrice"]   # ClosingPrice

# 原始資料視覺化
arima_data.plot(figsize=(12,8), label="TSMC")   # 請自行更改label名稱
plt.ylabel("ClosingPrice")   # 請自行更改y軸名稱
plt.legend()
plt.show()

n = 20 * 12     # 每個月23筆資料(交易天數)
ro= 0

result = seasonal_decompose(arima_data, model='multiplicative', period=n)
plt.figure(figsize=(12,8))

plt.subplot(4,1,1)
plt.plot(result.observed, label="TSMC")
plt.ylabel("ClosingPrice")
plt.xticks(df.index[::n], rotation=ro)
plt.margins(0)

plt.subplot(4,1,2)
plt.plot(result.trend)
plt.ylabel("Trend")
plt.xticks(df.index[::n], rotation=ro)
plt.margins(0)

plt.subplot(4,1,3)
plt.plot(result.seasonal)
plt.ylabel("Seasonal")
plt.xticks(df.index[::n], rotation=ro)
plt.margins(0)

plt.subplot(4,1,4)
plt.scatter(df.index, result.resid)
plt.ylabel("Resid")
plt.xticks(df.index[::n], rotation=ro)
plt.margins(0)   # 拆成四個subplot以利調整圖片間距、x軸刻度的呈現

plt.legend()
plt.show()


#時間序列的資料再跑統計模型前，先做平穩性檢驗，來判斷資料是否平穩(stationary)
#如果不平穩，要做差分處理
#Dickey Fuller test是常用的檢定方式：
#如果檢驗統計量小於臨界值，我們可以拒絕原假設(也就是序列是平穩的)
#當檢驗統計量大於臨界值時，無法拒絕原假設(這意味著序列不是平穩的)

def adf_test(timeseries):
    #Perform Dickey-Fuller test:
    print("Results of Dickey-Fuller Test\n==============================================")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(dftest[0:4], index = [
        "Test Statistic", "p-value", "#Lags Used", "Number of Observations Used"])
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)"%key] = value
    print(dfoutput)
    print("==============================================")
    #寫個自動判斷式
    if dfoutput[0] < dfoutput[4]:
        print("The data is stationary. (Criterical Value 1%)")
    elif dfoutput[0] < dfoutput[5]:
        print("The data is stationary. (Criterical Value 5%)")
    elif dfoutput[0] < dfoutput[6]:
        print("The data is stationary. (Criterical Value 10%)")
    else:
        print("The data is non-stationary, so do differencing!")

adf_test(arima_data)



diff_1 = arima_data - arima_data.shift(1)
diff_1 = diff_1.dropna()
diff_1.head()
diff_1.plot(figsize=(12,8), label="diff_1")

plt.legend()

adf_test(diff_1)

#畫ACF(Autocorrelation Function)、PACF(Partial Autocorrelation Function)圖
#可幫助我們判斷模型ARIMA(p, d, q)參數的選擇
#correlogram
f = plt.figure(facecolor='white', figsize=(9,7))
ax1 = f.add_subplot(211)
plot_acf(arima_data, lags=24, ax=ax1);
ax2 = f.add_subplot(212);
plot_pacf(arima_data, lags=24, ax=ax2);
plt.rcParams['axes.unicode_minus'] = False
plt.show()

#5. 樣本內預測模型建立
#在這個部分，我們選擇用透過尋找最小AIC方式來選擇p,d,q的值
def arima_AIC(data, p=4, d=3, q=4):
    best_AIC = ["pdq",10000]
    #L = len(data)
    AIC = []
    name = []
    for i in range(p):
        for j in range(1,d):
            for k in range(q):
                model = ARIMA(data, order=(i,j,k))
                #fitted = model.fit(disp=-1)
                fitted = model.fit()
                AIC.append(fitted.aic)
                name.append(f"ARIMA({i},{j},{k})")
                print(f"ARIMA({i},{j},{k}) : AIC={fitted.aic}")
                if fitted.aic < best_AIC[1]:
                    best_AIC[0] = f"ARIMA({i},{j},{k})"
                    best_AIC[1] = fitted.aic

    print("==============================================================")
    print(f"This best model is {best_AIC[0]} based on argmin AIC.")
    plt.figure(figsize=(12,5))
    plt.bar(name, AIC)
    plt.bar(best_AIC[0], best_AIC[1], color = "red")
    plt.xticks(rotation=30)
    plt.title("AIC")
    plt.savefig("Arima AIC")
    plt.show()

arima_AIC(arima_data, 4,2,3)

#ARIMA(0,1,1) : AIC=8037.9170939612
model = ARIMA(arima_data, order=(0, 1, 1))  #修改 p,d,q參數

#data split
period = 3  #預測 後續 週期
title = f'ARIMA(0, 1, 1) for Forecasting {period} Periods'
L = len(arima_data)
x_train = arima_data[:(L-period)]
x_test = arima_data[-period:]

#Build Model
model = ARIMA(x_train, order=(0, 1, 1))
#fitted = model.fit(disp=-1)
fitted = model.fit()

#Forecast
fc, se, conf = fitted.forecast(period, alpha=0.05)  # 95% conf
#Make as pandas series
fc_series = pd.Series(fc, index=x_test.index)

#Plot
plt.figure(figsize=(10,5), dpi=100)
plt.plot(x_train, label='training')
plt.plot(x_test, label='actual')
plt.plot(fc_series, label='forecast')
plt.xticks(df.index[::n], rotation=ro)  #plt.xticks(df.index[::12], rotation=90)
plt.title(title)
plt.ylabel("ClosingPrice")
plt.legend(loc='upper right', fontsize=8)
plt.savefig(title)
plt.show()

#Results
print(f'Mean Absolute Error : {mean_absolute_error(fc_series,x_test)}')
print(f'Mean Squared Error : {mean_squared_error(fc_series,x_test)}')
print('==============================')
print('fc_series:', fc_series)
print('==============================')
print('ClosingPrice', x_test)



