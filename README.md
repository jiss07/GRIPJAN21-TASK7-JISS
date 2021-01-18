
# Author: JISS PETER , Data Science Intern @ The Spark Foundation
The task related dataset is available on the web url https://finance.yahoo.com and https://bit.ly/36fFPI6. This dataset can be downloaded locally or can access directly in the code.

# Data Science & Business Analytics Internship at The Sparks Foundation.
#### #GRIPJAN21

## GRIP-Task 7 - Stock Market Prediction using Numerical and Textual Analysis

### Task Description

* Create a hybrid model for stock price/performance prediction using numerical analysis of historical stock prices, and sentimental analysis of news headlines
* Stock to analyze and predict - SENSEX (S&P BSE SENSEX)
* Download historical stock prices from finance.yahoo.com
* Download textual (news) data from https://bit.ly/36fFPI6

# Import all libraries


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import datetime
from datetime import date 
import yfinance as yf
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
```

# 1. Numerical Analysis


```python
#Web data reader is extension of pandas library to communicate with frequently updating data
import pandas_datareader.data as web
from pandas import Series, DataFrame
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime(2019, 12, 31)

df = web.DataReader("MSFT", 'yahoo', start, end)
df.tail()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-12-24</th>
      <td>157.710007</td>
      <td>157.119995</td>
      <td>157.479996</td>
      <td>157.380005</td>
      <td>8989200.0</td>
      <td>155.730255</td>
    </tr>
    <tr>
      <th>2019-12-26</th>
      <td>158.729996</td>
      <td>157.399994</td>
      <td>157.559998</td>
      <td>158.669998</td>
      <td>14520600.0</td>
      <td>157.006729</td>
    </tr>
    <tr>
      <th>2019-12-27</th>
      <td>159.550003</td>
      <td>158.220001</td>
      <td>159.449997</td>
      <td>158.960007</td>
      <td>18412800.0</td>
      <td>157.293686</td>
    </tr>
    <tr>
      <th>2019-12-30</th>
      <td>159.020004</td>
      <td>156.729996</td>
      <td>158.990005</td>
      <td>157.589996</td>
      <td>16348400.0</td>
      <td>155.938049</td>
    </tr>
    <tr>
      <th>2019-12-31</th>
      <td>157.770004</td>
      <td>156.449997</td>
      <td>156.770004</td>
      <td>157.699997</td>
      <td>18369400.0</td>
      <td>156.046890</td>
    </tr>
  </tbody>
</table>
</div>



* We have the data of Microsoft stocks of 19 years from January 2000 till December 2019.

* We analyse stocks using two measurements- Rolling mean and Return rate.

* Rolling Mean - Rolling is a very useful operation for time series data. Rolling means creating a rolling window with a specified size and perform calculations on the data in this window which, of course, rolls through the data. In pandas

* Moving Average- technical analysis tool that smooths out price data by creating a constantly updated average price.

# A) Data Analysis


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-01-03</th>
      <td>59.3125</td>
      <td>56.00000</td>
      <td>58.68750</td>
      <td>58.28125</td>
      <td>53228400.0</td>
      <td>37.102634</td>
    </tr>
    <tr>
      <th>2000-01-04</th>
      <td>58.5625</td>
      <td>56.12500</td>
      <td>56.78125</td>
      <td>56.31250</td>
      <td>54119000.0</td>
      <td>35.849308</td>
    </tr>
    <tr>
      <th>2000-01-05</th>
      <td>58.1875</td>
      <td>54.68750</td>
      <td>55.56250</td>
      <td>56.90625</td>
      <td>64059600.0</td>
      <td>36.227283</td>
    </tr>
    <tr>
      <th>2000-01-06</th>
      <td>56.9375</td>
      <td>54.18750</td>
      <td>56.09375</td>
      <td>55.00000</td>
      <td>54976600.0</td>
      <td>35.013741</td>
    </tr>
    <tr>
      <th>2000-01-07</th>
      <td>56.1250</td>
      <td>53.65625</td>
      <td>54.31250</td>
      <td>55.71875</td>
      <td>62013600.0</td>
      <td>35.471302</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.isnull().sum()
```




    High         0
    Low          0
    Open         0
    Close        0
    Volume       0
    Adj Close    0
    dtype: int64




```python
close_px = df['Adj Close']
mavg = close_px.rolling(window = 100).mean()
mavg.tail(10)
```




    Date
    2019-12-17    139.752159
    2019-12-18    139.893509
    2019-12-19    140.054802
    2019-12-20    140.273019
    2019-12-23    140.473642
    2019-12-24    140.685370
    2019-12-26    140.955960
    2019-12-27    141.205044
    2019-12-30    141.434773
    2019-12-31    141.630108
    Name: Adj Close, dtype: float64




```python
df.shape
```




    (5031, 6)



* The profit or loss calculation is usually determined by the closing price of a stock; hence we will consider the closing price as the target variable.

# B) Data Visualization

# i) Movement of Data: Closing Stock


```python
#Closing Stock
df['Close'].hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f35f0a885f8>




![png](GRIP-Task7-StockMarketPredictionUsingNumericalandTextualAnalysis_files/GRIP-Task7-StockMarketPredictionUsingNumericalandTextualAnalysis_15_1.png)



```python
df['Close'].plot()
plt.xlabel("Date")
plt.ylabel("Close")
```




    Text(0, 0.5, 'Close')




![png](GRIP-Task7-StockMarketPredictionUsingNumericalandTextualAnalysis_files/GRIP-Task7-StockMarketPredictionUsingNumericalandTextualAnalysis_16_1.png)



```python
df['Close'].plot(style='.')
plt.title("Scatter plot of Closing Price")
plt.title('Scatter plot of Closing Price',fontsize=20)
plt.show()
```


![png](GRIP-Task7-StockMarketPredictionUsingNumericalandTextualAnalysis_files/GRIP-Task7-StockMarketPredictionUsingNumericalandTextualAnalysis_17_0.png)


* It shows an Upward trend so, the data is not stationary. Now we go with rolling mean and rolling standard deviation.
* ADF (Augmented Dickey-Fuller) Test

* The Dickey-Fuller test is one of the most popular statistical tests. It can be used to determine the presence of unit root in the series, and hence help us understand if the series is stationary or not. The null and alternate hypothesis of this test is:

* 1.Null Hypothesis: The series has a unit root (value of a =1)

* 2.Alternate Hypothesis: The series has no unit root.


```python
#Test for staionarity
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolling_mean = timeseries.rolling(12).mean()
    rolling_std = timeseries.rolling(12).std()
    #Plot rolling statistics:
    plt.plot(timeseries, color='Black',label='Original')
    plt.plot(rolling_mean, color='Green', label='Rolling Mean')
    plt.plot(rolling_std, color='Red', label = 'Rolling Standard Deviation')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation',fontsize=20)
    plt.show(block=False)
    
    print("Results of dickey fuller test")
    adft = adfuller(timeseries,autolag='AIC')
    # output for dft will give us without defining what the values are.
    #hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key,values in adft[4].items():
        output['critical value (%s)'%key] =  values
    print(output)
    
test_stationarity(df['Close'])
```


![png](GRIP-Task7-StockMarketPredictionUsingNumericalandTextualAnalysis_files/GRIP-Task7-StockMarketPredictionUsingNumericalandTextualAnalysis_19_0.png)


    Results of dickey fuller test
    Test Statistics                   4.581411
    p-value                           1.000000
    No. of lags used                  8.000000
    Number of observations used    5022.000000
    critical value (1%)              -3.431653
    critical value (5%)              -2.862116
    critical value (10%)             -2.567076
    dtype: float64
    

* By looking at the above graph it can be seen that the mean and standard deviation are increasing and hence our data is not stationary. The p-value is greater than 0.05, and hence we cannot reject the null hypothesis, and the null hypothesis states that the series has unit root and hence is not stationary.


```python
#Doing a quick vanilla decomposition to see any trend seasonality etc in the ts
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df.Close, model='multiplicative',freq=30)
fig = decomposition.plot()
fig.set_figwidth(12)
fig.set_figheight(8)
fig.suptitle('Decomposition of multiplicative time series',fontsize=20)
plt.show()
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:3: FutureWarning: the 'freq'' keyword is deprecated, use 'period' instead
      This is separate from the ipykernel package so we can avoid doing imports until
    


![png](GRIP-Task7-StockMarketPredictionUsingNumericalandTextualAnalysis_files/GRIP-Task7-StockMarketPredictionUsingNumericalandTextualAnalysis_21_1.png)



```python
from pylab import rcParams
rcParams['figure.figsize'] = 10, 9
df_log = np.log(df.Close)
moving_avg = df.Close.rolling(12).mean()
std_dev = df.Close.rolling(12).std()
plt.legend(loc='best')
plt.title('Moving Average',fontsize=20)
plt.plot(std_dev, color ="Blue", label = "Standard Deviation")
plt.plot(moving_avg, color="Green", label = "Mean")
plt.legend()
plt.show()
```

    No handles with labels found to put in legend.
    


![png](GRIP-Task7-StockMarketPredictionUsingNumericalandTextualAnalysis_files/GRIP-Task7-StockMarketPredictionUsingNumericalandTextualAnalysis_22_1.png)



```python
#Split data into train and training set
train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
plt.figure(figsize=(10,9))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.plot(df_log, 'green', label='Train data')
plt.plot(test_data, 'blue', label='Test data')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f35e3bc8b38>




![png](GRIP-Task7-StockMarketPredictionUsingNumericalandTextualAnalysis_files/GRIP-Task7-StockMarketPredictionUsingNumericalandTextualAnalysis_23_1.png)



```python
import statsmodels
from statsmodels import compat
from statsmodels.compat import pandas
from pmdarima.arima import auto_arima

auto_arima_model = auto_arima(train_data, start_p=0, start_q=0,
                      test='adf',       # use adftest to find             optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
print(auto_arima_model.summary())
```

    Performing stepwise search to minimize aic
     ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=-22832.327, Time=0.77 sec
     ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=-22837.605, Time=0.28 sec
     ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=-22837.985, Time=2.15 sec
     ARIMA(0,1,0)(0,0,0)[0]             : AIC=-22834.211, Time=0.28 sec
     ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=-22843.360, Time=1.70 sec
     ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=-22838.800, Time=6.31 sec
     ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=-22839.784, Time=2.39 sec
     ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=-22839.018, Time=1.81 sec
     ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=-22838.391, Time=0.90 sec
     ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=-22836.985, Time=5.06 sec
     ARIMA(1,1,1)(0,0,0)[0]             : AIC=-22845.213, Time=0.39 sec
     ARIMA(0,1,1)(0,0,0)[0]             : AIC=-22839.859, Time=0.53 sec
     ARIMA(1,1,0)(0,0,0)[0]             : AIC=-22839.480, Time=0.17 sec
     ARIMA(2,1,1)(0,0,0)[0]             : AIC=-22840.661, Time=1.42 sec
     ARIMA(1,1,2)(0,0,0)[0]             : AIC=-22840.950, Time=0.41 sec
     ARIMA(0,1,2)(0,0,0)[0]             : AIC=-22840.885, Time=0.62 sec
     ARIMA(2,1,0)(0,0,0)[0]             : AIC=-22840.260, Time=0.40 sec
     ARIMA(2,1,2)(0,0,0)[0]             : AIC=-22841.334, Time=2.79 sec
    
    Best model:  ARIMA(1,1,1)(0,0,0)[0]          
    Total fit time: 28.401 seconds
                                   SARIMAX Results                                
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                 4524
    Model:               SARIMAX(1, 1, 1)   Log Likelihood               11425.607
    Date:                Mon, 18 Jan 2021   AIC                         -22845.213
    Time:                        13:50:33   BIC                         -22825.962
    Sample:                             0   HQIC                        -22838.432
                                   - 4524                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    ar.L1          0.6472      0.101      6.404      0.000       0.449       0.845
    ma.L1         -0.6888      0.096     -7.204      0.000      -0.876      -0.501
    sigma2         0.0004   3.27e-06    114.479      0.000       0.000       0.000
    ===================================================================================
    Ljung-Box (L1) (Q):                   0.00   Jarque-Bera (JB):             18358.03
    Prob(Q):                              0.95   Prob(JB):                         0.00
    Heteroskedasticity (H):               0.35   Skew:                            -0.18
    Prob(H) (two-sided):                  0.00   Kurtosis:                        12.86
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
    


```python
auto_arima_model.plot_diagnostics()
plt.show()
```


![png](GRIP-Task7-StockMarketPredictionUsingNumericalandTextualAnalysis_files/GRIP-Task7-StockMarketPredictionUsingNumericalandTextualAnalysis_25_0.png)


###### Plot Interpretation

* Standardized Residuals: The plot shows the residual errors, the residuals seems to fluctuate around zero i.e. the mean and hence it seems that the variance is uniform.

* Histogram plus estimated density: The plot is bell shapes and hence a normal distribution. So for a normal distribution the mean is zero.

* Q-Q Plot: In the Q-Q plot the dots fall perfectly on the line and hence the distribution is non-skewed.

* Correlogram: It is also called Auto Correlation Function, which shows serial correlation in the data that changes over time. On the y axis is the autocorrelation. The x axis tells you the lag. So, if x=1 we have a lag of 1. If x=2, we have a lag of 2.

* The auto_arima model suggets that the best fit model is ARIMA(2,1,0) and the fit time is 13.985 seconds.




```python
from statsmodels.tsa.arima_model import ARIMA
model_arima=ARIMA(train_data,order=(2,1,0))
fit_model=model_arima.fit()
print(fit_model.summary())
```

    /usr/local/lib/python3.6/dist-packages/statsmodels/tsa/arima_model.py:472: FutureWarning: 
    statsmodels.tsa.arima_model.ARMA and statsmodels.tsa.arima_model.ARIMA have
    been deprecated in favor of statsmodels.tsa.arima.model.ARIMA (note the .
    between arima and model) and
    statsmodels.tsa.SARIMAX. These will be removed after the 0.12 release.
    
    statsmodels.tsa.arima.model.ARIMA makes use of the statespace framework and
    is both well tested and maintained.
    
    To silence this warning and continue using ARMA and ARIMA until they are
    removed, use:
    
    import warnings
    warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                            FutureWarning)
    warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                            FutureWarning)
    
      warnings.warn(ARIMA_DEPRECATION_WARN, FutureWarning)
    /usr/local/lib/python3.6/dist-packages/statsmodels/tsa/base/tsa_model.py:583: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      ' ignored when e.g. forecasting.', ValueWarning)
    /usr/local/lib/python3.6/dist-packages/statsmodels/tsa/base/tsa_model.py:583: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      ' ignored when e.g. forecasting.', ValueWarning)
    

                                 ARIMA Model Results                              
    ==============================================================================
    Dep. Variable:                D.Close   No. Observations:                 4523
    Model:                 ARIMA(2, 1, 0)   Log Likelihood               11423.196
    Method:                       css-mle   S.D. of innovations              0.019
    Date:                Mon, 18 Jan 2021   AIC                         -22838.392
    Time:                        13:50:57   BIC                         -22812.724
    Sample:                             1   HQIC                        -22829.350
                                                                                  
    =================================================================================
                        coef    std err          z      P>|z|      [0.025      0.975]
    ---------------------------------------------------------------------------------
    const          9.787e-05      0.000      0.362      0.717      -0.000       0.001
    ar.L1.D.Close    -0.0411      0.015     -2.765      0.006      -0.070      -0.012
    ar.L2.D.Close    -0.0248      0.015     -1.670      0.095      -0.054       0.004
                                        Roots                                    
    =============================================================================
                      Real          Imaginary           Modulus         Frequency
    -----------------------------------------------------------------------------
    AR.1           -0.8280           -6.2941j            6.3484           -0.2708
    AR.2           -0.8280           +6.2941j            6.3484            0.2708
    -----------------------------------------------------------------------------
    


```python
fc, se, conf = fit_model.forecast(504, alpha=0.05) 
fc_series = pd.Series(fc, index=test_data.index)
low = pd.Series(conf[:, 0], index=test_data.index)
up = pd.Series(conf[:, 1], index=test_data.index)
```


```python
plt.figure(figsize=(10,9), dpi=100)
plt.plot(train_data, label='Training Set')
plt.plot(test_data, color = 'Black', label='Actual Stock Price')
plt.plot(fc_series, color = 'Green',label='Predicted Stock Price')
plt.fill_between(low.index, low, up, color='k', alpha=.10)
plt.title('S&P BSE SENSEX Stock Price Prediction',fontsize=20)
plt.xlabel('Year')
plt.ylabel('Stock Price')
plt.legend(fontsize=10,loc='upper left')
plt.show()
```


![png](GRIP-Task7-StockMarketPredictionUsingNumericalandTextualAnalysis_files/GRIP-Task7-StockMarketPredictionUsingNumericalandTextualAnalysis_29_0.png)


# ii) Analyze stocks of various IT giants like Microsoft, Apple, Amazon, Google ,IBM.                                                 


```python
dfcomp = web.DataReader(['MSFT' , 'AAPL' , 'AMZN' , 'GOOG' , 'IBM'], 'yahoo', start=start, end=end)['Adj Close']
dfcomp.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Symbols</th>
      <th>MSFT</th>
      <th>AAPL</th>
      <th>AMZN</th>
      <th>GOOG</th>
      <th>IBM</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-01-03</th>
      <td>37.102634</td>
      <td>0.862169</td>
      <td>89.3750</td>
      <td>NaN</td>
      <td>73.865021</td>
    </tr>
    <tr>
      <th>2000-01-04</th>
      <td>35.849308</td>
      <td>0.789480</td>
      <td>81.9375</td>
      <td>NaN</td>
      <td>71.357750</td>
    </tr>
    <tr>
      <th>2000-01-05</th>
      <td>36.227283</td>
      <td>0.801032</td>
      <td>69.7500</td>
      <td>NaN</td>
      <td>73.865021</td>
    </tr>
    <tr>
      <th>2000-01-06</th>
      <td>35.013741</td>
      <td>0.731712</td>
      <td>65.5625</td>
      <td>NaN</td>
      <td>72.591492</td>
    </tr>
    <tr>
      <th>2000-01-07</th>
      <td>35.471302</td>
      <td>0.766373</td>
      <td>69.5625</td>
      <td>NaN</td>
      <td>72.273109</td>
    </tr>
  </tbody>
</table>
</div>



* Correlation analysis using correlation function in pandas


```python
retscomp = dfcomp.pct_change()
corr = retscomp.corr()
retscomp.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Symbols</th>
      <th>MSFT</th>
      <th>AAPL</th>
      <th>AMZN</th>
      <th>GOOG</th>
      <th>IBM</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-01-03</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2000-01-04</th>
      <td>-0.033780</td>
      <td>-0.084310</td>
      <td>-0.083217</td>
      <td>NaN</td>
      <td>-0.033944</td>
    </tr>
    <tr>
      <th>2000-01-05</th>
      <td>0.010543</td>
      <td>0.014633</td>
      <td>-0.148741</td>
      <td>NaN</td>
      <td>0.035137</td>
    </tr>
    <tr>
      <th>2000-01-06</th>
      <td>-0.033498</td>
      <td>-0.086538</td>
      <td>-0.060036</td>
      <td>NaN</td>
      <td>-0.017241</td>
    </tr>
    <tr>
      <th>2000-01-07</th>
      <td>0.013068</td>
      <td>0.047369</td>
      <td>0.061010</td>
      <td>NaN</td>
      <td>-0.004386</td>
    </tr>
    <tr>
      <th>2000-01-10</th>
      <td>0.007291</td>
      <td>-0.017588</td>
      <td>-0.005391</td>
      <td>NaN</td>
      <td>0.039648</td>
    </tr>
    <tr>
      <th>2000-01-11</th>
      <td>-0.025612</td>
      <td>-0.051151</td>
      <td>-0.035230</td>
      <td>NaN</td>
      <td>0.008475</td>
    </tr>
    <tr>
      <th>2000-01-12</th>
      <td>-0.032571</td>
      <td>-0.059973</td>
      <td>-0.047753</td>
      <td>NaN</td>
      <td>0.004202</td>
    </tr>
    <tr>
      <th>2000-01-13</th>
      <td>0.018902</td>
      <td>0.109677</td>
      <td>0.037365</td>
      <td>NaN</td>
      <td>-0.010460</td>
    </tr>
    <tr>
      <th>2000-01-14</th>
      <td>0.041159</td>
      <td>0.038114</td>
      <td>-0.025592</td>
      <td>NaN</td>
      <td>0.011628</td>
    </tr>
  </tbody>
</table>
</div>



## Using Heat Maps to visualize correlation range of various stocks


```python
plt.imshow(corr, cmap = 'hot', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns)
plt.yticks(range(len(corr)), corr.columns);
```


![png](GRIP-Task7-StockMarketPredictionUsingNumericalandTextualAnalysis_files/GRIP-Task7-StockMarketPredictionUsingNumericalandTextualAnalysis_35_0.png)


* Stocks Return rate and Risk


```python
plt.scatter(retscomp.mean(), retscomp.std())
plt.xlabel('Expected returns')
plt.ylabel('Risk')
for label, x, y in zip(retscomp.columns, retscomp.mean(), retscomp.std()):
  plt.annotate(label, xy = (x, y), xytext = (20, -20),
               textcoords = 'offset points', ha = 'right', va = 'bottom',
               bbox = dict(boxstyle = 'round, pad = 0.5', fc  = 'yellow',
                           alpha = 0.5), arrowprops = dict(arrowstyle = '->',
                                                           connectionstyle = 'arc3,rad = 0'))
```


![png](GRIP-Task7-StockMarketPredictionUsingNumericalandTextualAnalysis_files/GRIP-Task7-StockMarketPredictionUsingNumericalandTextualAnalysis_37_0.png)


# iii) Performance Metrics


```python
from sklearn.metrics import r2_score
from sklearn import metrics
```


```python
#Mean Squared Error
mse = metrics.mean_squared_error(test_data, fc)
print('Mean Squared Error:',mse)
```

    Mean Squared Error: 0.09036525601085134
    


```python
#RMSE is the standard deviation of the errors it is same as MSE but the root of the value is considered
print("Root Mean Squared Error: ",np.sqrt(metrics.mean_squared_error(test_data, fc)))
```

    Root Mean Squared Error:  0.300608143620314
    

# 2. Textual Analysis


```python
import pandas as pd
import numpy as np 
df1 = pd.read_csv('/content/JISS/india-news-headlines.csv')
df1.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>publish_date</th>
      <th>headline_category</th>
      <th>headline_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20010101</td>
      <td>sports.wwe</td>
      <td>win over cena satisfying but defeating underta...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20010102</td>
      <td>unknown</td>
      <td>Status quo will not be disturbed at Ayodhya; s...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20010102</td>
      <td>unknown</td>
      <td>Fissures in Hurriyat over Pak visit</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20010102</td>
      <td>unknown</td>
      <td>America's unwanted heading for India?</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20010102</td>
      <td>unknown</td>
      <td>For bigwigs; it is destination Goa</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20010102</td>
      <td>unknown</td>
      <td>Extra buses to clear tourist traffic</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20010102</td>
      <td>unknown</td>
      <td>Dilute the power of transfers; says Riberio</td>
    </tr>
    <tr>
      <th>7</th>
      <td>20010102</td>
      <td>unknown</td>
      <td>Focus shifts to teaching of Hindi</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20010102</td>
      <td>unknown</td>
      <td>IT will become compulsory in schools</td>
    </tr>
    <tr>
      <th>9</th>
      <td>20010102</td>
      <td>unknown</td>
      <td>Move to stop freedom fighters' pension flayed</td>
    </tr>
  </tbody>
</table>
</div>



* The dataset has 3 columns date, category and news. The publish_date column is parsed into date format. The headine_category column is not important and won't contribute for the model building and hence the column will be dropped in the later step.

# A) Data Analysis


```python
df1.tail(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>publish_date</th>
      <th>headline_category</th>
      <th>headline_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3160319</th>
      <td>20190930</td>
      <td>home.education</td>
      <td>IIT M.Tech fee hike not for existing students:...</td>
    </tr>
    <tr>
      <th>3160320</th>
      <td>20190930</td>
      <td>city.chandigarh</td>
      <td>Jananayak Janata Party releases second list of...</td>
    </tr>
    <tr>
      <th>3160321</th>
      <td>20190930</td>
      <td>city.gurgaon</td>
      <td>Jananayak Janata Party releases second list of...</td>
    </tr>
    <tr>
      <th>3160322</th>
      <td>20190930</td>
      <td>city.faridabad</td>
      <td>Jananayak Janata Party releases second list of...</td>
    </tr>
    <tr>
      <th>3160323</th>
      <td>20190930</td>
      <td>city.mumbai</td>
      <td>Man who lifted Andheri kid held in Gujarat aft...</td>
    </tr>
    <tr>
      <th>3160324</th>
      <td>20190930</td>
      <td>city.kolkata</td>
      <td>Bar bar dekho but note mat pheko: Kolkata cops</td>
    </tr>
    <tr>
      <th>3160325</th>
      <td>20190930</td>
      <td>city.thane</td>
      <td>Kalyan man held for rape</td>
    </tr>
    <tr>
      <th>3160326</th>
      <td>20190930</td>
      <td>sports.more-sports.athletics</td>
      <td>India's 4x400m mixed relay team finishes 7th i...</td>
    </tr>
    <tr>
      <th>3160327</th>
      <td>20190930</td>
      <td>entertainment.hindi.bollywood</td>
      <td>PHOTOS: Shahid Kapoor and Mira Rajput get papp...</td>
    </tr>
    <tr>
      <th>3160328</th>
      <td>20190930</td>
      <td>tv.news.hindi</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1.shape
```




    (3160329, 3)




```python
df1.isnull().sum()
```




    publish_date         0
    headline_category    0
    headline_text        1
    dtype: int64




```python
df1.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3160329 entries, 0 to 3160328
    Data columns (total 3 columns):
     #   Column             Dtype 
    ---  ------             ----- 
     0   publish_date       int64 
     1   headline_category  object
     2   headline_text      object
    dtypes: int64(1), object(2)
    memory usage: 72.3+ MB
    


```python
df1.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>publish_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3.160329e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.012156e+07</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.756475e+04</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.001010e+07</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.009081e+07</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.013033e+07</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.016061e+07</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.019093e+07</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1.max()
```




    publish_date            20190930
    headline_category    young-turks
    dtype: object




```python
df1.min()
```




    publish_date                 20010101
    headline_category    2008-in-pictures
    dtype: object




```python
df1['headline_category'].unique()
```




    array(['sports.wwe', 'unknown', 'entertainment.hindi.bollywood', ...,
           'times-fact-check.news', 'elections.assembly-elections.haryana',
           'elections.assembly-elections.maharashtra'], dtype=object)




```python
df1.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>publish_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>publish_date</th>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1.shape
```




    (3160329, 3)



# B) Data Visualization

# i) EDA (Exploratry data analysis) using NLP and NLTK tools


```python
sns.set_palette('viridis')
sns.pairplot(df1)
plt.show()
```


![png](GRIP-Task7-StockMarketPredictionUsingNumericalandTextualAnalysis_files/GRIP-Task7-StockMarketPredictionUsingNumericalandTextualAnalysis_58_0.png)


* Remove URL from the text


```python
df1['headline_text'].value_counts()
```


```python
df1['headline_category'].value_counts()
```




    india                                                         280286
    unknown                                                       207732
    city.mumbai                                                   129819
    city.delhi                                                    119509
    business.india-business                                       111953
    city.chandigarh                                               104768
    city.hyderabad                                                 91094
    city.bengaluru                                                 89257
    entertainment.hindi.bollywood                                  87879
    city.ahmedabad                                                 81306
    city.lucknow                                                   80837
    city.pune                                                      79555
    city.kolkata                                                   77147
    city.nagpur                                                    72438
    city.goa                                                       71207
    city.chennai                                                   68672
    city.patna                                                     66321
    city.jaipur                                                    48694
    sports.icc-world-cup-2015                                      40537
    business.international-business                                36292
    city.gurgaon                                                   30458
    city.bhubaneswar                                               27885
    tv.news.hindi                                                  27019
    home.campaigns                                                 26834
    citizen.stories                                                26516
    entertainment.english.hollywood                                25587
    city.kochi                                                     25500
    city.guwahati                                                  23939
    city.ranchi                                                    22898
    city.ludhiana                                                  22317
    city.bhopal                                                    22307
    city.thiruvananthapuram                                        21945
    city.noida                                                     19740
    tech.tech-news                                                 19112
    city.vadodara                                                  18865
    home.education                                                 18804
    city.madurai                                                   18730
    city.kanpur                                                    18529
    sports.football                                                18500
    city.coimbatore                                                18474
    city.allahabad                                                 17946
    world.us                                                       17618
    city.surat                                                     16695
    city.mangaluru                                                 16169
    city.indore                                                    15286
    entertainment.tamil.movies.news                                15172
    city.nashik                                                    15071
    city.varanasi                                                  14891
    city.mysuru                                                    14277
    city.hubballi                                                  14099
    news                                                           13974
    edit-page                                                      13699
    city.rajkot                                                    13493
    world.rest-of-world                                            13441
    home.science                                                   13408
    city.navimumbai                                                11726
    city.aurangabad                                                10473
    city.visakhapatnam                                             10265
    city.trichy                                                    10061
    entertainment.kannada.movies.news                               9972
    city.thane                                                      9826
    world.pakistan                                                  9249
    life-style.health-fitness.health-news                           8810
    city.kolhapur                                                   8757
    tech                                                            8493
    entertainment.english.music.news                                7988
    city.kozhikode                                                  7944
    top-headlines                                                   7603
    sports.tennis                                                   7413
    delhi-times                                                     7356
    entertainment.malayalam.movies.news                             7067
    sports.more-sports.others                                       6904
    bombay-times                                                    6828
    entertainment.telugu.movies.news                                6805
    blogs                                                           6796
    lucknow-times                                                   6704
    top-stories                                                     6637
    sports.golf                                                     6582
    entertainment.hindi.music.news                                  6195
    calcutta-times                                                  5717
    city.raipur                                                     5498
    cricket                                                         5451
    pune-times                                                      5390
    hyderabad-times                                                 4960
    world.uk                                                        4917
    bangalore-times                                                 4885
    city.vijayawada                                                 4836
    world.china                                                     4575
    removed                                                         4553
    entertainment.events.mumbai                                     4472
    entertainment.bengali.movies.news                               4284
    city.shimla                                                     4263
    home.sunday-times.deep-focus                                    4245
    nri.community                                                   4175
    entertainment.events.delhi                                      4152
    world.middle-east                                               4082
    auto                                                            4032
    ahmedabad-times                                                 3961
    world.europe                                                    3953
    others.news-interviews                                          3784
    sports.football.epl                                             3637
    home.environment.flora-fauna                                    3616
    life-style.relationships.man-woman                              3555
    world.south-asia                                                3487
    sports.badminton                                                3472
    home.sunday-times.all-that-matters                              3410
    city.dehradun                                                   3181
    tv.news.english                                                 3056
    city.amritsar                                                   3006
    entertainment.events.hyderabad                                  2957
    entertainment.events.others                                     2954
    city.ghaziabad                                                  2843
    sports.racing                                                   2839
    city.agra                                                       2629
    life-style.beauty                                               2585
    sports.cricket.news                                             2495
    entertainment.events.bangalore                                  2465
    tech.it-services                                                2422
    city.puducherry                                                 2401
    sports.more-sports.athletics                                    2352
    entertainment.events.chennai                                    2265
    sports.icc-world-t20-2016                                       2258
    city.meerut                                                     2220
    sports.hockey                                                   2197
    life-style.health-fitness.diet                                  2139
    city.bareilly                                                   2122
    life-style                                                      2103
    life-style.people                                               2093
    sports.nba                                                      2018
    sports.boxing                                                   2001
    tech.computing                                                  1967
    life-style.spotlight                                            1914
    entertainment                                                   1907
    tech.jobs                                                       1864
    life-style.health-fitness.fitness                               1837
    tv.news.kannada                                                 1816
    city.salem                                                      1739
    speak-out                                                       1719
    sports                                                          1709
    entertainment.events.kolkata                                    1613
    sports.chess                                                    1611
    business.personal-finance                                       1592
    entertainment.tamil.music                                       1592
    spirituality                                                    1590
    gadgets-special                                                 1578
    sports.off-the-field                                            1513
    videos                                                          1477
    entertainment.events.lucknow                                    1397
    life-style.books.features                                       1381
    entertainment.events                                            1366
    city.amaravati                                                  1339
    life-style.fashion.style-guide                                  1315
    home.environment.pollution                                      1258
    entertainment.marathi.movies.news                               1254
    city.jammu                                                      1222
    city.cuttack                                                    1222
    tech.social                                                     1212
    life-style.fashion.shows                                        1178
    city.agartala                                                   1158
    tv.news.tamil                                                   1157
    home.environment.developmental-issues                           1156
    city.srinagar                                                   1154
    life-style.food.food-reviews                                    1144
    photos                                                          1144
    city.imphal                                                     1130
    elections.assembly-elections.uttar-pradesh.news                 1097
    sports.cricket.ipl                                              1097
    sports.football.indian-super-league                             1089
    sports.ipl.news                                                 1088
    tv.news.malayalam                                               1074
    home.environment.global-warming                                 1052
    elections.news                                                  1051
    entertainment.kannada.music                                     1040
    interviews                                                      1036
    sports.football.i-league                                        1034
    city.ajmer                                                      1033
    life-style.relationships.parenting                              1023
    life-style.home-garden                                          1007
    city.shillong                                                   1003
    city.jodhpur                                                    1001
    city.jamshedpur                                                  991
    life-style.relationships.work                                    914
    gadgets-news                                                     895
    home.sunday-times                                                889
    sports.football.world-cup-2014                                   878
    entertainment.malayalam.music                                    860
    life-style.food.food-features                                    857
    entertainment.kannada.theatre                                    852
    city.erode                                                       849
    india-news                                                       849
    city.udaipur                                                     827
    sports.more-sports.cycling                                       808
    assembly-elections-2012.uttar-pradesh                            802
    life-style.food.recipes                                          776
    sports.football.champions-league                                 766
    politics.news                                                    765
    sports.more-sports.shooting                                      761
    life-style.fashion.designers                                     760
    tv.news.telugu                                                   758
    entertainment.bhojpuri.movies.news                               754
    life-style.fashion.buzz                                          733
    tv.news.bengali                                                  712
    city.itanagar                                                    695
    tech.mobiles                                                     692
    entertainment.hindi.music.music-videos                           687
    television-highlights                                            681
    entertainment.punjabi.music                                      678
    times-view                                                       676
    sports.new-zealand-in-india-2016                                 664
    regional                                                         656
    city.faridabad                                                   639
    home.environment.the-good-earth                                  610
    assembly-elections-2013.delhi-assembly-elections                 599
    sports.more-sports.snooker.billiards                             594
    entertainment.hindi.music.music-events                           590
    sports.headline1                                                 589
    life-style.health-fitness.de-stress                              583
    life-style.fashion.trends                                        560
    sports.football.fifa-world-cup                                   559
    entertainment.malayalam                                          548
    assembly-elections-2011.west-bengal                              543
    business                                                         527
    elections.assembly-elections.telangana                           527
    tv.news.marathi                                                  522
    entertainment.telugu.music                                       518
    elections.lok-sabha-elections-2019.uttar-pradesh.news            509
    entertainment.hindi.theatre                                      507
    gujarat-assembly-elections                                       504
    entertainment.beauty-pageants.news                               504
    sports.more-sports.wrestling                                     503
    life-style.fashion.celeb-style                                   502
    world.mad-mad-world                                              501
    talk-radio                                                       501
    assembly-elections-2013.rajasthan-assembly-elections             494
    life-style.health-fitness.weight-loss                            474
    entertainment.english.movie-review                               468
    world                                                            468
    delhi-elections-2015                                             462
    entertainment.hindi.movie-review                                 462
    city.gaya                                                        455
    elections.assembly-elections.gujarat                             455
    assembly-elections-2012.goa                                      437
    auto.miscellaneous                                               433
    home.specials.assembly-elections-2014.maharashtra-news           430
    assembly-elections-2013.madhya-pradesh-assembly-elections        423
    city.kohima                                                      418
    sports.cricket.england-in-india-2016                             418
    elections.assembly-elections.madhya-pradesh                      417
    entertainment.punjabi.movies.news                                417
    entertainment.events.kanpur                                      409
    elections.assembly-elections.punjab.news                         407
    elections.assembly-elections.rajasthan                           406
    life-style.relationships.pets                                    390
    life-style.relationships.love-sex                                389
    sports.cricket.icc-world-cup                                     388
    entertainment.gujarati.movies.news                               386
    entertainment.events.gurgaon                                     385
    entertainment.events.kochi                                       378
    nagpur-times                                                     374
    entertainment.hindi.music.music-reviews                          373
    assembly-elections-2011.tamil-nadu                               370
    life-style.health-fitness.every-heart-counts                     370
    entertainment.malayalam.movies                                   366
    more-stories                                                     365
    citizens-grievances                                              360
    elections.lok-sabha-elections-2019.maharashtra.news              341
    specials                                                         330
    entertainment.events.jaipur                                      329
    mocktale                                                         328
    entertainment.bengali.music                                      314
    entertainment.hindi                                              314
    home.environment                                                 314
    tech.how-to                                                      312
    tech.more-gadgets                                                312
    entertainment.events.goa                                         312
    entertainment.events.varanasi                                    308
    entertainment.events.bhopal                                      307
    tech.apps                                                        306
    entertainment.telugu.movies                                      305
    sports.cricket.australia-in-india                                302
    sports.cricket.india-in-england                                  302
    entertainment.tamil.movies.did-you-know                          300
    hyderabad.local-sports                                           295
    life-style.books                                                 294
    city.jind                                                        291
    elections.lok-sabha-elections-2019.delhi.news                    288
    elections.bihar-elections-2015.news                              279
    assembly-elections-2013.chhattisgarh-assembly-elections          279
    sports.cricket.india-domestic.ranji-trophy                       274
    delhi                                                            274
    sports.cricket.india-in-south-africa                             270
    tech.reviews                                                     263
    entertainment.movies                                             262
    asian-games-2014.india-at-incheon                                262
    union-budget                                                     261
    indian-challenge                                                 257
    subverse                                                         256
    sports.cricket.india-in-australia                                251
    sports.hockey.hockey-india-league                                249
    life-style.food-news                                             248
    uttar-pradesh                                                    247
    entertainment.events.indore                                      244
    entertainment.events.nagpur                                      241
    entertainment.kannada.movie-reviews                              237
    budget-2015.union-budget-2015                                    235
    rail-budget                                                      228
    off-the-field                                                    227
    home.specials.assembly-elections-2014.haryana-news               225
    entertainment.hindi.music                                        220
    entertainment.events.pune                                        219
    sports.asian-games                                               218
    entertainment.tamil.movie-reviews                                218
    elections.lok-sabha-elections-2019.bihar.news                    218
    entertainment.bengali.movie-reviews                              213
    life-style.fashion                                               212
    auto.cars                                                        211
    city                                                             209
    tech.gaming                                                      208
    speaking-tree                                                    208
    sports.racing.indian-gp                                          202
    elections.lok-sabha-elections-2019.karnataka.news                197
    sports.football.interviews                                       196
    entertainment.events.ahmedabad                                   193
    entertainment.gujarati.music                                     189
    elections.assembly-elections.goa.news                            189
    life-style.food.drinks-corner                                    188
    politics.politics-specials                                       188
    sports.commonwealth-games                                        186
    sa-aiyar.swaminomics                                             185
    astrology.horoscope                                              182
    life-style.events                                                182
    trivia                                                           181
    sports.india-in-australia                                        181
    sports.tennis.australian-open-2017                               180
    life-style.relationships.ask-the-expert                          175
    stories                                                          175
    war-on-iraq.news                                                 173
    bachi-karkaria.erratica                                          173
    sports.cricket.west-indies-in-india                              172
    entertainment.kannada.movies.did-you-know                        169
    other-racing.others                                              169
    entertainment.marathi.theatre                                    167
    entertainment.hindi.top-10                                       166
    jug-suraiya.jugular-vein                                         165
    entertainment.malayalam.movie-reviews                            163
    entertainment.telugu.movies.did-you-know                         162
    entertainment.telugu                                             162
    sports.football.2017-u-17-world-cup.news                         161
    sports.cricket.england-in-australia                              160
    life-style.books.reviews                                         159
    commonwealth-games-2014.india-at-glasgow                         159
    gaming                                                           157
    quickstir.red-hot                                                156
    life-style.relationships.soul-curry                              156
    sports.cricket.sri-lanka-in-india                                154
    mf-news                                                          154
    entertainment.marathi.music                                      150
    first-look                                                       149
    diwali                                                           148
    sports.pro-kabaddi-league                                        148
    sports.cricket.india-in-sri-lanka                                148
    entertainment.events.noida                                       147
    sports.tennis.interviews                                         146
    indian-challenges                                                145
    maharashtra                                                      143
    elections.lok-sabha-elections-2019.madhya-pradesh.news           143
    entertainment.marathi                                            140
    entertainment.malayalam.movies.previews                          140
    entertainment.kannada.movies                                     140
    sports.tennis.australian-open-2015                               140
    entertainment.malayalam.movies.did-you-know                      137
    sports.india-in-west-indies-2016                                 136
    entertainment.telugu.movies.previews                             136
    budget-2015.rail-budget-2015                                     135
    sports.tennis.australian-open-2018                               134
    tv.news.gujarati                                                 134
    new-year                                                         133
    centre                                                           132
    tv.news                                                          131
    politics                                                         130
    silver-jubilee                                                   128
    elections.lok-sabha-elections-2019.rajasthan.news                128
    hyderabad                                                        127
    assembly-elections-2012.punjab                                   127
    union-budget-2011                                                127
    assembly-elections-2011.assam                                    126
    elections.lok-sabha-elections-2019.gujarat.news                  124
    elections.lok-sabha-elections-2019.kerala.news                   122
    sports.cricket.india-in-west-indies                              122
    elections.lok-sabha-elections-2019.west-bengal.news              122
    sports.rio-2016-olympics.news.athletics                          120
    ganesh-chaturthi                                                 120
    city.rajahmundry                                                 116
    other-top-stories                                                116
    pune                                                             116
    elections.assembly-elections.uttarakhand.news                    115
    mumbai                                                           113
    christmas                                                        111
    entertainment.tamil.movies.previews                              110
    formula-one                                                      109
    sports.tennis.us-open-2016                                       109
    entertainment.events.aurangabad                                  107
    home.specials.2014-assembly-elections.jammu-kashmir-news         106
    durga-puja                                                       105
    himachal-pradesh-assembly-elections                              105
    asian-games-2014.news                                            104
    entertainment.kannada                                            104
    life-style.health-fitness.home-remedies                          104
    sports.hockey.hockey-world-cup                                   104
    leisure                                                          103
    elections.lok-sabha-elections-2019.tamil-nadu.news               103
    entertainment.english.music                                      103
    home.specials.2014-assembly-elections.jharkhand-news             102
    home.environment.wild-wacky                                      102
    life-style.travel                                                100
    sports.cricket.asia-cup                                           99
    entertainment.bengali.movies.did-you-know                         98
    tarun-vijay.the-right-view                                        97
    entertainment.gujarati.theatre                                    97
    entertainment.telugu.movie-reviews                                97
    life-style.health-fitness                                         96
    campaign-trail                                                    96
    elections.lok-sabha-elections-2019.telangana.news                 95
    shobhaa-de.politically-incorrect                                  95
    entertainment.events.patna                                        95
    young-india-votes.news                                            95
    life-style.relationships                                          94
    sports.tennis.wimbledon-2014                                      93
    indian-badminton-league                                           93
    life-style.food                                                   92
    commonwealth-games-2014.news                                      92
    elections.lok-sabha-elections-2019.goa.news                       92
    sports.football.under-17-world-cup.news                           89
    players-to-watch                                                  89
    companies                                                         89
    sports.cricket.pakistan-in-australia                              88
    movie-reviews                                                     87
    tv.trade-news.hindi                                               87
    also-read                                                         87
    sports.new-zealand-in-india-2016.interviews                       86
    elections.assembly-elections.manipur.news                         86
    quickstir.lifestyle                                               86
    sports.golf.interviews                                            85
    valentines-day                                                    84
    entertainment.events.bhubaneswar                                  84
    gurcharan-das.men-ideas                                           84
    sports.tennis.french-open-2017                                    84
    sports.tennis.wimbledon-2016                                      84
    shashi-tharoor.shashi-on-sunday                                   83
    life-style.books.interviews                                       82
    sports.rio-2016-olympics.news.miscellaneous                       82
    bangalore                                                         82
    chidanand-rajghatta.indiaspora                                    82
    sars-scare.news                                                   81
    sports.rio-2016-olympics.news.aquatics                            81
    pro-kabaddi-league                                                81
    entertainment.bengali.theatre                                     80
    india-business-news-wire                                          79
    entertainment.marathi.movies                                      79
    entertainment.events.raipur                                       78
    entertainment.events.kolhapur                                     77
    republic-day                                                      76
    sports.expert-column.sunil-gavaskar                               76
    hot-on-the-web                                                    75
    others.parties                                                    75
    sports.cricket.australia-in-south-africa                          75
    rajasthan                                                         74
    best-products.todays-deals.paytm-mall                             73
    sports.rio-2016-olympics.news.tennis                              73
    elections.lok-sabha-elections-2019.haryana.news                   73
    iraq-hostage-drama                                                73
    entertainment.events.nashik                                       72
    quote-of-the-day                                                  72
    life-style.books.book-launches                                    71
    sports.football.copa-america-2015                                 70
    sports.wwe                                                        68
    elections.lok-sabha-elections-2019.punjab.news                    68
    birla-will-saga                                                   68
    gurgaon-shamed                                                    67
    entertainment.telugu.movies.box-office                            67
    it-services-news-wire                                             67
    indo-pak-monitor.news                                             67
    sports.tennis.wimbledon-2017                                      67
    sports.racing.interviews                                          66
    sports.cricket.south-africa-in-australia                          65
    pre-budget                                                        65
    sports.rio-2016-olympics.news.wrestling                           65
    sports.tennis.us-open-2014                                        64
    sports.rio-2016-olympics.india-in-olympics-2016.wrestling         64
    navaratra-2013                                                    64
    young-india-votes.from-the-states                                 64
    sports.icc-world-t20-2016.interviews                              63
    assembly-elections-2011.kerala                                    63
    2010-stars                                                        61
    west-bengal                                                       61
    sports.cricket                                                    61
    onam                                                              61
    sports.rio-2016-olympics.news.badminton                           60
    assembly-elections-2013.mizoram-assembly-elections                60
    madhya-pradesh                                                    60
    vinita-dawra-nangia                                               59
    election-2008                                                     59
    sports.cricket.india-domestic                                     59
    reporters-diary                                                   58
    assembly-elections-2012.manipur                                   58
    sports.cricket.bangladesh-in-india                                58
    sports.cricket.new-zealand-in-india                               58
    karnatka                                                          57
    jug-suraiya.second-opinion                                        57
    sports.rio-2016-olympics.india-in-olympics-2016.badminton         57
    ranji-trophy                                                      57
    times-quality-of-life-survey-                                     57
    olympics                                                          57
    tech.pcs                                                          57
    sports.cricket.icc-womens-world-cup-2017                          56
    entertainment.events.chandigarh                                   56
    sports.ipl.interviews                                             56
    sports.tennis.international-premier-tennis-league                 56
    gandhi-jayanti                                                    56
    sports.rio-2016-olympics.news.gymnastics                          55
    entertainment.gujarati.movies                                     54
    elections.lok-sabha-elections-2019.odisha.news                    54
    nagpur                                                            54
    economic-survey                                                   54
    elections.assembly-elections.andhra-pradesh                       54
    swine-flu-outbreak-in-india                                       53
    toi-social-impact-awards-2013                                     53
    star-candidates                                                   52
    home.infographics                                                 51
    sports.football.world-cup-2014.interviews                         50
    questions-and-answers                                             50
    sports.cricket.west-indies-in-england                             50
    elections.lok-sabha-elections-2019.uttarakhand.news               50
    sports.football.indian-super-league.interviews                    50
    elections.assembly-elections.chhattisgarh                         50
    sports.2016-asia-cup                                              49
    baroda                                                            49
    auto.bikes                                                        49
    india-hopes                                                       48
    sports.tennis.wimbledon-2018                                      48
    corporate-espionage-in-ministries                                 48
    sports.cricket.bangladesh-in-new-zealand                          47
    life-style.debate                                                 47
    venues                                                            46
    swapan-dasgupta.right-wrong                                       46
    sports.tennis.french-open                                         45
    sports.nfl.news                                                   45
    sports.cricket.india-sri-lanka-bangladesh-tri-series              44
    queens-baton-relay                                                43
    rail-budget-2011                                                  43
    raksha-bandhan-2013                                               43
    chandigarh                                                        42
    sports.cricket.u-19-world-cup                                     42
    times-news-radio                                                  42
    other-racing.a1-gp                                                42
    tv.trade-news.tamil                                               42
    ayodhya-imbroglio                                                 41
    business.faqs.income-tax-faqs                                     41
    viral-news                                                        41
    budget-2015.common-man                                            41
    odisha                                                            41
    pravasi-bhartiya-news                                             41
    sports.football.epl.interviews                                    41
    business.mf-simplified.mf-news                                    41
    war-on-iraq.opinion                                               40
    life-style.health-fitness.health                                  40
    mj-akbar.the-siege-within                                         40
    elections.lok-sabha-elections-2019.jharkhand.news                 39
    chhattisgarh                                                      39
    passport-pangs                                                    39
    litfest.litfest-delhi.news                                        39
    miscellaneous                                                     39
    sports.rio-2016-olympics.news.shooting                            39
    santosh-desai.city-city-bang-bang                                 39
    sports.cricket.icc-world-cup.teams.new-zealand.news               39
    sports.yearender-2016                                             39
    schedule                                                          38
    haryana                                                           38
    elections.lok-sabha-elections-2019.punjab                         38
    sports.nba.off-the-court                                          38
    teachers-day                                                      37
    sports.headline5                                                  37
    players-profiles                                                  37
    entertainment.marathi.movie-reviews                               37
    sports.more-sports                                                37
    anti-terror-law                                                   36
    sports.rio-2016-olympics.news.boxing                              36
    uttarakhand                                                       36
    sports.rio-2016-olympics.news.hockey                              36
    aap-crisis                                                        36
    elections.lok-sabha-elections-2019.andhra-pradesh.news            36
    sports.rio-2016-olympics.india-in-olympics-2016.hockey            36
    history                                                           36
    good-day-good-news                                                35
    text-after-pic                                                    35
    budget-2015.you-taxes-2015                                        35
    top-news-of-2011                                                  35
    sports.hockey.champions-trophy-2014                               35
    heads-and-tales                                                   35
    tv.trade-news.malayalam                                           34
    home.auto                                                         34
    young-guns                                                        34
    life-style.food.quick-food                                        34
    elections.assembly-elections.mizoram                              34
    sports.cricket.west-indies-in-new-zealand                         34
    ambani                                                            33
    sports.cricket.icc-world-cup.teams.england.news                   33
    pm-on-china-visit.news                                            33
    life-style.specials                                               33
    rupee-symbol-survey                                               33
    sports.rio-2016-olympics.news.cycling                             33
    sports.cricket.pakistan-in-new-zealand                            33
    international-womens-day                                          33
    asian-games-2014.venues                                           33
    andhra-pradesh                                                    33
    elections.assembly-elections.odisha                               32
    budget-2015.politics-budget-2015                                  32
    sports.rio-2016-olympics.india-in-olympics-2016.athletics         32
    gujarat                                                           32
    sports.cricket.south-africa-in-india                              32
    sports.tennis.us-open-2019                                        31
    elections.lok-sabha-elections-2019.assam.news                     31
    humour.social-humour                                              31
    budget-2015.economic-survey-2015                                  31
    liverpool                                                         31
    business.mf-simplified.jargon-busters.debt                        31
    sports.rio-2016-olympics.india-in-olympics-2016.tennis            31
    sports.rio-2016-olympics.news.football                            30
    entertainment.kannada.movies.previews                             30
    entertainment.gujarati                                            30
    entertainment.punjabi.movies.did-you-know                         30
    bihar-assembly-polls                                              29
    goa                                                               29
    delhis-century                                                    29
    entertainment.tamil                                               29
    sports.headline4                                                  29
    sports.rio-2016-olympics.india-in-olympics-2016.shooting          29
    most-searched-products.todays-deals.amazon                        29
    articles                                                          28
    young-india-votes.deep-focus                                      28
    vote-maadi                                                        28
    sports.rio-2016-olympics.news                                     28
    assembly-elections-2012.uttarakhand                               28
    pm-on-europe-tour.news                                            28
    best-products.fashion.accessories                                 28
    tamil-nadu                                                        27
    lohri                                                             27
    mumbai-pluses                                                     27
    times-litfest-bengaluru-complete-coverage                         27
    complete-results-2009                                             27
    life-style.fashion.specials                                       27
    india-challenges                                                  27
    nrs-2003                                                          27
    features                                                          27
    sports.tennis.wimbledon-2019                                      27
    tv.trade-news.telugu                                              26
    sports.racing.schumacher-battling-for-life                        26
    entertainment.hindi.specials                                      26
    sports.cricket.sri-lanka-in-south-africa                          26
    people                                                            26
    teams                                                             26
    kolkata                                                           25
    sunderland                                                        25
    goa-plus                                                          25
    trend-tracking                                                    25
    sports.winter-olympics                                            25
    best-products.beauty.skin-care                                    25
    sports.cricket.icc-womens-world-t20                               25
    arsenal                                                           25
    sports.cricket.icc-world-cup.teams.india                          25
    sports.tennis.french-open-2018                                    25
    manchester-city                                                   25
    recipes                                                           25
    sports.cricket.sri-lanka-in-pakistan                              24
    elections.lok-sabha-elections-2019.chandigarh.news                24
    sports.cricket.england-in-new-zealand                             24
    bolton-wanderers                                                  24
    telangana                                                         24
    chelsea                                                           24
    sports.cricket.england-in-bangladesh                              24
    life-style.relationships.specials                                 24
    sports.rio-2016-olympics.india-in-olympics-2016.gymnastics        24
    past-winners                                                      24
    health-news-corner                                                23
    ad-links                                                          23
    asian-games-2014.medal-tally                                      23
    sports.cricket.icc-world-cup.teams.pakistan.news                  23
    sports.rio-2016-olympics.profiles.athletics                       23
    elections.lok-sabha-elections-2019.uttar-pradesh                  23
    audi-cars                                                         23
    spirituality.https.timesofindia-speakingtree-in.article           23
    world-environment-day.world-environment-day-stories               22
    everton                                                           22
    life-style.parenting.moments                                      22
    young-india-votes.gallup-poll                                     22
    stoke-city                                                        22
    tendulkar                                                         22
    teams-players                                                     22
    friendship-day                                                    22
    mahashivratri                                                     22
    manchester-united                                                 22
    quickstir.entertainment                                           22
    janmashtami-2013                                                  22
    tottenham-hotspur                                                 22
    elections.assembly-elections.maharashtra                          22
    fulham                                                            22
    sports.cricket.afghanistan-tour-of-india                          21
    sports.cricket.new-zealand-in-australia                           21
    blackburn-rovers                                                  21
    conversion.ghar-wapsi-complete-coverage                           21
    aston-villa                                                       21
    cwg-history                                                       21
    entertainment.marathi.movies.did-you-know                         21
    schedule-results                                                  21
    delhi-ncr-pluses                                                  20
    drivers                                                           20
    wigan-athletic                                                    20
    maharashtra-ngo                                                   20
    regional.tamil                                                    20
    punjab                                                            20
    sports.rio-2016-olympics.news.judo                                20
    entertainment.events.ranchi                                       20
    bihar                                                             20
    elections.assembly-elections.himachal-pradesh                     20
    pune-pluses                                                       20
    newcastle-united                                                  20
    2011-top-stories                                                  20
    sports.rio-2016-olympics.news.archery                             20
    twin-city-pluses                                                  20
    sports.rio-2016-olympics.news.golf                                20
    maharashtra-pluses                                                20
    best-products.fashion.womens-fashion                              20
    elections.interactives                                            20
    2011-top-slideshow                                                20
    dussehra                                                          20
    up-pluses                                                         20
    eid-ul-fitr                                                       20
    fight-dengue.news                                                 20
    auto.launches                                                     20
    asian-games-2014.sports                                           19
    food-facts                                                        19
    sports.ipl-2015.news                                              19
    elections.lok-sabha-elections-2019                                19
    sports.football.epl.club-profiles.profiles                        19
    sports.tennis.french-open-2016                                    19
    life-style.food.specials                                          19
    world-heart-day                                                   19
    bobilli-vijay-kumar                                               19
    sports.cricket.australia-in-england                               19
    entertainment.punjabi.movies                                      19
    donts                                                             19
    top-news-of-2012                                                  19
    venture-capital                                                   19
    2014-sochi-winter-olympics                                        19
    advisory                                                          19
    life-style-landing.health-fitness.health-news                     19
    west-bengal-pluses                                                19
    tv.trade-news.gujarati                                            19
    best-products.amazon-deals                                        19
    circuits.formula-one                                              19
    hay-festival                                                      19
    best-products.gift-ideas                                          19
    tv.trade-news.kannada                                             19
    sports.cricket.icc-world-cup.teams.australia.news                 18
    sino-indian-ties                                                  18
    life-style.health-fitness.specials                                18
    stars-speak                                                       18
    sports.cricket.pakistan-v-west-indies                             18
    sports.tennis.australian-open                                     18
    entertainment.marathi.movies.previews                             18
    entertainment.english.music.music-videos                          18
    sports.tennis.australian-open-2015.indian-challenge               18
    only-in-america                                                   18
    elections.lok-sabha-elections-2019.himachal-pradesh.news          18
    down-memory-lane                                                  18
    all-colour-edition                                                17
    sports.cricket.australia-in-bangladesh                            17
    muharram                                                          17
    best-products.mobile-phones                                       17
    sports.rio-2016-olympics.news.weightlifting                       17
    sports.cricket.pakistan-in-england                                17
    entertainment.bengali.movies.previews                             17
    sports.sri-lanka-in-india                                         17
    entertainment.bengali                                             17
    budget-2015                                                       17
    madras-plus                                                       17
    budget-2015.student                                               17
    thirupuram                                                        17
    childrens-day                                                     17
    analysis                                                          16
    ahmedabad-events                                                  16
    astrology.hindu-mythology                                         16
    entertainment.english                                             16
    pre-budget-2011                                                   16
    cwg-stars                                                         16
    assembly-elections-2011.puducherry                                16
    red-hot                                                           16
    elections.lok-sabha-elections-2019.west-bengal                    16
    elections.assembly-elections.uttar-pradesh.interactives           16
    news-features                                                     16
    assembly-elections-2011.assembly-elections-results                15
    sports.cricket.icc-world-cup.teams.south-africa.news              15
    sports.cricket.sri-lanka-in-west-indies                           15
    sports.cricket.india-in-new-zealand                               15
    sports.cricket.icc-world-cup.player-of-the-day                    15
    elections.assembly-elections.uttar-pradesh                        15
    young-india-votes.talking-point                                   15
    missile-game                                                      15
    sports.ipl                                                        15
    commonwealth-games-2014.medals-tally                              15
    west-bromwich-albion                                              15
    tv.trade-news.marathi                                             15
    spirituality.https.timesofindia-speakingtree-in.allslides         15
    world-aids-day                                                    15
    sports.cricket.south-africa-in-england                            15
    business.faqs.aadhar-faqs                                         15
    fathers-day-2013                                                  14
    education                                                         14
    sports.cricket.pakistan-in-west-indies                            14
    sports.rio-2016-olympics.india-in-olympics-2016.boxing            14
    sports.cricket.bangladesh-in-south-africa                         14
    year-ender-2015.december                                          14
    columbia-crash                                                    14
    entertainment.tamil.movies                                        14
    raksha-bandhan.rakhi-stories                                      14
    astrology.rituals-puja                                            14
    elections.assembly-elections.haryana                              14
    sports.cricket.world-cup-qualifiers                               14
    uttar-pradesh-ngo                                                 14
    get-healthy-get-fit                                               14
    elections.lok-sabha-elections-2019.chandigarh                     14
    gujarat-pluses                                                    14
    the-himalayan-blunder                                             14
    sports.nba.team-profiles                                          14
    others.previews                                                   14
    world-no-tobacco-day.no-tobacco-day-stories                       14
    sports.tennis.australian-open-2015.interviews                     14
    sports.rio-2016-olympics.india-in-olympics-2016.golf              13
    sports.football.2017-u-17-world-cup                               13
    guru-nanak-jayanti                                                13
    sports.football.epl.past-winners                                  13
    gujarat-ngo                                                       13
    humour.third-edit                                                 13
    best-products.electronics.cameras                                 13
    elections.lok-sabha-elections-2019.haryana                        13
    toi-social-impact-awards-2015                                     13
    best-products.beauty.makeup-tips                                  13
    news.hardware                                                     13
    afghan-children                                                   13
    entertainment.bengali.movies                                      13
    iit-bengaluru                                                     13
    timeline                                                          13
    karnatka-ngo                                                      13
    dos                                                               13
    best-products.fashion.mens-fashion                                13
    jaipur                                                            13
    entertainment.bhojpuri.movies.did-you-know                        12
    travel                                                            12
    entertainment.punjabi.movies.previews                             12
    entertainment.english.specials                                    12
    spotlight                                                         12
    chevrolet-cars                                                    12
    life-style.health-fitness.homeopathy                              12
    sports.rio-2016-olympics.news.basketball                          12
    diwali-rituals                                                    12
    year-ender-2015.february                                          12
    user-generated                                                    12
    year-ender-2015.january                                           12
    budget-2015.women                                                 12
    elections.lok-sabha-elections-2019.jammu-and-kashmir.news         12
    mp-pluses                                                         12
    sports.football.under-17-world-cup                                12
    top-news-of-2008                                                  12
    years-headline-makers                                             12
    sports.rio-2016-olympics.india-in-olympics-2016.archery           12
    home                                                              12
    sports.headline2                                                  12
    venus-transit                                                     11
    jugular-vein                                                      11
    best-products.fashion.footwear                                    11
    sports.tennis.us-open-2018                                        11
    life-style.listen-to-your-sugar                                   11
    friendship-day-2012                                               11
    elections.assembly-elections.punjab                               11
    pakistan-postcard                                                 11
    hyundai-cars                                                      11
    education-fest.united-kingdom.stories                             11
    regional.kannada                                                  11
    sports.racing.f1-teams                                            11
    indians-abroad                                                    11
    player-profile                                                    11
    economic-survey-2011                                              11
    life-style.parenting.toddler-year-and-beyond                      11
    sports.cricket.australia-in-new-zealand                           11
    sports.cricket.icc-world-cup.teams.sri-lanka.news                 11
    poll-pourri                                                       11
    sports.india-in-zimbabwe-2016                                     10
    deep.pathankot-terrorist-attack.questions-and-answers             10
    ahmedabad                                                         10
    young-india-votes.aaj-ka-neta                                     10
    comunistst                                                        10
    fathers-day.fathers-day-stories                                   10
    kerala                                                            10
    other-racing.moto-gp                                              10
    life-style.food.bar-reviews                                       10
    2013-the-year-sachin-bids-adieu.more-sports-2013                  10
    budget-2015.opinion                                               10
    top-videos                                                        10
    lodge-a-complaint                                                 10
    world-cup-venues                                                  10
    apply-for                                                         10
    chidanand-rajghatta.desiderata                                    10
    tv.specials.kannada                                               10
    unsw.study-at-unsw                                                10
    2008-in-pictures                                                  10
    sports.racing.tech-tonic                                          10
    uber-cab-rape-case                                                10
    tech-news-news-wire                                               10
    bmw-cars                                                          10
    chattisgarh                                                       10
    best-products.electronics.headphones                              10
    sports.rio-2016-olympics.profiles.mens-hockey                     10
    mahavir-jayanti.mahavir-jayanti-stories                            9
    north-east-pluses                                                  9
    janmashtami.janmashtami-stories                                    9
    matches-results                                                    9
    nagpur-pluses                                                      9
    swaminomics                                                        9
    honda-cars                                                         9
    budget-2015.homemaker                                              9
    glamour-the-game.stories                                           9
    air-pollution                                                      9
    new-test-audio                                                     9
    brandwire.services.education                                       9
    special-coverage.other-specials                                    9
    west-bengal-ngo                                                    9
    sports.cricket.icc-world-cup.teams.west-indies.news                9
    best-products.kitchen-and-dining.small-appliances                  9
    years-young-achievers                                              9
    sports.rio-2016-olympics.news.volleyball                           9
    humour.mocktale                                                    9
    business.mf-simplified.jargon-busters.equity                       9
    world-cancer-day                                                   9
    shop                                                               9
    certificates                                                       9
    madhya-pradesh-ngo                                                 9
    world-environment-day                                              9
    popular-shows                                                      9
    everything-you-need-to-know                                        9
    sports.rio-2016-olympics.news.canoe-kayak                          9
    world.us.india-and-us                                              9
    teams-profile                                                      9
    vday-articles                                                      9
    sports.cricket.sri-lanka-in-new-zealand                            9
    elections.bihar-elections-2015.news-coverage                       9
    sports.rio-2016-olympics.news.sailing                              8
    year-ender-2015.march                                              8
    himachal-pradesh                                                   8
    health-case-studies                                                8
    mahindra-cars                                                      8
    brandwire.technology.internet-apps                                 8
    elections.assembly-elections.goa                                   8
    best-products.home-decor-and-garden.living-room-decor              8
    sports.cricket.sri-lanka-in-australia                              8
    sports.tennis.top-stories.tennis-atp                               8
    brandwire.media-entertainment.newspapers-magazines-books           8
    2013-the-year-sachin-bids-adieu.football-2013                      8
    preeti-shenoy                                                      8
    editorialt                                                         8
    advanis-us-visit                                                   8
    sports.football.indian-super-league.team-profiles                  8
    business.faqs.gst-faqs                                             8
    sports.asian-games-2018                                            8
    young-turks                                                        8
    entertainment.hindi.music.singer-of-the-week                       8
    sports.hockey.hockey-world-cup-2014                                8
    lifespan-news                                                      8
    sports.cricket.south-africa-in-sri-lanka                           8
    indias-vision                                                      8
    business.mf-simplified.faq                                         8
    scorecard-and-statistics                                           8
    2013-the-year-sachin-bids-adieu.tennis-2013                        8
    sports.hockey.hockey-india-league.interviews                       8
    sports.cricket.icc-world-cup.teams.england                         8
    elections.lok-sabha-elections-2019.tripura.news                    8
    best-products.beauty.grooming                                      8
    nepal-india-earthquake.opinion                                     8
    pms-us-visit                                                       8
    iim-fee-row                                                        8
    sports.cricket.bangladesh-in-west-indies                           8
    profiles.india-profiles                                            8
    delhi-ncr                                                          7
    actresses                                                          7
    sports.cricket.icc-world-test-championship                         7
    sports.headline3                                                   7
    ballot-talk                                                        7
    did-you-know                                                       6
    tv.trade-news.bengali                                              6
    auto.reviews                                                       6
    life-style.parenting.teen                                          5
    life-style.parenting.pregnancy                                     4
    life-style.parenting.getting-pregnant                              4
    sports.headline6                                                   3
    sports.tokyo-olympics                                              2
    life-style.parenting.ask-the-expert                                2
    world.us.us-and-world                                              2
    party-manifestos                                                   2
    world.us.gun-violence-and-crimes                                   1
    times-fact-check.news                                              1
    Name: headline_category, dtype: int64




```python
df1['headline_text'].str.len().hist()
plt.show()
```


![png](GRIP-Task7-StockMarketPredictionUsingNumericalandTextualAnalysis_files/GRIP-Task7-StockMarketPredictionUsingNumericalandTextualAnalysis_62_0.png)


* Graph shows that news headlines range from 10 to 120 characters generally.


```python
def basic_clean(text):
        wnl = nltk.stem.WordNetLemmatizer()
        stopwords = nltk.corpus.stopwords.words('english')
        words = re.sub(r'[^\w\s]', '', text).split()
        return [wnl.lemmatize(word) for word in words if word not in stopwords]
```


```python
words = basic_clean(''.join(str(df1['headline_text'].tolist())))
words[:10]
```




    ['win',
     'cena',
     'satisfying',
     'defeating',
     'undertaker',
     'bigger',
     'roman',
     'reign',
     'Status',
     'quo']



# ii) N-Gram Analysis
* In the fields of computational linguistics and probability, an n-gram is a contiguous sequence of n items from a given sample of text or speech. The items can be phonemes, syllables, letters, words or base pairs according to the application. The n-grams typically are collected from a text or speech corpus. When the items are words, n-grams may also be called shingles.
## N-gram Analysis - Unigram, Bigram and Trigram

# a) Unigram Analysis


```python
words_unigram_series = (pd.Series(nltk.ngrams(words, 1)).value_counts())[:20]
```


```python
words_unigram_series.sort_values().plot.barh(color='lightcoral', width=.9, figsize=(12, 8))
plt.title('20 Most Frequently Occuring Unigrams - India News Headlines')
plt.ylabel('Bigram')
plt.xlabel('# of Occurances')
```




    Text(0.5, 0, '# of Occurances')




![png](GRIP-Task7-StockMarketPredictionUsingNumericalandTextualAnalysis_files/GRIP-Task7-StockMarketPredictionUsingNumericalandTextualAnalysis_69_1.png)


# b) Bigram Analysis


```python
words_bigrams_series = (pd.Series(nltk.ngrams(words, 2)).value_counts())[:20]
```


```python
words_bigrams_series.sort_values().plot.barh(color='thistle', width=.9, figsize=(12, 8))
plt.title('20 Most Frequently Occuring Bigrams - India News Headlines')
plt.ylabel('Bigram')
plt.xlabel('# of Occurances')
```




    Text(0.5, 0, '# of Occurances')




![png](GRIP-Task7-StockMarketPredictionUsingNumericalandTextualAnalysis_files/GRIP-Task7-StockMarketPredictionUsingNumericalandTextualAnalysis_72_1.png)


# c) Trigram Analysis


```python
words_trigrams_series = (pd.Series(nltk.ngrams(words, 3)).value_counts())[:20]
words_trigrams_series.sort_values().plot.barh(color='darksalmon', width=.9, figsize=(12, 8))
plt.title('20 Most Frequently Occuring Trigrams - India News Headlines')
plt.ylabel('Trigram')
plt.xlabel('# of Occurances')
```




    Text(0.5, 0, '# of Occurances')




![png](GRIP-Task7-StockMarketPredictionUsingNumericalandTextualAnalysis_files/GRIP-Task7-StockMarketPredictionUsingNumericalandTextualAnalysis_74_1.png)


# Conclusion
### Inorder to complete this GRIP-Task 7 , we have created a hybrid model for stock price/performance prediction using numerical analysis of historical stock prices, and sentimental analysis of news headlines Stock to analyze and predict - SENSEX (S&P BSE SENSEX) is also completed and in addition to the same we have done the n gram analysis and have ploted the outputs graphically.
## Completed Task 7.
### Thank you for going through this solution
