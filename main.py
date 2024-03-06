# pip install streamlit fbprophet yfinance plotly
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date,datetime,timedelta
import matplotlib.pyplot as plt
import yfinance as yf
from plotly import graph_objs as go
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from sklearn.metrics import r2_score

def normaltest(x):
    print(stats.normaltest(x))
    fig, ax = plt.subplots()
    ax.hist(x ,bins=20)
    ax.set_title("Histogram of data")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    if(stats.normaltest(x)[1]<0.05):
        st.write("***Not normally distributed***")
        st.subheader('Apply Log() to the Data')
        fig, ax = plt.subplots()
        ax.hist(np.log(x) ,bins=20)
        ax.set_title("Histogram of data")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        if(stats.normaltest(np.log(x))[1]<0.05):
            st.write("***Log (Data): not normally distributed***")
        else:
            st.write(" ***Log(Data): normally distributed***")
        
    else:
        st.write("***Normally distributed***")
        
                
def check_stationarity(series):

    result = adfuller(series.values)

    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    if (result[1] <= 0.05) & (result[4]['5%'] > result[0]):
        st.write(":green[Stationary]")
        return 1
    else:
        st.write(":red[Non-stationary]")
        return 0

def plot_acf_pacf(series):
    fig, ax = plt.subplots(2,1,figsize=(10,10))
    ax[0] = plot_acf(series, ax=ax[0], lags=50)
    ax[1] = plot_pacf(series, ax=ax[1], lags=50)
    st.pyplot(fig)
  

    

st.title('Stock Forecast App')
n_years = st.slider('Years of prediction:', 0.5, 4.0, 0.5,0.5)
period = n_years * 364


TODAY = datetime.now()
START = TODAY- timedelta(days=period)

stocks = ('GOOG', 'AAPL', 'SPY', 'IBM')
selected_stock = st.selectbox('Select dataset for prediction', stocks)



@st.cache_data
def load_data(ticker,START):
    data = yf.download(ticker, START, TODAY)
    print(START)
    data.reset_index(inplace=True)
    return data



def AutoArima_AIC(series):
    x = len(data)
    size = int(x * 0.8)
    index = x - size
    train_data = data.iloc[:size]
    test_data = data.iloc[size:]  
    model=auto_arima(train_data['Close'],start_p=0,d=1,start_q=0,
          max_p=5,max_d=5,max_q=5, start_P=0,
          D=1, start_Q=0, max_P=5,max_D=5,
          max_Q=5, m=7, seasonal=True,
          error_action='warn',trace=True,
          supress_warnings=True,stepwise=True,
          random_state=30,n_fits=60)
    model.fit(train_data['Close'])
    forecast = model.predict(n_periods=len(test_data))
    forecast = pd.DataFrame(forecast,index = test_data.index,columns=['Prediction'])
    st.write(model)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data['Date'], y=train_data['Close'], name="Train"))
    fig1.add_trace(go.Scatter(x=test_data['Date'], y=test_data['Close'], name="Test"))
    fig1.add_trace(go.Scatter(x=test_data['Date'], y=forecast['Prediction'], name="Forecast"))

    
    st.plotly_chart(fig1)
   
   
    
def AutoArima_BIC(series):
    x = len(data)
    size = int(x * 0.8)
    index = x - size
    train_data = data.iloc[:size]
    test_data = data.iloc[size:]  
    model=auto_arima(train_data['Close'],start_p=0,d=1,start_q=0,
          max_p=5,max_d=5,max_q=5, start_P=0,
          D=1, start_Q=0, max_P=5,max_D=5,
          max_Q=5, m=7, seasonal=True,
          error_action='warn',trace=True,
          supress_warnings=True,stepwise=True,
          random_state=30,n_fits=60,information_criterion="bic")
    model.fit(train_data['Close'])
    forecast = model.predict(n_periods=len(test_data))
    forecast = pd.DataFrame(forecast,index = test_data.index,columns=['Prediction'])
    st.write(model)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data['Date'], y=train_data['Close'], name="Train"))
    fig1.add_trace(go.Scatter(x=test_data['Date'], y=test_data['Close'], name="Test"))
    fig1.add_trace(go.Scatter(x=test_data['Date'], y=forecast['Prediction'], name="Forecast"))
    
    st.plotly_chart(fig1)

def AutoArima_HQIG(series):
    x = len(data)
    size = int(x * 0.8)
    index = x - size
    train_data = data.iloc[:size]
    test_data = data.iloc[size:]  
    model=auto_arima(train_data['Close'],start_p=0,d=1,start_q=0,
          max_p=5,max_d=5,max_q=5, start_P=0,
          D=1, start_Q=0, max_P=5,max_D=5,
          max_Q=5, m=7, seasonal=True,
          error_action='warn',trace=True,
          supress_warnings=True,stepwise=True,
          random_state=30,n_fits=60,information_criterion="hqic",)
    model.fit(train_data['Close'])
    forecast = model.predict(n_periods=len(test_data))
    forecast = pd.DataFrame(forecast,index = test_data.index,columns=['Prediction'])
    st.write(model)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data['Date'], y=train_data['Close'], name="Train"))
    fig1.add_trace(go.Scatter(x=test_data['Date'], y=test_data['Close'], name="Test"))
    fig1.add_trace(go.Scatter(x=test_data['Date'], y=forecast['Prediction'], name="Forecast"))
    
    st.plotly_chart(fig1)

def AutoArima_OOB(series):
    x = len(data)
    size = int(x * 0.8)
    index = x - size
    train_data = data.iloc[:size]
    test_data = data.iloc[size:]  
    model=auto_arima(train_data['Close'],start_p=0,d=1,start_q=0,
          max_p=5,max_d=5,max_q=5, start_P=0,
          D=1, start_Q=0, max_P=5,max_D=5,
          max_Q=5, m=7, seasonal=True,
          error_action='warn',trace=True,
          supress_warnings=True,stepwise=True,
          random_state=30,n_fits=60,information_criterion="oob")
    model.fit(train_data['Close'])
    forecast = model.predict(n_periods=len(test_data))
    forecast = pd.DataFrame(forecast,index = test_data.index,columns=['Prediction'])
    st.write(model)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data['Date'], y=train_data['Close'], name="Train"))
    fig1.add_trace(go.Scatter(x=test_data['Date'], y=test_data['Close'], name="Test"))
    fig1.add_trace(go.Scatter(x=test_data['Date'], y=forecast['Prediction'], name="Forecast"))
    
    st.plotly_chart(fig1)

data_load_state = st.text('Loading data...')
data = load_data(selected_stock,START)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
print(data['Close'])
plot_raw_data()

st.header('Check Data Normality')
normaltest(data['Close'])
close_dff = pd.DataFrame()
close_dff['Differenced_Data'] = data['Close'] - data['Close'].shift(1)
close_dff = close_dff.dropna()
st.header('Check Data Stationarity')
check = check_stationarity(data['Close'])
if check == 0:
    st.subheader('Apply Differenced Data')
    
    check_stationarity(close_dff['Differenced_Data'])
    st.line_chart(close_dff['Differenced_Data'])


plot_acf_pacf(close_dff['Differenced_Data'])   

st.header('Auto Arima Forcast') 
st.subheader('AIC Minimization')
AutoArima_AIC(data)
st.subheader('BIC Minimization')
AutoArima_BIC(data)
st.subheader('HQIC Minimization')
AutoArima_HQIG(data)
st.subheader('Out of Bag Minimization')
AutoArima_OOB(data)     

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
    
    
    
    
