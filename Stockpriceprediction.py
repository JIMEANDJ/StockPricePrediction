#!/usr/bin/env python
# coding: utf-8

# In[6]:


pip install pandas


# In[7]:


pip install yfinance


# In[8]:


pip install matplotlib


# In[9]:


pip install mpl-finance


# In[10]:


pip install scikit-learn


# In[11]:


pip install beautifulsoup4


# In[12]:


import yfinance as yf
import datetime as dt 

start = dt.datetime(2023, 1 , 1)
end = dt.datetime.now()
nvda = yf.download('NVDA', start=start, end=end)


# In[13]:


df = yf.download('NVDA', start=start, end=end)
# de esta manera descargamos desde la api de yahoo. 


# In[16]:


df.to_csv('NVDA.csv', sep =";")
df.to_excel('nvda.xlsx')
df.to_html('nvda.html')
df.to_json('nvda.json')
#esta funcion guarda la data descargada en el formato q desees usas, la ventaja es que para backtesting no tiens q consultar la api porq tiens ya la data/



# In[ ]:


#df = pd.read_csv( "NVDA.csv" , sep = ";" )
#df = pd.read_excel( "nvda.xlsx" )
#df = pd.read_html( "nvda.html" )
#df = pd.read_json( "nvda.json" )
# estas funciones son para leer los archivos guardados en el formato deseado#


# In[14]:


import matplotlib.pyplot as plt
from matplotlib import style # para poder dar estilo

df ['Adj Close'].plot()

style.use( 'fast' )
plt.ylabel( 'Adjusted Close' )
plt.title( 'NVDA Share Price' )
plt.show()



# https://bit.ly/2OSCTdm en este link podemos encontrar estilos para nuestro plot. 
# para plotear data. 


# In[16]:


nvda = yf.download('NVDA', start=start, end=end)
nvda_ohlc = nvda[['Open', 'High', 'Low', 'Close']].resample('10D').ohlc()
nvda_ohlc.reset_index(inplace=True)
nvda_ohlc['Date'] = nvda_ohlc['Date'].map(mdates.date2num)

nvda_volume = nvda['Volume'].resample('10D').sum()

ax1 = plt.subplot2grid(( 6 , 1 ),( 0 , 0 ),
rowspan = 4 , colspan = 1 )
ax2 = plt.subplot2grid(( 6 , 1 ),( 4 , 0 ),
rowspan = 2 , colspan = 1 ,
sharex =ax1)
ax1.xaxis_date()

candlestick_ohlc(ax1, nvda_ohlc.values, width = 5 ,
colorup= 'g' , colordown = 'r' )
ax2.fill_between(nvda_volume.index.map(mdates.date2num),
nvda_volume.values)
plt.tight_layout()
plt.show()



# In[17]:


nvda['100d_ma'] = nvda['Adj Close'].rolling(window=100, min_periods=0).mean()
nvda.dropna(inplace=True)

ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=4, colspan=1)
ax2 = plt.subplot2grid((6, 1), (4, 0), rowspan=2, colspan=1, sharex=ax1)

ax1.plot(nvda.index, nvda['Adj Close'])
ax1.plot(nvda.index, nvda['100d_ma'])

ax2.fill_between(nvda.index, nvda['Volume'], color='b', alpha=0.4)

plt.tight_layout()
plt.show()


# In[18]:


import bs4 as bs
import requests

def load_sp500_tickers():
    link = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(link)
    soup = bs.BeautifulSoup(response.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.strip()
        tickers.append(ticker)
    return tickers


# In[19]:


import pickle

# Asegúrate de que la variable 'tickers' está definida
tickers = load_sp500_tickers()

with open("sp500tickers.pickle", 'wb') as f:
    pickle.dump(tickers, f)


# In[26]:


import os
import datetime as dt
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def load_prices(reload_tickers=False):
    if reload_tickers:
        tickers = load_sp500_tickers()
    else:
        if os.path.exists('sp500tickers.pickle'):
            with open('sp500tickers.pickle', 'rb') as f:
                tickers = pickle.load(f)
    return tickers

tickers = load_prices(reload_tickers=True)

if not os.path.exists('companies'):
    os.makedirs('companies')
    
start = dt.datetime(2023, 1, 1)
end = dt.datetime.now()

for ticker in tickers:
    if not os.path.exists('companies/{}.csv'.format(ticker)):
        df = yf.download(ticker, start=start, end=end)
        df.to_csv('companies/{}.csv'.format(ticker))
    else:
        print('Already have {}'.format(ticker)) 

main_df = pd.DataFrame()

print ("Compiling data...")
for ticker in tickers:
    df = pd.read_csv('companies/{}.csv'.format(ticker))
    df.set_index('Date', inplace=True)
    df.rename(columns={'Adj Close': ticker}, inplace=True)
    df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
    
    if main_df.empty:
        main_df = df
    else:
        main_df = main_df.join(df, how='outer')

main_df.to_csv('sp500_data.csv')
print ("Data compiled and saved to sp500_data.csv")	
load_prices(reload_tickers= True )



# In[31]:


sp500= pd.read_csv('sp500_data.csv')
sp500['AMD'].plot()
plt.show()
sp500.reset_index(inplace=True)

# Convierte la columna 'Date' a datetime si aún no lo es
sp500['Date'] = pd.to_datetime(sp500['Date'])

# Establece 'Date' como el índice
sp500.set_index('Date', inplace=True)

# Ahora puedes calcular la correlación
correlation = sp500.corr()
print(correlation)

plt.matshow(correlation)
plt.show()


# In[36]:


import numpy as np

start = dt.datetime(2023, 1 , 1)
end = dt.datetime.now()
nvda = yf.download('NVDA', start=start, end=end)
data = nvda['Adj Close'] 

x = data.index.map(mdates.date2num)

fit = np.polyfit(x, data.values, 1)
fit1d = np.poly1d(fit)

plt.grid()
plt.plot(data.index, data.values, 'b' )
plt.plot(data.index, fit1d(x), 'r' )
plt.show()


# In[39]:


rstart = dt.datetime(2023, 1, 1)
rend = dt.datetime(2023, 12, 12)

fit_data = data.reset_index()
filtered_data = fit_data[fit_data.Date > rstart]
if not filtered_data.empty:
    pos1 = filtered_data.index[0]
else:
    print("No data after start date")
    pos1 = None

filtered_data = fit_data[fit_data.Date > rend]
if not filtered_data.empty:
    pos2 = filtered_data.index[-1]
else:
    print("No data before end date")
    pos2 = None

if pos1 is not None and pos2 is not None:
    fit_data = fit_data.iloc[pos1:pos2]

    dates = fit_data.Date.map(mdates.date2num)
    fit = np.polyfit(dates, fit_data['Adj Close'].values, 1)
    fit1d = np.poly1d(fit)

    plt.grid()
    plt.plot(data.index, data.values, 'b' )
    plt.plot(fit_data.Date, fit1d(dates), 'r' )
    plt.show()


# In[46]:


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import datetime as dt
import yfinance as yf
import numpy as np

start = dt.datetime(2023, 1 , 1)
end = dt.datetime(2024,1,1)
nvda = yf.download('NVDA', start=start, end=end)
data = nvda[['Adj Close']].copy()   # Convert 'Adj Close' to DataFrame
days = 30
data['Shifted'] = data['Adj Close'].shift(-days)
data.dropna(inplace=True)

X = np.array(data.drop(['Shifted'], axis=1))
Y = np.array(data['Shifted'])
X = preprocessing.scale(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)
print(accuracy)

X = X[:-days]
X_new = X[-days:]
prediction = clf.predict(X_new)
print (prediction)


# In[ ]:




