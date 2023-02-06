import pandas as pd
import numpy as np
import itertools
import datetime
from utils.getTestData import getTestDataCsv

df = getTestDataCsv(daily=True)

'''
All functions take a dataframe and return a dataframe with features added
Input should always should have datetime index and datetime index in returned dataframe
function name should be same as column name, if has args should colname==funcname+args in order

'''


def higherTimeFrame(df,time = 'D'):
    grouped = df.groupby('ticker').resample(time).agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last','date': 'last','volume':'sum','count':'sum',
         'value':'sum'}
    ).reset_index()

    return grouped.dropna()

def firstbarrng(df):
    if 'barcount' not in df.columns:
        df = barcount(df)
    df['firstbarrng'] = (df[df['barcount'] == 0]['close'] / df[df['barcount'] == 0]['open']) - 1
    return df

# write a function to group dataframe by ticker and date then calculate the exponential moving average
def ema(df,period = '20min',column = 'close'):
    '''
    period: period of ema, include min for minutes and D for days
    if D need to use daily data
    '''
    df = df.copy()
    colname = 'ema' + str(period) + column
    if period[-3:].upper() == 'MIN':
        period = int(period[:-3])
    df.loc[:,colname] = df.groupby(['ticker',df.index.date])[column].transform(lambda x: x.ewm(span=period,min_periods=period).mean())
    return df


def gap(df):
    '''
    param df: dataframe with datetime index
    return: dataframe with gap column

    ''' 
    df = df.copy()
    df['prevdayclose'] = df.groupby('ticker')['close'].shift(1)
    df['gap'] = df['open']/df['prevdayclose'] - 1
    df['gapdollars'] = df['open'] - df['prevdayclose']
    df.drop(columns = 'prevdayclose',inplace=True)
    return df

def gapatr(df):
    df = df.copy()
    if 'gap' not in df.columns:
        df = gap(df)
    if 'atr' not in df.columns:
        df = atr(df)
        
    df['gapatr'] = df['gapdollars']/df['atr']

    return df


def ma(df,period =[20],column = ['close'],time= ['1min'],type = ['sma']):
    data = df.copy()

    for t in time:
        for i in period:
            for col in column:
                for x in type:
                    # naming convention sma20close1min
                    # typeperiodcolumninterval
                    newcol  = x + str(i) + col + t

                    if t != '1min' and t !='D':
                        grouped = higherTimeFrame(df = data,time = t)[['ticker','datetime',col]].dropna()
                        if x =='sma':
                            grouped[newcol] = grouped.groupby(['ticker',grouped['datetime'].dt.date])[col].transform(lambda x: x.rolling(i).mean())
                        elif x == 'ema':
                            grouped[newcol] = grouped.groupby(['ticker',grouped['datetime'].dt.date])[col].transform(lambda x: x.ewm(span=i,min_periods=i).mean())

                        grouped.drop(columns = col,inplace=True)
                        data = data.reset_index().merge(grouped, on=['datetime', 'ticker'], how='left').set_index('datetime')
                        data[newcol] = data.groupby(['ticker',data.index.date])[newcol].ffill()

                    elif t == 'D':
                        grouped = higherTimeFrame(data, t)[['ticker', 'date', col]].dropna()
                        if x =='sma':
                            grouped[newcol] = grouped.groupby(['ticker'])[col].transform(lambda x: x.rolling(i).mean())
                        elif x =='ema':
                            grouped[newcol] = grouped.groupby(['ticker'])[col].transform(
                            lambda x: x.ewm(span=i, min_periods=i).mean())

                        grouped.drop(columns=col, inplace=True)
                        data = data.reset_index().merge(grouped, on=['date', 'ticker'], how='left').set_index(
                            'datetime')

                    else:
                        if x =='sma':

                            data[newcol] = data.groupby(['ticker',data.index.date])[col].transform(
                                lambda x: x.rolling(i).mean())

                        elif x =='ema':
                            data[newcol] = data.groupby(['ticker',data.index.date])[col].transform(
                                lambda x: x.ewm(span=i, min_periods=i).mean())


    return data


def bollinger(df,n=20,k=2):
    df['bollmid'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(n).mean())
    df['bollup'] = df.groupby('ticker')['bollmid'].transform(lambda x: x.rolling(n).std() * k) + df['bollmid']
    df['bolldown'] =df.groupby('ticker')['bollmid'].transform(lambda x: x.rolling(n).std() * k)*-1 + df['bollmid']
    return df

def bollup(df,n=20,k=2):
    if 'bollmid' not in df.columns:
        df['bollmid'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(n).mean())
    df['bollup'] = df.groupby('ticker')['bollmid'].transform(lambda x: x.rolling(n).std() * k) + df['bollmid']
    return df

def bolldown(df,n=20,k=2):
    if 'bollmid' not in df.columns:
        df['bollmid'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(n).mean())
    df['bolldown'] =df.groupby('ticker')['bollmid'].transform(lambda x: x.rolling(n).std() * k)*-1 + df['bollmid']
    return df

def turnover():
    data = getsharesout()
    df['date'] = pd.to_datetime(df['date'])
    df = df.merge(data.reset_index(), on=['date', 'ticker'], how='inner')
    df['cumvol'] = df.groupby(['date', 'ticker'])['volume'].transform(lambda x: x.expanding().sum())

    df['registerturn'] = df['cumvol']/(df['sharesout']*1000000)

    return df

def pctofflevel(df):

    df['daylow'] = df.groupby(['ticker', 'date'])['low'].expanding().min().reset_index().set_index('datetime')['low']
    df['pctofflow'] = df['close']/df['low'] - 1
    return df

def pctbelowvwap(df):
    df['belowvwap'] = np.where(df['close']<=df['vwap'],1,0)
    def getpct(x):
        pct = x.expanding().mean()
        return pct
    df['pctbelowvwap'] = df.groupby(['ticker', 'date'])['belowvwap'].transform(getpct)

    return df

def touchMa(df,ma = ['ema20close1min'],fromAbove = False):
    for line in ma:

        if fromAbove == False:
            df['touch' + line+'below'] = np.where((df['high']>= df[line])
                                          & (df['close'] < df[line]),1,0)
        else:
            df['touch' + line+'above'] = np.where((df['low']<= df[line])
                                          & (df['close'] > df[line]),1,0)

    return df

def maResponse(df):
    df['priorminhigh'] = df.groupby(['ticker','date'])['high'].shift(1)
    df['priorminhigh2'] = df.groupby(['ticker', 'date'])['high'].shift(2)
    df['priorminhigh3'] = df.groupby(['ticker', 'date'])['high'].shift(3)

    df['touchema20close1minbelowshift1'] = df.groupby(['ticker','date'])['touchema20close1minbelow'].shift(1)
    df['touchema20close1minbelowshift2'] = df.groupby(['ticker', 'date'])['touchema20close1minbelow'].shift(2)
    df['touchema20close1minbelowshift3'] = df.groupby(['ticker', 'date'])['touchema20close1minbelow'].shift(3)



    df['maresponse1min'] = df.apply(lambda x: x['close']/x['priorminhigh']-1 if x['touchema20close1minbelowshift1']==1 else 0,axis =1)
    df['maresponse2min'] = df.apply(lambda x: x['close']/x['priorminhigh2']-1 if x['touchema20close1minbelowshift2']==1 else 0,axis =1)
    df['maresponse3min'] = df.apply(lambda x: x['close']/x['priorminhigh3']-1 if x['touchema20close1minbelowshift3']==1 else 0,axis =1)

    df.drop(columns = ['priorminhigh','priorminhigh2','priorminhigh3','touchema20close1minbelowshift1',
                       'touchema20close1minbelowshift2','touchema20close1minbelowshift3'],inplace = True,axis = 1)

    return df

def pctbelowma(df,ma = ['ema20close1min'],period = 15):
    for line in ma:
        df['below' + line] = np.where(df['close']<=df[line],1,0)
        def getpct(x):
            pct = x.rolling(period).mean()
            return pct
        df['pctbelow'+line] = df.groupby(['ticker', 'date'])['below' + line].transform(getpct)

        return df

def spreadroc(df, spread = ['ANZ','WBC'],period= [10]):
    data = df.copy()
    for i in period:
        data['spreadroc'+str(i)] = data.groupby('date')[spread[0]+spread[1] +'close'].pct_change(i)
    return data

def roc(df,period= '10min',column ='close'):
    colname = 'roc' + str(period) + column
    period = int(period[:-3])
    df.loc[:, colname] = df.groupby(['date','ticker'])[column].pct_change(period)
    return df

def tyronestdev(df,period = 30):
    df['tystdevclose' + str(period)] = df.groupby(['date','ticker'])['close'].transform(lambda x: x.rolling(period).std())
    df['tystdevlow' + str(period)] = df.groupby(['date','ticker'])['low'].transform(lambda x: x.rolling(period).std())
    df['tystdevhigh' + str(period)] = df.groupby(['date','ticker'])['high'].transform(lambda x: x.rolling(period).std())
    df['tystdev'] = df['tystdevhigh' + str(period)] + df['tystdevlow' + str(period)] + df['tystdevclose' + str(period)]
    df = df.drop(columns = {'tystdevhigh' + str(period),'tystdevlow' + str(period),'tystdevclose' + str(period)})
    return df

def barcount(df):
    df['barcount'] = df.groupby(['ticker','date']).cumcount()
    return df

def atr(df):
    df = df.copy()
    df['prevdayclose'] = df.groupby('ticker')['close'].shift(1)
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = np.abs(df['high'] - df['prevdayclose'])
    df['low_close'] = np.abs(df['low'] - df['prevdayclose'])
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].values.max(1)
    df['atr'] = df.groupby('ticker')['true_range'].transform(lambda x: x.rolling(10).sum() / 10)
    df.drop(columns = ['prevdayclose','high_low','high_close','low_close','true_range'],inplace = True)
    return df

def atrmultiday(df, days = 4):
    '''
    atr mutliple from look back period
    '''
    df = df.copy()
    days = int(days)
    df['refclose'] = df.groupby('ticker')['close'].shift(days)
    df['multidaymove'] = df['open'] - df['refclose']
    df['atrmultiday'+str(days)] = df['multidaymove']/df['atr']
    df.drop(columns = ['refclose','multidaymove'],inplace = True)
    return df

def atrpct(df):
    '''
    atr as a percentage of close
    '''
    if 'atr' not in df.columns:
        df = atr(df)
    df['atrpct'] = df['atr'].abs()/df['close']
    return df

#create vwap feature
def vwap(df):
    df.loc[:,'vwap'] = df.groupby(['ticker',df.index.date])['value'].transform('cumsum')/df.groupby(['ticker',df.index.date])['volume']\
    .transform('cumsum')
    return df

#create average volume at time
def rvol(df):
    '''
    rvol for 5 days only, comes with averagevolattime and cumulative average volume at 
    time for convience
    '''
    df['avat'] = df.groupby(['ticker',df.index.time])['volume'].transform(lambda x: round(x.rolling(5).mean(),0))
    #create cumulative volume at time
    df['cavat'] = df.groupby(['ticker',df.index.date])['avat'].transform('cumsum')
    df['rvol'] = df.groupby(['ticker',df.index.date])['volume'].transform('cumsum')/df['cavat']
    df['ravat'] = df['volume']/df['avat']
    return df

def cumval(df):
    df['cumval'] = df.groupby(['date','ticker'])['value'].transform(lambda x: x.expanding().sum())
    return df

def rvoldelta(df):
    df['priorRvol'] =df.groupby(['date','ticker'])['rvol5'].shift(15)
    df['rvolDelta'] = df['rvol5']/df['priorRvol']
    df.drop(columns=['priorRvol'],inplace=True)
    return df

def pctbelowvwap(df):
    df['belowvwap'] = np.where(df['close']<=df['vwap'],1,0)
    def getpct(x):
        pct = x.expanding().mean()
        return pct
    df['pctbelowvwap'] = df.groupby(['ticker', 'date'])['belowvwap'].transform(getpct)
    return df

def vwapstretch(df):                
    df['vwapstr'] = df['close']/df['vwap'] - 1
    return df

def distance(df,col1,col2):
    df['distance'+ col1 + col2] = df[col1]/df[col2] - 1
    return df

def wap(df):
    df['wap'] = df[['close','volume','value']].apply(lambda x: x['close'] if x['volume'] == 0 else round(x['value']/x['volume'],4),axis =1)
    return df

def g_h_filter(data, x0=2, dx=1, g=6./10, h=.1, dt=1.):
    x = x0
    results = []
    for z in data:
        #prediction step
        x_est = x + (dx*dt)
        dx = dx

        # update step
        residual = z - x_est
        dx = dx    + h * (residual) / dt
        x  = x_est + g * residual
        results.append(x)
    return np.array(results)

def kalman(df):
    df['kalman'] = df.groupby('ticker')['close'].transform(g_h_filter,x0=df.close[0], dx=1, g=.5, h=.5, dt=1.)
    return df

def zeds(df):
    def zscore(x):
        m = x.expanding().mean()
        s = x.expanding().std()
        z = (x-m)/s
        return z
    df['zed'] = df.groupby(['ticker','date'])['close'].transform(zscore)
    return df

#function to  calculate number of days between consecutive dates in pandas dataframe
def days_between(df):
    return df.groupby('ticker')['date'].transform(lambda x: x.diff().dt.days)  


def prevclose(df):
    tempdf = df.groupby(['ticker','date'])['close'].last().groupby('ticker').shift().reset_index()
    tempdf.rename(columns = {'close':'prevdayclose'},inplace = True)
    df.reset_index(inplace=True)
    df = df.merge(tempdf, on=['ticker', 'date']).set_index('datetime')
    df['daychg'] = df['close']/df['prevdayclose'] - 1
    return df

def nextxdayclose(df,days =1):
    tempdf = df.reset_index().groupby(['ticker','date'])['datetime'].last().groupby('ticker').shift(-days).reset_index()
    if days>0:
        colname = 'tplus' + str(days)
    else:
        colname =  'tminus' + str(abs(days))
    tempdf.rename(columns = {'datetime':colname},inplace = True)
    df.reset_index(inplace=True)
    df = df.merge(tempdf, on=['ticker', 'date']).set_index('datetime')
    df[colname] = pd.to_datetime(df[colname])
    df.sort_values(by = ['ticker','datetime'],inplace=True)
    df[colname].fillna(method='ffill',inplace= True)
    return df


def stdev(df,column = 'close',i=20):
    data = df.copy()
    data['stdev' + str(i) + column] = data.groupby(['date', 'ticker'])[column].transform(lambda x: x.rolling(i).std())
    data['bol' + str(i) + column+'up'] = data['stdev'+str(i) + column] + data.groupby(['date', 'ticker'])[column].transform(lambda x: x.rolling(i).mean())
    data['bol' + str(i) + column + 'down'] = data.groupby(['date', 'ticker'])[column].transform(lambda x: x.rolling(i).mean()) - data['stdev'+str(i) + column]
    return data

def spread(df,ticker:list,vals = ['close','volume']):
    spreaddf = df.pivot_table(index = 'datetime',columns='ticker',values=vals)
    cross = itertools.combinations(ticker,2)

    for i in cross:
        spreaddf[('close',str(i[0]) + str(i[1]))] = spreaddf[('close',i[0])]/spreaddf[('close',i[1])]

    namelist = [j + i for i, j in spreaddf.columns]
    spreaddf = spreaddf.droplevel(0, axis=1)
    spreaddf.columns = namelist
    if spreaddf.index.dtype == '<M8[ns]':
        spreaddf['date'] = spreaddf.index.date
        spreaddf['time'] = spreaddf.index.time


    return spreaddf

def dayopenopen(df):
    df['dayopenopen'] = df.groupby(['ticker','date'])['open'].transform('first')
    return df

def dayopenlow(df):
    '''
    low of first bar
    '''
    df['dayopenlow'] = df.groupby(['ticker','date'])['low'].transform('first')
    return df


def sharesout(df):

    def getsharesout():
        data = pd.read_csv('all_sharesout.csv').drop(columns='fset')
        data = data.melt(id_vars='ticker',var_name='date',value_name='sharesout').dropna()
        data['sharesout'] = pd.to_numeric(data['sharesout'])
        data['date'] = pd.to_datetime(data['date'],format='%d/%m/%Y')
        data.set_index('date', inplace=True)
        data = data.groupby('ticker').resample('1D').ffill().droplevel(0)
        return data


