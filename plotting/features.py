import pandas as pd
import numpy as np
import itertools
import datetime

'''
All functions take a dataframe and return a dataframe with features added
Input should always should have datetime index and datetime index in returned dataframe

'''


def higherTimeFrame(df,time = 'D'):
    grouped = df.groupby('ticker').resample(time).agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last','date': 'last','volume':'sum','count':'sum',
         'value':'sum'}
    ).reset_index()

    return grouped.dropna()


def gap(df):
    '''
    param df: dataframe with datetime index
    return: dataframe with gap column

    ''' 
    data = df.copy()
    daily = higherTimeFrame(data)
    daily = prevclose(daily)
    daily['gap'] = daily['open']/daily['prevdayclose'] - 1
    data = data.reset_index().merge(daily[['date','gap']]).set_index('datetime')
    return data



def ma(df,period =[20],column = ['close'],time= ['1min'],type = ['sma']):
    data = df.copy()

    for t in time:
        for i in period:
            for col in column:
                for x in type:

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

def bollinger(df,n=20,k=2,time = 'D'):
    data = df.copy()
    #only setup to take daily - will likely never use bollinger intraday
    colcheck = f'sma{n}close{time}'

    if colcheck not in data.columns:
        data = ma(df=data,period = [n],time = [time])

    grouped = higherTimeFrame(data, time)[['ticker', 'date', 'close']].dropna()


    grouped.drop(columns=col, inplace=True)
    data = data.reset_index().merge(grouped, on=['date', 'ticker'], how='left').set_index(
        'datetime')


    df['bollmid'] = df[colcheck]
    df['bollup'] = df.groupby('ticker')['bollmid'].transform(lambda x: x.rolling(n).std() * k) + df['bollmid']
    df['bolldown'] =df.groupby('ticker')['bollmid'].transform(lambda x: x.rolling(n).std() * k) - df['bollmid']
    return df



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

def roc(df,period= [10],column ='close'):
    for i in period:
        df['roc'+ column + str(i)] = df.groupby(['date','ticker'])[column].pct_change(i)
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

def addbarcount(df):
    data = df.copy()
    data['barcount'] = 0
    barcountlist = []
    prevcount =0

    for row in data.itertuples():
        if row.match =='opening match':
            barcountlist.append(1)
            prevcount =1

        elif barcountlist[-1]>0:
            barcountlist.append(prevcount +1 )
            prevcount +=1
        else:
            barcountlist.append(0)

    return barcountlist




def addmatches(df):
    #finds first non zero volume bar for each ticker and date
    data = df.copy()
    datanonzero = data[data['volume']!=0]
    datanonzero.reset_index(inplace=True)
    openmatchdf = datanonzero.groupby(['ticker',datanonzero.date])['datetime'].first().reset_index()
    closingmatchdf = datanonzero.groupby(['ticker',datanonzero.date])['datetime'].last().reset_index()
    openmatchdf['match'] = 'opening match'
    closingmatchdf['match'] = 'closing match'
    matchdf = pd.concat([openmatchdf,closingmatchdf])
    matchdf.drop(columns = ['date'],inplace  = True)
    data = pd.merge(data,matchdf,how ='left',left_on=['datetime','ticker'],right_on=['datetime','ticker'])
    #add 10day average volume for opening match
    openingmatchdf = data[data['match']=='opening match']
    openingmatchdf['10dayavgopenvol'] = openingmatchdf.groupby('ticker')['volume'].transform(lambda x: x.rolling(10).mean())
    openingmatchdf.reset_index(inplace=True)
    openingmatchdf = openingmatchdf[['ticker','datetime','10dayavgopenvol']]
    data = pd.merge(data,openingmatchdf,on = ['ticker','datetime'],how = 'left')
    data.set_index('datetime',inplace = True)
    return data


def atr(df):
    tempdf = df.copy()
    if 'daychg' not in tempdf.columns:
        tempdf = prevclose(tempdf)
    grouped = tempdf.groupby('ticker').resample('D').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'prevdayclose': 'last',
         'date': 'last'}).dropna().reset_index(level=0)
    grouped['high_low'] = grouped['high'] - grouped['low']
    grouped['high_close'] = np.abs(grouped['high'] - grouped['prevdayclose'])
    grouped['low_close'] = np.abs(grouped['low'] - grouped['prevdayclose'])
    grouped['true_range'] = grouped[['high_low', 'high_close', 'low_close']].values.max(1)
    grouped['atr'] = grouped.groupby('ticker')['true_range'].transform(lambda x: x.rolling(10).sum() / 10)

    atrdf = grouped[['date', 'ticker', 'atr']]
    tempdf = tempdf.reset_index().merge(atrdf, on=['date', 'ticker'], how='left').set_index('datetime')
    tempdf['atrpct'] = tempdf['atr'] / tempdf['close']
    tempdf['atrrel'] = tempdf['daychg'].abs() / tempdf['atrpct']
    #tempdf.drop(columns=['atrpct'], inplace=True)
    return tempdf

#create vwap feature
def vwap(df):
    df['vwap'] = df.groupby(['ticker',df.index.date])['value'].transform('cumsum')/df.groupby(['ticker',df.index.date])['volume']\
    .transform('cumsum')
    return df
#create average volume at time
def avat(df):
    df['avat5'] = df.groupby(['ticker',df.index.time])['volume'].transform(lambda x: round(x.rolling(5).mean(),0))
    #create cumulative volume at time
    df['cavat5'] = df.groupby(['ticker',df.index.date])['avat5'].transform('cumsum')
    df['rvol5'] = df.groupby(['ticker',df.index.date])['volume'].transform('cumsum')/df['cavat5']
    df['ravat5'] = df['volume']/df['avat5']
    return df

def cumval(df):
    df['cumval'] = df.groupby(['date','ticker'])['value'].transform(lambda x: x.expanding().sum())
    return df

def rvolDelta(df):
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

def sharesout(df):

    def getsharesout():
        data = pd.read_csv('all_sharesout.csv').drop(columns='fset')
        data = data.melt(id_vars='ticker',var_name='date',value_name='sharesout').dropna()
        data['sharesout'] = pd.to_numeric(data['sharesout'])
        data['date'] = pd.to_datetime(data['date'],format='%d/%m/%Y')
        data.set_index('date', inplace=True)
        data = data.groupby('ticker').resample('1D').ffill().droplevel(0)
        return data


