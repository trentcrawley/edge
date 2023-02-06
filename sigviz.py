from colmap import columnFeatureMap as col
from utils import connectdb,dbquery,getTestData
import pandas as pd
from prepdata import add_all_features, get_signals
import datetime
from datetime import timedelta
from plotting import stockplot
from backtester import Backtester

minuteSignalInput = [{'sequential': {'10min':[
    [{'barcount':[None]}, {'value':[None]}, '<', 20],
    [{'rvol':[None]}, {'value':[None]}, '>', 1.2],
    [{'cumval':[None]}, {'value':[None]}, '>', 200000],
    [{'close':[None]}, {'value':[None]}, '<', {'vwap':[None]}],
    [{'vwap':[None]}, {'roc':['1min']}, '<', 0],
    [{'close':[None]}, {'value':[None]}, '<', {'dayopenlow':[None]}]
    ]}}]


# minuteSignalInput = [{'sequential': {'10min':[
#     [{'rvol':[None]}, {'value':[None]}, '>', 0.8],
#     [{'barcount':[None]}, {'value':[None]}, '<', 15],
#     [{'firstbarrng':[None]}, {'value':[None]}, '<', -0.0015],
#     [{'cumval':[None]}, {'value':[None]}, '>', 300000],
#     ]}}]
#[{'vwap':[None]}, {'roc':['1min']}, '<', 0]

dailySignalInput = [{'sequential': {'daily':[
    [{'open':[None]}, {'value':[None]}, '>',{'bollup':[None]}],
    [{'gapatr':[None]}, {'value':[None]}, '>', 1.5],
    [{'close':[None]}, {'value':[None]}, '>', 0.5],
    [{'atrpct':[None]}, {'value':[None]}, '>', 0.015],
    ]}}]

#[{'atrmultiday':['4']}, {'value':[None]}, '>', 3],
#[{'rvol':[None]}, {'roc':['5min']}, '>', 0.1],
#[{'rvol':[None]}, {'value':[None]}, '>', 0.5],
#df = getTestData.getTestDataCsv(daily = True)


# TODO: 
# Add functions to handle sequential
# make days before handle weekends
# only display first signal
# figure out why signal displaying for some days after real signal

# is small issue where feature function doesn't care acount which days it groups, so if stock halted
# will use data pre halt on batch feature, then on precision database call will not have enough
# data as stock halted immediately prior to signal.

# fix backtester and test
# convert alert to vscode/ ib_insync


def get_signal_dates(signals,query ='select * from asxminutedata order by (ticker,datetime) limit 100000',chunksize=5000000,
    daysbefore = 7,daysafter = 1,daysbeforeplot = 1, daysafterplot = 1,params = None,signalDaily = None):
    '''loops through db and returns a chunk of data at a time. adds features and 
    adds signals, then generates signalsdf. gets all unique ticker date and assigns 
    plotIds''' 
    engine,conn = connectdb.create_db_connection()
    def get_all_signals():
        counter  = 1
        signaldf = pd.DataFrame()

        for df in pd.read_sql(query ,engine , parse_dates= {'datetime':{"format":"%Y-%m-%d %H:%M:S"}},index_col = 'datetime', params = params,chunksize=chunksize):
            print(f'getting chunk {counter}')
            df['date'] = df.index.date
            df['time'] = df.index.time
            df.sort_values(by=['ticker','date'],inplace=True)
            #add features
            #needs days before to calc certain features
            df,signals_new = add_all_features(df,signals)# signals_new adds transform column name
            # if 'LDR' in df['ticker'].values:
            #     df[df['ticker']=="LDR"].to_csv('LDRtestfeat.csv')
            print('adding features')
            #add signals
            # need to filter the data to only include daily signals if daily specified
            if signalDaily:
                df['primarykey'] = df['date'].astype(str) + df['ticker']
                df = df[df['primarykey'].isin(signalDaily)]

            if not df.empty:
                df = get_signals(df,signals_new)
                # if 'LDR' in df['ticker'].values:
                #     if df[df['ticker']=="LDR"]['signal'].sum()>0:
                #         df[df['ticker']=="LDR"].to_csv('LDRtest.csv')

                print('adding signals')
                if df.signal.sum() > 0:
                    signaldf = pd.concat([signaldf,df[df['signal']==1]],axis=0,ignore_index=True)# concat signals
                counter+=1

        engine.dispose()
        conn.close()
        
        if not signaldf.empty:
            output = signaldf.groupby(['date','ticker']).last().reset_index()
            return output
        else:
            return None
        

    def get_unique_ticker_dates(signals):
        dayDict = {'date': [], 'ticker': []}
        
        for row in signals.itertuples():
            startDate = row.date + datetime.timedelta(-daysbefore)
            endDate = row.date + datetime.timedelta(daysafter)
            daysbetween = endDate - startDate
            
            for days in range(daysbetween.days + 1):
                current_date = startDate + datetime.timedelta(days)
                if current_date.weekday() < 5:
                    dayDict['date'].append(current_date)
                    dayDict['ticker'].append(row.ticker)
            
        dayids = pd.DataFrame(dayDict)
        dayids.drop_duplicates(inplace=True)

        plotDict = {'date': [], 'ticker': [],'plotId':[]}
        plotId = 1
        for row in signals.itertuples():
            startDate = row.date + datetime.timedelta(-daysbeforeplot)
            endDate = row.date + datetime.timedelta(daysafterplot)
            daysbetween = endDate - startDate

            for days in range(daysbetween.days + 1):
                current_date = startDate + datetime.timedelta(days)
                if current_date.weekday() < 5:
                    plotDict['date'].append(current_date)
                    plotDict['ticker'].append(row.ticker)
                    plotDict['plotId'].append(plotId)
            
            plotId+=1

        plotIds = pd.DataFrame(plotDict)
        plotIds['primarykey'] = plotIds['date'].astype(str) + plotIds['ticker']
        plotIds.drop_duplicates(inplace=True)

        queryDates = list(dayids['date'].astype(str) + dayids['ticker']) #sigdate + dates either side
        signalDates = list(signals['date'].astype(str) + signals['ticker']) # just sig dates same format as queryDates

        return queryDates,plotIds,signalDates
    
    signals = get_all_signals()
    
    if signals is not None:
        queryDates,plotIds,signalDates = get_unique_ticker_dates(signals)
        numberOfDays = plotIds['plotId'].max()
        print(f'signals found for {numberOfDays} days')
        return queryDates,plotIds,signalDates
    else:
        print('no signals found')
        return None,None,None

def getDates():
    if dailySignalInput:
        # get daily signal dates
        print('filtering by daily signals')
        queryDates,plotIds,signalDates = get_signal_dates(dailySignalInput,query='select * from asxdailydata order by (ticker,datetime)')#get all unique ticker,date combinations for requested dates 
        # get only dates from queryDates
        query = """
                WITH temp AS (
                    select *, concat(datetime::date,ticker) as datetimeticker from asxminutedata order by (ticker,datetime) 
                )
                SELECT * FROM temp WHERE datetimeticker = any(%(queryDates)s) ;
                """
        print('filtering by minute signals')
        queryDates,plotIds,signalDates = get_signal_dates(minuteSignalInput,query=query,params = {'queryDates':queryDates},signalDaily = signalDates)#get all unique ticker,date combinations for requested dates 

    else:
        queryDates,plotIds,signalDates = get_signal_dates(minuteSignalInput,query='select * from asxminutedata limit 100000 order by (ticker,datetime)')#get all unique ticker,date combinations for requested dates 
    #now use query dates as filter for minute signals.
    
    
    
    df = dbquery.querySignalDates(queryDates)#re-query for these days
    df,new_signals = add_all_features(df,minuteSignalInput)#re-add features, need new signals because orig has been updated in first loop
    df = get_signals(df,new_signals)#re-add signals
    return df, plotIds, queryDates

df, plotIds,queryDates = getDates()

plot = stockplot.StockPlot(df,plotIds = plotIds)
plot.sigviz(overlays = ['vwap'])

# print("done")


# btest = Backtester(df)

# print('yes')
#plotIds.to_csv('test.csv')



# import importlib
# importlib.reload(features)
# engine,conn = connectdb.create_db_connection()
# params = queryDates
# chunksize = 1000000
# query = """
#         WITH temp AS (
#             select *, concat(datetime::date,ticker) as datetimeticker from asxminutedata
#         )
#         SELECT * FROM temp WHERE datetimeticker = any(%(queryDates)s) limit 1000000;
#         """


# for df in pd.read_sql(query ,engine , parse_dates= {'datetime':{"format":"%Y-%m-%d %H:%M:S"}},index_col = 'datetime', params = {'queryDates':queryDates},chunksize=chunksize):
#     print(df.head())

# conn=connectdb.create_db_connection()


# query = """
# WITH temp AS (
#     select *, concat(datetime::date,ticker) as datetimeticker from asxminutedata
# )
# SELECT * FROM temp WHERE datetimeticker = any(%(queryDates)s);
# """

# for chunk in pd.read_sql(query, engine, params={'queryDates':queryDates}, chunksize=10):
#     print(chunk)


