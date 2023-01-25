from colmap import columnFeatureMap as col
from utils import connectdb,dbquery,getTestData
import pandas as pd
from prepdata import add_all_features, get_signals
import datetime
from datetime import timedelta


# testSignalInput = [{'sequential': {'10min':[
#     [{'ema':['9min','close']}, {'roc':['1min']}, '<', 0],
#     [{'vwap':[None]}, {'roc':['1min']}, '<', 0],
#     [{'vwap':[None]}, {'value':[None]}, '<', {'ema':['9min','close']}]
# ]}}]

minuteSignalInput = [{'sequential': {'10min':[
    [{'rvol':[None]}, {'value':[None]}, '>', 1.5],
    [{'rvol':[None]}, {'roc':['10min']}, '>', 0.2],
    [{'barcount':[None]}, {'value':[None]}, '>', 10],
    ]}}]

dailySignalInput = [{'sequential': {'daily':[
    [{'bollinger':[None]}, {'bollup':[None]}, '<', 100,],
    [{'gapatr':[None]}, {'value':[None]}, '>', 1.5]
    ]}}]



#df = getTestData.getTestDataCsv(daily = True)


# TODO: need to handle daily data, for daily use asxdailydata database.
# Need to make sure feature functions compaitble with this new format
# Add functions to handle sequential
# make days before handle weekends

#Daily - 
# add atr function
# create function to filter for daily then do minutely.

def get_signal_dates(signals,query ='select * from asxminutedata limit 100000',chunksize=5000000,
    daysbefore = 7,daysafter = 1,params = None):
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
            #add features
            df,signals_new = add_all_features(df,signals)# signals_new adds transform column name
            print('adding features')
            #add signals
            df = get_signals(df,signals_new)
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
        dayDict = {'date': [], 'ticker': [],'plotId':[]}
        plotId = 1
        for row in signals.itertuples():
            startDate = row.date + datetime.timedelta(-daysbefore)
            endDate = row.date + datetime.timedelta(daysafter)
            daysbetween = endDate - startDate
            

            for days in range(daysbetween.days + 1):
                current_date = startDate + datetime.timedelta(days)
                if current_date.weekday() < 5:
                    dayDict['date'].append(current_date)
                    dayDict['ticker'].append(row.ticker)
                    dayDict['plotId'].append(plotId)
        
            plotId+=1
            
        dayids = pd.DataFrame(dayDict)
        plotIds = dayids
        plotIds['primarykey'] = plotIds['date'].astype(str) + plotIds['ticker']

        dayids.drop_duplicates(inplace=True)

        queryDates = list(dayids['date'].astype(str) + dayids['ticker'])

        return queryDates,plotIds
    
    signals = get_all_signals()
    
    if signals is not None:
        queryDates,plotIds = get_unique_ticker_dates(signals)
        numberOfDays = plotIds['plotId'].max()
        print(f'signals found for {numberOfDays} days')
        return queryDates,plotIds
    else:
        print('no signals found')
        return None

def getDates():
    if dailySignalInput:
        # get daily signal dates
        queryDates,plotIds = get_signal_dates(dailySignalInput,query='select * from asxdailydata')#get all unique ticker,date combinations for requested dates 
        # get only dates from queryDates
        query = """
                WITH temp AS (
                    select *, concat(datetime::date,ticker) as datetimeticker from asxminutedata
                )
                SELECT * FROM temp WHERE datetimeticker = any(%(queryDates)s);
                """

        queryDates,plotIds = get_signal_dates(minuteSignalInput,query=query,params = {'queryDates':queryDates})#get all unique ticker,date combinations for requested dates 

    else:
        queryDates,plotIds = get_signal_dates(minuteSignalInput,query='select * from asxdailydata limit 100000')#get all unique ticker,date combinations for requested dates 
    #now use query dates as filter for minute signals.
    
    
    
    #df = dbquery.querySignalDates(queryDates1)#re-query for these days
    # df,new_signals = add_all_features(df,testSignalInput)#re-add features
    # df = get_signals(df,new_signals)#re-add signals
    return plotIds, queryDates

plotIds,queryDates = getDates()

plotIds.to_csv('test.csv')
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







# 


# TODO: need to add ability to say some condition has happened in past x periods.
# Add ability to first filter on daily data, then filter on minute data.
