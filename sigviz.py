from colmap import columnFeatureMap as col
from utils import connectdb,dbquery
import pandas as pd
from prepdata import add_all_features, get_signals
import datetime
from datetime import timedelta


# testSignalInput = [{'sequential': {'10min':[
#     [{'ema':['9min','close']}, {'roc':['1min']}, '<', 0],
#     [{'vwap':[None]}, {'roc':['1min']}, '<', 0],
#     [{'vwap':[None]}, {'value':[None]}, '<', {'ema':['9min','close']}]
# ]}}]

testSignalInput = [{'sequential': {'10min':[
    [{'rvol':[None]}, {'value':[None]}, '>', 1.5],
    [{'rvol':[None]}, {'roc':['10min']}, '>', 0.2],
    [{'barcount':[None]}, {'value':[None]}, '>', 10],
    [{'emadaily':['9day','close']}, {'value':[None]}, '<', {'emadaily':['20day','close']},]
    ]}}]

# TODO: need to handle daily data, for daily use asxdailydata database.
# Need to make sure feature functions compaitble with this new format
# Add functions to handle sequential
# make days before handle weekends

#Daily - maintain list of daily function names. if any of these are in the signal then use asxdailydata first

def get_signal_dates(signals,query ='select * from asxminutedata limit 100000',chunksize=5000000,
    conn=connectdb.create_db_connection(),daysbefore = 1,daysafter = 1):
    '''loops through db and returns a chunk of data at a time. adds features and 
    adds signals, then generates signalsdf. gets all unique ticker date and assigns 
    plotIds''' 
    
    def get_all_signals():
        counter  = 1
        signaldf = pd.DataFrame()
        for df in pd.read_sql(query ,conn , parse_dates= {'datetime':{"format":"%Y-%m-%d %H:%M:S"}},index_col = 'datetime', chunksize=chunksize):
            df['date'] = df.index.date
            df['time'] = df.index.time
            #add features
            df,signals_new = add_all_features(df,signals)# signals_new adds transform column name
            #add signals
            df = get_signals(df,signals_new)
            if df.signal.sum() > 0:
                signaldf = pd.concat([signaldf,df[df['signal']==1]],axis=0,ignore_index=True)
            
            conn.close()
            if not signaldf.empty:
                output = signaldf.groupby(['date','ticker']).last().reset_index()
            else:
                return None
            return output

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
        queryDates = get_unique_ticker_dates(signals)
        print(f'signals found for {len(queryDates[0])} days')
        return queryDates
    else:
        print('no signals found')
        return None

def getDates():
    queryDates,plotIds = get_signal_dates(testSignalInput)#get all unique ticker,date combinations for requested dates 
    df = dbquery.querySignalDates(queryDates)#re-query for these days
    df,new_signals = add_all_features(df,testSignalInput)#re-add features
    df = get_signals(df,new_signals)#re-add signals
    return df, plotIds

df,plotIds = getDates()

print('done')
df.to_csv('test.csv')


# TODO: need to add ability to say some condition has happened in past x periods.
# Add ability to first filter on daily data, then filter on minute data.
