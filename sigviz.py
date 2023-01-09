from colmap import columnFeatureMap as col
from utils import connectdb,dbquery
import pandas as pd
from testing import add_required_features, get_signals
import datetime
from datetime import timedelta


# input [feature, transform, parameters, operator, value] as list
# testsignal = [
#     [col['vwap'],col['rocvwap1'],'<',0],
#     [col['9emaclose1min'],col['rocema9close1min1'],'<',col['rocvwap1']]
#     ]

testsignal = [
    ['vwap','rocvwap1','<',0],
    ['9emaclose1min','rocema9close1min1','<',0]
    ]

#need to handle daily data, for daily use asxdailydata database.

def get_signal_dates(signals,query ='select * from asxminutedata limit 100000',chunksize=5000000,
    conn=connectdb.create_db_connection(),daysbefore = 1,daysafter = 1):
    '''loops through db and returns a chunk of data at a time''' 
    
    def get_all_signals():
        signaldf = pd.DataFrame()
        for df in pd.read_sql(query ,conn , parse_dates= {'datetime':{"format":"%Y-%m-%d %H:%M:S"}},index_col = 'datetime', chunksize=chunksize):
            df['date'] = df.index.date
            df['time'] = df.index.time

            #add features
            df = add_required_features(df,signals,col)
            #get signals
            df = get_signals(df,signals)

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
        return queryDates
    else:
        print('no signals found')
        return None

def getDates():
    queryDates,plotIds = get_signal_dates(testsignal)#get all unique ticker,date combinations for requested dates 
    df = dbquery.querySignalDates(queryDates)#re-query for these days

    df = add_required_features(df,testsignal,col)#re-add features
    df = get_signals(df,testsignal)#re-add signals
    return df, plotIds

#plot and save to folder



#pass queryDates to stockplot
