from utils.dbquery import query_db
import pandas as pd
from plotting.features import *


df = query_db(['FMG','BHP'],datefrom='2020-01-01',dateto='2022-01-02')

#need to map inputs to 
columnFeatureMap = {
'9emaclose1min':[ma,{'period':[9],'type':['ema']}],
'vwap':[vwap,None],
'rocclose1':[roc,{'period':[1],'column':'close'}],
'rocvwap1':[roc,{'period':[1],'column':'vwap'}],
'rocema9close1min1':[roc,{'period':[1],'column':'ema9close1min'}],   
}

testsignal = [['vwap','rocvwap1','<',0],['9emaclose1min','rocema9close1min1','<',0]]

def add_required_features(df,signals,functionsMap):
    # add all required features to df
    for signal in signals:
        for i in [0,1,3]: #functions can only be in elements 0,1,3
            if signal[i] not in df.columns:
                if signal[i] in functionsMap:
                    if functionsMap[signal[i]][1] != None: #if there are parameters
                        df = functionsMap[signal[i]][0](df,**functionsMap[signal[i]][1])
                    else:
                        df = functionsMap[signal[i]][0](df)
                elif signal[i] != 0:
                    print(f'{signal[i]} not in functions map')
    return df


def get_signals(df,signals):

    df['signal'] = 1

    for signal in signals:

        #check if we are comparing to a scalar or another feature
        if isinstance(signal[3],int):
            cond = signal[3] #is value
        else:
            cond = df[signal[1]] # is column name
        
        if signal[2] == '<':
            df['signal'] = np.where(df[signal[1]] < cond,1,0) * df['signal']
        elif signal[2] == '>':
            df['signal'] = np.where(df[signal[1]] > cond,1,0) * df['signal']
        elif signal[2] == '<=':
            df['signal'] = np.where(df[signal[1]] <= cond,1,0) * df['signal']
        elif signal[2] == '>=':
            df['signal'] = np.where(df[signal[1]] >= cond,1,0) * df['signal']
        elif signal[2] == '==':
            df['signal'] = np.where(df[signal[1]] == cond,1,0) * df['signal']
        elif signal[2] == '!=':
            df['signal'] = np.where(df[signal[1]] != cond,1,0) * df['signal']

    return df

df = add_required_features(df,testsignal,columnFeatureMap)
print(df.columns)
df = get_signals(df,testsignal)
print(df)
df.to_csv('test.csv')