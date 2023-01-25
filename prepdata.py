from utils.dbquery import query_db
import pandas as pd
from plotting.features import *

def add_all_features(df,signals):
    """ e.g.
    testSignal = [{'sequential': {'10min':[
    [{'ema':['10Min','close']}, {'roc':['1min']}, '<', 0],
    [{'vwap':[None]}, {'roc':['1min']}, '<', 0],
    [{'vwap':[None]}, {'value':[None]}, '<', {'ema':['9Min','close']}]
    ]}}]
    
    checks if need to add transform and ands it to params
    then calls all feature functions with params
    returns df with all features added

    """
    
    def call_feature_function(feature,df):
        '''
        takes dict of feature and params and calls function with parameters
        '''
        for k,v in feature.items():
            feat_func = k
            #print(v) #parameters for function as a list
            if v[0] != None: # if has params pass into function
                feat_params = v
                df = globals()[feat_func](df,*feat_params)
            else:
                df = globals()[feat_func](df)

        return df
    
    def get_new_col_name(i):
        '''
        gets the name of transformed feature column r.g. ema10minclose
        '''
        func_name = list(i[0].keys())[0]
        if i[0][func_name][0] != None:
            param_names = ''.join(list(i[0].values())[0])
            new_col_name = func_name + param_names
        else:
            new_col_name = func_name
        return new_col_name
    
    for signal in signals: # grouped by type
        for type, v in signal.items(): # for each type i.e. sequential, parallel
            for period,v in signal[type].items(): # if sequential time period for sequence
                for i in v: #each discrete signal list within period within type
                    if list(i[1].keys())[0] != 'value': #check if we need to add col name to transform func
                        newcol = get_new_col_name(i)
                        if newcol != None:
                            if newcol not in list(i[1].values())[0]: #if new col name not in transform function (e.g. roc1min
                                transform = list(i[1].keys())[0]
                                i[1][transform].append(newcol) #appends new col name to transform function

                    # go through elements which potential have features
                    if list(i[1].keys())[0] != 'value': # if its not value add feature and transform
                        if list(i[0].keys())[0] not in df.columns:
                            df = call_feature_function(i[0],df) #feature
                        if list(i[1].keys())[0] not in df.columns:
                            df = call_feature_function(i[1],df)#transform
                    else:
                        if list(i[0].keys())[0] not in df.columns: #else add feature
                            df = call_feature_function(i[0],df)

                    if isinstance(i[3],dict): #if comparator is feature add
                        if list(i[3].keys())[0] not in df.columns: 
                            df = call_feature_function(i[3],df)  
    
    return df, signals


##### testing
# TODO: add signal function to check for sequential or non-squential signals
                
#from utils.getTestData import getTestDataCsv
#df = getTestDataCsv()
#df,signals = add_all_features(df[:50000],testSignal)



def get_signals(df, signals):

    df['signal'] = 1

    def get_col_name(i):
        '''
        gets the returned column name from a feature function
        '''
        func_name = list(i.keys())[0]
        if i[func_name][0] != None:
            param_names = ''.join(list(i.values())[0])
            col_name = func_name + param_names
        else:
            col_name = func_name
        return col_name

    for signal in signals:
        for type,v in signal.items():
            for period,v in signal[type].items():
                for signal in signal[type][period]:

                    value_col = ''
                    cond_col = ''
                    
                    #get value
                    if list(signal[1].keys())[0] == 'value': # if it is value then use the feature column
                        value_col = get_col_name(signal[0])
                        value = df[value_col]
                    else:
                        value_col = get_col_name(signal[1])
                        value = df[value_col]
                    
                    # get condition
                    if isinstance(signal[3],(int,float)):#check if we are comparing to a scalar or another feature
                        cond = signal[3] #is value
                    else:
                        cond_col = get_col_name(signal[3])
                        cond = df[cond_col] # is column name                    

                    if cond_col=='':
                        condstring = str(signal[3])
                    else:
                        condstring = cond_col

                    #print(f'getting signal: {value_col} {str(signal[2])} {condstring}')

                    if signal[2] == '<':
                        df['signal'] = np.where(value < cond,1,0) * df['signal']
                    elif signal[2] == '>':
                        df['signal'] = np.where(value > cond,1,0) * df['signal']
                    elif signal[2] == '<=':
                        df['signal'] = np.where(value <= cond,1,0) * df['signal']
                    elif signal[2] == '>=':
                        df['signal'] = np.where(value >= cond,1,0) * df['signal']
                    elif signal[2] == '==':
                        df['signal'] = np.where(value == cond,1,0) * df['signal']
                    elif signal[2] == '!=':
                        df['signal'] = np.where(value != cond,1,0) * df['signal']

    return df


#df = get_signals(df,signals)
#df.to_csv('test.csv')

# df = add_required_features(df,testsignal,columnFeatureMap)
# print(df.columns)
# df = get_signals(df,testsignal)
# print(df)
# df.to_csv('test.csv')



