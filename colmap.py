from plotting.features import *

#map column names to functions
columnFeatureMap = {
'9emaclose1min':[ma,{'period':[9],'type':['ema']}],
'vwap':[vwap,None],
'rocclose1':[roc,{'period':[1],'column':'close'}],
'rocvwap1':[roc,{'period':[1],'column':'vwap'}],
'rocema9close1min1':[roc,{'period':[1],'column':'ema9close1min'}],
'rvol5':[rvol,None]   
}

