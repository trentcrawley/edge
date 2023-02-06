signal = [['vwap','rocvwap1','<',0],['9emaclose1min','rocema9close1min1','<',0]]




daysbefore = 1
daysafter = 1

minuteSignalInput = [{'sequential': {'10min':[
    [{'rvol':[None]}, {'value':[None]}, '>', 0.5],
    [{'barcount':[None]}, {'value':[None]}, '<', 20],
    [{'cumval':[None]}, {'value':[None]}, '>', 200000],
    [{'close':[None]}, {'value':[None]}, '<', {'vwap':[None]}],
    [{'close':[None]}, {'value':[None]}, '<', {'dayopenlow':[None]}]
    ]}}]

'''
None if column name and no feature needs to be added or is function and no arguments
2nd argument is transformation of first, should be value with None value if no transformation
3rd argument is operator
4th argument is value to compare to. can be value or feature name - feature cannot be transformed

'''