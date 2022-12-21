import pandas as pd

def getTestDataCsv():
    '''grab the test data from the csv file and format it'''

    data = pd.read_csv(r"C:\Users\trent\VSCode\edge\data\testData.csv")
    data['datetime'] = pd.to_datetime(data['datetime'], format="%Y-%m-%d %H:%M:%S")
    data.set_index('datetime', inplace=True)
    data['date'] = data.index.date
    data['time'] = data.index.time
    data.sort_values(by=['ticker', 'datetime'], inplace=True)
    
    return data

