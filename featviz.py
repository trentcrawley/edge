from utils import getTestData
import pandas as pd
import plotly.graph_objs as go
import pandas as pd
from plotly.subplots import make_subplots

def calculate_rsi(data, window=14):
    delta = data.diff()
    up_days = delta.copy()
    up_days[delta<=0]=0.0
    down_days = abs(delta.copy())
    down_days[delta>0]=0.0
    RS_up = up_days.rolling(window).mean()
    RS_down = down_days.rolling(window).mean()
    rsi= 100-100/(1+RS_up/RS_down)
    return rsi

def calculate_adherence_weight(data, window=9,rollingWindow = 9):
    # Calculate EMA
    data['ema'] = data['close'].ewm(span=window).mean()

    # Check for adherence and extra adherence

    # adherenceWeight = (data['close'] < data['ema']).astype(int)
    # extraAdherenceWeight = ((data['high'] > data['ema']) & (data['close'] < data['ema'])).astype(int)

    adherenceWeight = (data['close'] < data['ema']).astype(int) * -1 + (data['close'] >= data['ema']).astype(int)
    extraAdherenceWeight = ((data['high'] > data['ema']) & (data['close'] < data['ema'])).astype(int) * -1 \
                         + ((data['high'] < data['ema']) & (data['close'] >= data['ema'])).astype(int)


    # Combine adherence and extra adherence weights
    totalAdherenceWeight = adherenceWeight + 2 * extraAdherenceWeight
    colname = 'adhereema' + str(window) + str(rollingWindow)
    data[colname] = totalAdherenceWeight.rolling(window=rollingWindow).mean()
    return data

def calculate_range_breakout(data, window, threshold):
    # Calculate rolling high and low
    data['Rolling High'] = data['High'].rolling(window).max()
    data['Rolling Low'] = data['Low'].rolling(window).min()

    # Calculate rolling range
    data['Rolling Range'] = data['Rolling High'] - data['Rolling Low']

    # Check if range is broken
    range_broken = (data['Rolling Range'] > threshold).astype(int)

    # Clean up temporary columns
    data = data.drop(columns=['Rolling High', 'Rolling Low', 'Rolling Range'])

    return range_broken




def plot_data(data,overlays = [],indicator = 'RSI'):
    data['plotdatetime'] = data.index.strftime("%y/%m/%d %H:%M:%S") 
    # Create subplots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    # Candlestick plot
    fig.add_trace(go.Candlestick(x=data['plotdatetime'],
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'], name='Candlestick'), row=1, col=1)

    if overlays:
        for overlay in overlays:
            fig.add_trace(go.Scatter(x=data['plotdatetime'], y=data[overlay], name=f'{overlay}', line=dict(color='purple')), row=1, col=1)



    # Volume bar plot
    fig.add_trace(go.Bar(x=data['plotdatetime'], y=data['volume'], name='volume'), row=2, col=1)

    # indicator line plot
    fig.add_trace(go.Scatter(x=data['plotdatetime'], y=data[indicator], name=f'{indicator}'), row=3, col=1)

    fig.update_layout(height=800, title_text="Stock Data with Indicator")
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.show()



data= getTestData.getTestDataCsv()
# Calculate indicator (RSI)
data = calculate_adherence_weight(data)
plot_data(data[data['ticker']=='FMG'][:800], ['ema'], 'adhereema99')

