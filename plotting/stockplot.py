import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import os
from utils import getTestData
import math
import sigviz

df,plotIds =  sigviz.getDates()
         

def make_multiplots():
    fig = make_subplots(
    rows=4,
    cols=3,
    shared_xaxes="columns",
    shared_yaxes="rows",
    column_width=[0.3,0.3,0.3],
    row_heights=[0.2,0.06,0.2,0.06],
    horizontal_spacing=0,
    vertical_spacing= 0, #[0,0.02,0,0.02],
    subplot_titles=["Candlestick", "Price Bins", "Volume", ""]
    )
    return fig

def add_trades(data,row,col):
    fig.add_trace(
    go.Candlestick(
        x=data['plotdatetime'],
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name='Price'
    ),
    row=row,
    col=col
    )

    fig.add_trace(
        go.Bar(
            x=data['plotdatetime'],
            y=data['volume'],
            name='Volume'
        ),
        row=row+1,
        col=col
    )



def saveFiles(self):
        """
        saves the plot as a png file
        """
        self.fig.write_image("plot.png")

class StockPlot:
    """
    creates different types of plots for stock data
        1. sigviz plot  i.e. multiplot with signals
        2. single stock plot with selected features
        3. backtest plot with buys and sells (basically sigviz plot with buy/sell overlay)
        4. backtest stats plot
        5. featureviz which shows feature values to help define signals
    
    """

    def __init__(self,df,daysbefore = 0,daysafter = 0,plotIds = []):
        self.plotDf = df.copy()
        self.daysBefore = daysbefore
        self.daysAfter = daysafter
        self.addColumnsforPlottingFunc()
        self.plotIds = plotIds
        self.axisRanges = {}

    def addColumnsforPlottingFunc(self):
        self.plotDf['plotdatetime'] = self.plotDf.index.strftime("%y/%m/%d %H:%M:%S") #Need to add datetime as string as plotly is terribl with datetime
        df['primarykey'] = df['date'].astype(str) + df['ticker']

    def multiplot_specs(self):
        secaxlist = [[]]* 4
        for i in range(3):
            secaxlist[0].append({"horizontal_spacing": None})
        return secaxlist

    def make_subplots_multiplot(self):
        self.fig = make_subplots(
        rows=5,
        cols=3,
        #shared_xaxes="columns",
        #shared_yaxes="columns",
        # column_width=[0.3,0.3,0.3],
        row_heights=[0.2,0.06,0.025,0.2,0.06],
        #specs=self.multiplot_specs(),
        horizontal_spacing=0.04,
        vertical_spacing=0.04
        #subplot_titles=["Candlestick", "Price Bins", "Volume", ""]
        )
        #TODO: set volume colour, share x across cols and set titles.

    def sigviz(self):
        """
        plots a multiplot with signals
        """
        if not self.plotIds.empty :
            plotCount = self.plotIds.iloc[-1]['plotId']
            print(f'{plotCount} plots')

            totalPlotCount =1
            currentPlotCount = 1
            for pagecount in range(1,math.ceil(plotCount/(2 * 3))+1):
                self.make_subplots_multiplot()
                for row in range(1,6,3):
                    for col in range(1,4):
                        filteredDf = self.plotDf[self.plotDf['primarykey'].isin(self.plotIds[self.plotIds['plotId'] == totalPlotCount]['primarykey'])] #filter df for each plot
                        self.add_traces_multiplot(filteredDf,row,col,currentPlotCount)
                        totalPlotCount +=1
                        currentPlotCount +=1
                        self.update_multiplot_axes(row, col)

                self.update_multiplot_layout()
                self.fig.show()
                self.axisRanges = {}
                currentPlotCount = 1

        else:
            raise Exception('plotIds not defined')
 
    def add_traces_multiplot(self,data,row,col,currentPlotCount):

        self.fig.add_trace(
            go.Candlestick(
                x=data['plotdatetime'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price'
            ),
            row=row,
            col=col
        )

        self.fig.add_trace(
            go.Bar(
                x=data['plotdatetime'],
                y=data['volume'],
                name='Volume'
            ),
            row=row+1,
            col=col
        )
        
        if currentPlotCount >3:
            name = currentPlotCount + 9 #need to account for blank row
        else:
            name = currentPlotCount +3 

        self.axisRanges['yaxis'+str(name)] = [data['volume'].min(),data['volume'].quantile(0.99)]

   
    def update_multiplot_axes(self,row,col):
        self.fig.update_xaxes(dtick =60, row=row, col=col,rangeslider_visible=False,
        showticklabels=False)
        self.fig.update_xaxes(nticks =20,  row=row+1, col=col,rangeslider_visible=False,
        showticklabels=True,
        tickangle = -45)

    def update_multiplot_layout(self):
        self.fig.update_layout(
            template="plotly_dark",
            hovermode="x unified",
            title_text="Hollow Candlesticks",
            showlegend=False,
            yaxis4 = dict(range = self.axisRanges['yaxis4']),
            yaxis5 = dict(range = self.axisRanges['yaxis5']),
            yaxis6 = dict(range = self.axisRanges['yaxis6']),
            yaxis13 = dict(range = self.axisRanges['yaxis13']),
            yaxis14 = dict(range = self.axisRanges['yaxis14']),
            yaxis15 = dict(range = self.axisRanges['yaxis15'])

        )



plot = StockPlot(df,plotIds = plotIdTest)
plot.sigviz()


plot.plotDf.head()


