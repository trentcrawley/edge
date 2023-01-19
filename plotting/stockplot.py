import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import os
from tkinter.filedialog import askdirectory
from utils import getTestData
import math
import sigviz

df,plotIds =  sigviz.getDates()
plotIdTest = plotIds[:18]         

class StockPlot:
    """
    creates different types of plots for stock data
        1. sigviz plot  i.e. multiplot with signals
        2. single stock plot with selected features
        3. backtest plot with buys and sells (basically sigviz plot with buy/sell overlay)
        4. backtest stats plot
        5. featureviz which 
        s feature values to help define signals
    
    """

    def __init__(self,df,daysbefore = 0,daysafter = 0,plotIds = [],titleText = ""):
        self.plotDf = df.copy()
        self.daysBefore = daysbefore
        self.daysAfter = daysafter
        self.addColumnsforPlottingFunc()
        self.plotIds = plotIds
        self.axisRanges = []
        self.titleText = titleText

    def addColumnsforPlottingFunc(self):
        self.plotDf['plotdatetime'] = self.plotDf.index.strftime("%y/%m/%d %H:%M:%S") #Need to add datetime as string as plotly is terribl with datetime
        self.plotDf['primarykey'] = self.plotDf['date'].astype(str) + self.plotDf['ticker']

    def multiplot_specs(self): # add second axis to all plots
        secaxlist = [[]]* 2
        for i in range(3):
            secaxlist[0].append({"secondary_y": True})
        return secaxlist

    def make_subplots_multiplot(self): 
        self.fig = make_subplots(
        rows=2,
        cols=3,
        #shared_xaxes=True,
        #shared_yaxes=True,
        #column_width=[0.3,0.3,0.3],
        #row_heights=[0.2,0.2],
        specs=self.multiplot_specs(),
        horizontal_spacing=0.06,
        vertical_spacing=0.15,
        subplot_titles=("Plot 1", "Plot 2", "Plot 3", "Plot 4","Plot 5", "Plot 6")
        )

    def add_traces_multiplot(self,data,row,col,overlays = []):
        # temp force only one signal
        data['signal'] = 0
        data['signal'].iloc[50] =1
        if row ==1 and col==1: #only display legend for first plot to avoid repetition
            displayLegend = True
        else:
            displayLegend=False

        self.fig.add_trace(
            go.Candlestick(
                x=data['plotdatetime'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price',
                showlegend=displayLegend
                ),
                row=row,
                col=col
            )
        
        #set axis ranges
        ybottom = data['low'].min()*0.995
        ytop = data['high'].max()*1.0095
        self.axisRanges.append([ybottom,ytop])

        if overlays: # add overlays in overlays list
            colors = ['#636EFA','#FF7F0E','green','red','orange','purple','pink','brown','grey','black']
            for overlay in overlays:
                if overlay in data.columns:
                    self.fig.add_trace(
                        go.Scatter(
                            x=data['plotdatetime'],
                            y=data[overlay],
                            name=overlay,
                            mode='lines',
                            line=dict(color=colors[0], width=1),
                            showlegend=displayLegend
                        ),
                        row=row,
                        col=col
                    )
                    colors.pop(0)
                else:
                    print("overlay {overlay} not available")

        self.fig.add_trace(
            #add volume to second axis
            go.Bar(
                x=data['plotdatetime'],
                y=data['volume'],
                name='Volume',
                marker = {'color':'lightgoldenrodyellow'},
                showlegend=displayLegend            
            ),
            row=row,
            col=col,
            secondary_y=True
        )

        #Add signals to plot as verticl line
        signaldf = data[data['signal']==1]
        if not signaldf.empty:
            for index, line in signaldf.iterrows():
                
                self.fig.add_shape(
                    dict(
                        type='line',
                        yref = 'y', y0 =ybottom,y1=ytop,
                        xref = 'x', x0 = line['plotdatetime'],x1=line['plotdatetime'],
                        line=dict(color='white',width =1),
                        ),
                     row=row,
                     col=col,
                     secondary_y=False
                )
          
    def update_multiplot_axes(self,row,col):
        self.fig.update_xaxes(dtick =60, row=row, col=col,rangeslider_visible=False,
        showticklabels=True,tickangle = -35)
        self.fig.update_yaxes(title_text="Price", row=row, col=col,secondary_y=False)
        self.fig.update_yaxes(title_text="Volume", row=row, col=col,secondary_y=True)

    def update_multiplot_layout(self,plotTitleDict):
        self.fig.update_layout(
            template="plotly_dark",
            hovermode="x unified",
            title_text=self.titleText,
            margin=dict(l=70, r=5, t=70, b=70))

        # update candlestick ranges
        rangeIndex = 0
        plot_count = len([v for v in plotTitleDict.values() if v])# count get non blank titles
        for i in range(1,plot_count*2,2):
            self.fig.layout[f'yaxis{i}'].update(range = self.axisRanges[rangeIndex])
            rangeIndex+=1

        self.fig.for_each_annotation(lambda a: a.update(text = plotTitleDict[a.text]))

    def sigviz(self,overlays = []):
        """
        plots a multiplot with signals
        """
        if not self.plotIds.empty :
            plotCount = self.plotIds.iloc[-1]['plotId']
            print(f'{plotCount} plots')

            totalPlotCount =1
            currentPlotCount = 1

            # get user input to determine output method
            self.outputMethod = input('Output to file or browser? (f/b)')
            if self.outputMethod == 'f':
                Savedir = askdirectory()
                fileName = input('Enter file name: ')

            # 6 plots per page
            for pagecount in range(1,math.ceil(plotCount/(2 * 3))+1): #loop through all pages
                plotTitleDict = {}
                self.make_subplots_multiplot() #create figure object
                for row in range(1,3):
                    for col in range(1,4):
                        filteredDf = self.plotDf[self.plotDf['primarykey'].isin(self.plotIds[self.plotIds['plotId'] == totalPlotCount]['primarykey'])] #filter df for each plot
                        if not filteredDf.empty:
                            plotTitleDict['Plot '+ str(currentPlotCount)] = str(filteredDf.iloc[0]['ticker']) + " " + str(filteredDf.iloc[0]['date'])
                            self.add_traces_multiplot(filteredDf,row,col,overlays) #add price and volume traces
                        else:
                            plotTitleDict['Plot '+ str(currentPlotCount)] = ''
                        totalPlotCount +=1
                        currentPlotCount +=1
                        self.update_multiplot_axes(row, col)

                self.update_multiplot_layout(plotTitleDict)

                # output fig each page
                if self.outputMethod == 'b':
                    self.fig.show()
                else:
                    self.fig.write_html(f'{Savedir}/{fileName}_{pagecount}.html')
                self.axisRanges = []
                currentPlotCount = 1

        else:
            raise Exception('plotIds not defined')
 

plot = StockPlot(df,plotIds = plotIdTest)
plot.sigviz(overlays = ['vwap','ema9close1min'])

