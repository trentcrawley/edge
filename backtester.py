import numpy as np
import pandas as pd
from plotting import features as ft
import datetime
pd.options.mode.chained_assignment = None

class Backtester():

    def __init__(self,dataframe,target = 0.01,stoptype = 'hard',stoploss = 0.005,brokerage = 0.002,tradevalue  = 10000, maxDays = 2):
        #take in dataframe with a signal column as either 'buy' or 'sell'
        self.dataframe = dataframe.copy()
        self.target = target
        self.stoptype = stoptype
        self.stoploss = stoploss
        if self.stoptype=='trail':
            self.stopcol = 'close'
        else:
            self.stopcol = self.stoptype
        self.brokerage=  brokerage
        self.tradevalue = tradevalue
        self.maxDays = maxDays
        self.minDays = -5
        self.dataframe = ft.nextxdayclose(self.dataframe, self.maxDays)
        self.trades = self.generatetrades(self.dataframe, self.target, self.maxDays)
        self.trades['date'] = pd.to_datetime(self.trades['date'])
        self.dataframe['date'] = pd.to_datetime(self.dataframe['date'])
        self.dataframe = self.jointrades(self.trades, self.dataframe)
        self.dataframe = self.backteststats(self.dataframe)

    def checkData(self):
        if 'date' not in self.dataframe.columns:
            self.dataframe['date'] = self.dataframe.index.date

        if 'datetime' in self.dataframe.columns:
            self.dataframe.set_index('datetime',inplace=True)
        
        if 'sell' or 'buy' not in self.dataframe['signal'].unique():
            self.dataframe['signal'] = np.where(self.dataframe['signal'] == 1, 'sell', '')
            print('signal not defined as buy or sell, setting to sell')

    def sortData(self,df):
        df.sort_values(by = ['datetime','ticker'],inplace=True)
        return df

    def FilterTradeDays(self):
        #go though and filter for only days with a trade + days set in maxDays for multiday trades
        self.tradedays = self.dataframe[self.dataframe['signal'] != ''][['date', 'ticker','tminus'+str(abs(self.minDays)),'tplus' + str(abs(self.maxDays))]].drop_duplicates()
        dayDict = {'date': [], 'ticker': []}

        for row in self.tradedays.itertuples():
         if pd.isna(getattr(row, 'tminus'+str(abs(self.minDays)))) == False and pd.isna(getattr(row,'tplus'+str(abs(self.maxDays)))) == False:

                startDate = getattr(row, 'tminus'+str(abs(self.minDays))).date()
                endDate = getattr(row, 'tplus'+str(abs(self.maxDays))).date()
                daysbetween = endDate - startDate
                for days in range(daysbetween.days + 1):
                    current_date = startDate + datetime.timedelta(days)
                    if current_date.weekday() < 5:
                        dayDict['date'].append(current_date)
                        dayDict['ticker'].append(row.ticker)

                dayids = pd.DataFrame(dayDict)

                if not dayids.empty:
                    self.filteredDf = self.dataframe.reset_index().merge(dayids, on=['ticker', 'date'],
                                                                                how='inner').set_index('datetime')
                else:
                    self.filteredDf = pd.DataFrame()

    def getStop(self, value, stopvalue, type, loss, side,openlong=False, openshort=False):

        """
        :param type:'hard','trail','vwap',TODO: 'hod','lod'
        """
        if openlong == False and openshort == False:
            # initialise
            if side == 'buy':
                stop = (1 - loss) * value
            else:
                stop = (1 + loss) * value

        elif openlong == True:
            if type != 'hard':
                stop = max((1 - loss) * value, stopvalue)
            else:
                stop = stopvalue

        elif openshort == True:
            if type != 'hard':
                stop = min((1 + loss) * value, stopvalue)
            else:
                stop = stopvalue
        return round(stop, 4)

    def generatetrades(self,df, target, maxdays):

        # takes a data frame with sell and buy signals, iterates through and opens and closes trades based on stop, target
        # and time rules. Only opens a trade when not already open. Return dataframe of trades.
        openlong = False
        openshort = False
        skip = True
        counter = 1
        data = df.copy()
        data.sort_values(by = ['ticker','datetime'],inplace=True)
        trades = pd.DataFrame(columns=['datetime', 'ticker', 'side', 'price', 'quantity', 'tradeid', 'date', 'state'])


        def createtrades(trades, side, openpos):
            # fills a dict with trades based on signal then appends to trades dataframe

            price = row.close

            if side == 'sell' and openlong == False:
                quantity = -round(10000 / price)
                state = 'short'
            elif side == 'buy' and openshort == False:
                quantity = round(10000 / price)
                state = 'long'
            elif side == 'sell' and openlong == True:
                quantity = openpos[row.ticker]
                state = 'long'
            elif side == 'buy' and openshort == True:
                quantity = openpos[row.ticker]
                state = 'short'

            tradedict = {'tradeid': counter, 'datetime': row.Index, 'ticker': row.ticker, 'side': side,
                         'price': row.close, 'quantity': quantity, 'state': state}


            trades = trades.append(tradedict, ignore_index=True)
            #print(trades)

            return trades


        for row in data.itertuples():
            if openlong == False and openshort == False:
                openpos = {}
                if row.signal == 'buy' and row.Index.time() < datetime.time(15, 50):

                    stoppx = self.getStop(openlong = openlong,openshort = openshort,value = getattr(row,self.stopcol),type = self.stoptype,loss = self.stoploss,side =row.signal,stopvalue=0)
                    targetpx = round(row.close * (1 + target), 4)
                    trades = createtrades(trades, row.signal, openpos)
                    openlong = True
                    lasttrades = trades[-1:]
                    openpos = {lasttrades['ticker'].iloc[0]: lasttrades['quantity'].iloc[0]}
                    maxdate = getattr(row,'tplus' + str(maxdays))
                    skip = True

                elif row.signal == 'sell' and row.Index.time() < datetime.time(15, 50):
                    stoppx = self.getStop(openlong=openlong, openshort=openshort, value=getattr(row, self.stopcol),
                                          type=self.stoptype, loss=self.stoploss, side=row.signal,stopvalue=0)

                    targetpx = round(row.close * (1 - target), 4)
                    trades = createtrades(trades, row.signal, openpos)
                    openshort = True
                    lasttrades = trades[-1:]
                    openpos = {lasttrades['ticker'].iloc[0]: lasttrades['quantity'].iloc[0]}
                    maxdate = getattr(row, 'tplus' + str(maxdays))
                    skip = True

            elif openlong == True:
                stoppx = self.getStop(openlong=openlong, openshort=openshort, value=getattr(row, self.stopcol),
                                      type=self.stoptype, loss=self.stoploss, side=row.signal,stopvalue=stoppx)
                if row.close < stoppx or row.close > targetpx or row.Index == maxdate:
                    trades = createtrades(trades, 'sell', openpos)
                    openlong = False
                    openpos = {}
                    stoppx = 0
                    targetpx = 0
                    counter += 1
                    skip = True

            elif openshort == True:
                stoppx = self.getStop(openlong=openlong, openshort=openshort, value=getattr(row, self.stopcol),
                                      type=self.stoptype, loss=self.stoploss, side=row.signal,stopvalue=stoppx)
                if row.close > stoppx or row.close < targetpx or row.Index == maxdate:
                    trades = createtrades(trades, 'buy', openpos)
                    openshort = False
                    openpos = {}
                    stoppx = 0
                    targetpx = 0
                    counter += 1
                    skip = True


        trades['date'] = trades.datetime.dt.date
        trades.set_index('datetime', inplace=True)
        return trades

    def jointrades(self,tradesdf, fulldf):
        # join the trades df back to the full data df for ease of plotting backtest results
        def openpositionfilter(merged):
            # filters larger dataframe by when there is an open trade and fills down the trade id so we can groupby tradeid
            grouped = merged.groupby('tradeid')
            for name, group in grouped:

                mask = ((merged['datetime'] >= grouped.get_group(name).datetime.iloc[0]) & (
                        merged['datetime'] <= grouped.get_group(name).datetime.iloc[-1])
                        & (merged['ticker'].isin(grouped.get_group(name).ticker.unique())))
                if name == 1:
                    filter = mask
                else:
                    filter = mask | filter

            opentrade = merged.loc[filter]
            cols = ['tradeid', 'quantity', 'state']
            opentrade.loc[:, cols] = opentrade.loc[:, cols].ffill()
            tradeids = opentrade[['tradeid', 'datetime', 'ticker', 'quantity', 'state']]
            merged.drop(columns=['tradeid', 'quantity', 'state'], inplace=True)
            merged = pd.merge(left=merged, right=tradeids, left_on=['datetime', 'ticker'],
                              right_on=['datetime', 'ticker'],
                              how='left')
            return merged

        filterdays = tradesdf[['ticker', 'date']].drop_duplicates()
        filtereddf = pd.merge(left=fulldf.reset_index(), right=filterdays, how='left', left_on=['date', 'ticker'],
                              right_on=['date', 'ticker'])
        filtereddf.drop(columns='date', inplace=True)
        merged = pd.merge(left=filtereddf, right=tradesdf, how='left', left_on=['datetime', 'ticker'],
                          right_on=['datetime', 'ticker']).drop_duplicates()  # .sort_values(by = 'datetime')
        merged = openpositionfilter(merged)
        merged['date'] = merged.datetime.dt.date
        merged.set_index('datetime', inplace=True)
        return merged



    def backteststats(self,df):

        df['tradevalue'] = df['quantity'] * df['close']
        #df['pl'] = df.groupby('tradeid')['tradevalue'].transform(firstdiff)
        df['pl'] = df['tradevalue'].diff(1)
        df['brokerage'] = np.where((df['side'] == 'buy') | (df['side'] == 'sell'), df['tradevalue'].abs() * 0.0006, 0)
        df['pl'].fillna(0, inplace=True)
        df['netpl'] = df['pl'] - df['brokerage']
        return df
