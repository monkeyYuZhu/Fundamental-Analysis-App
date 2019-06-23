"""
Updated on Wed Aug 15 10:26:54 2018

@author: Yu (Zack) Zhu
"""
from __future__ import division
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import numpy as np

#================================================Fundamental Analysis================================================#
""" Store common attributes of financial analysis """

    
class FinMetric(object):
    count = 0  # calculate how many instances have been created so far
    
    def __init__(self, symbol, CoGS, NI, dividend, liability, equity, B_Inventory, E_Inventory):
        FinMetric.count += 1
        self.symbol = symbol
        self.CoGS = CoGS
        self.NI = NI
        self.dividend = max(0, dividend)  # Ensure the dividend number is positive
        self.liability = liability
        self.equity = equity
        self.B_Inventory = B_Inventory
        self.E_Inventory = E_Inventory
    
    """ So far the list of financial metrics we can calculate here """
    def cal_Debt_to_Equity(self):
        return self.liability/self.equity
    
    def cal_Inventory_Turnover(self):
        avg_Inventory = (self.B_Inventory + self.E_Inventory)/2.0
        return self.CoGS/avg_Inventory
    
    def cal_ROE(self):
        return self.NI/self.equity
    
    """ Return those financial metrics """
    def set_precision(self, num):
        self.num = max(1, num)
    
    def get_inputinfo(self):
        return {'cost of goods sold': self.CoGS, 
                'net income': self.NI, 
                'dividend': self.dividend,
                'liability': self.liability,
                'equity': self.equity,
                'beginning inventory': self.B_Inventory,
                'ending inventory': self.E_Inventory}
    
    def get_metrics(self):
        Debt_to_Equity = round(self.cal_Debt_to_Equity(), self.num)
        Inventory_Turnover = round(self.cal_Inventory_Turnover(), self.num)
        ROE = round(self.cal_ROE(), self.num)
        
        return {'Debt to Equity': Debt_to_Equity, 
                'Inventory Turnover': Inventory_Turnover, 
                'Return on Equity': ROE}

# Give it a try
F = FinMetric('F', 20833, 1611, 1.01, 31437, 20447, 3209, 3560)

HMC = FinMetric('HMC', 20833, 1611, 1.01, 31437, 20447, 3209, 3560)
HMC.set_precision(2)
HMC.get_metrics()

Tata = FinMetric('TTM', 1670895, 61211, 0, 2132449, 534197, 326370, 352954)
Tata.set_precision(2)
Tata.get_metrics()

Ashok = FinMetric('ASHOKLEY.NS', 196384722, 17603817, 0, 252720712, 57056115, 29010292, 22076891)
Ashok.set_precision(2)
Ashok.get_metrics()


#================================================Data Source================================================#
pool = ['BOSCHLTD', 'PCAR', 'TEN', 'NVDA', 'LEA', 'MEI', 'HMC', 'RACE', 'CMI', 'PAG']
start = datetime.datetime(2008, 8, 5)
end = datetime.datetime(2018, 8, 14)

class BasicInfo(object):
    
    # attributes
    def __init__(self, ticker, start, end, source):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.source = source

# child class
class Data(BasicInfo):
    
    def download(self):
        result = web.DataReader(self.ticker, self.source, self.start, self.end)['close']
        frame = result.to_frame()
        return frame
        
    def movingAvg(self, num):
        frame = self.download()
        SMA = frame.rolling(window=num).mean()
        return SMA

def getTable(data):
    ticker_10 = data.rolling(window=10).mean()
    ticker_50 = data.rolling(window=50).mean()
    Table = pd.concat([data, ticker_10, ticker_50], axis=1)
    Table.columns = ['Close Price', '10-day SMA', '50-day SMA']
    return Table

def plotTable(ticker, table):
    plt.style.use('dark_background') # For plotting
    table.plot(lw=2, colormap='Paired', markersize=3, grid = "on", title='%s 3-year Moving Average' % ticker, figsize = (12,5))
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.show()

## download data
BOSCHLTD = pd.read_csv("BOSCHLTD.csv", sep=",")
BOSCHLTD_data = BOSCHLTD.set_index("date")
BOSCHLTD_data.columns = ["close"]

## COnvert the currency for BOSCHLTD
def curConver(x):
    return x*0.0145

r = map(curConver, list(BOSCHLTD_data["close"]))
rr = list(r)
BOSCHLTD_data = pd.DataFrame({'close':rr})
BOSCHLTD_data = BOSCHLTD_data.set_index(BOSCHLTD["date"])
boschTable = getTable(BOSCHLTD_data)
plotTable('BOSSCHLTD', boschTable)

PCAR = pd.read_csv("PCAR.csv", sep=",")
PCAR_data = PCAR.set_index("date")
pcarTable = getTable(PCAR_data)
plotTable('PCAR', pcarTable)

TEN = Data("TEN", start, end, 'yahoo')
TEN_data = TEN.download()
tenTable = getTable(TEN_data)
plotTable('TEN', tenTable)

NVDA = Data("NVDA", start, end, 'iex')
NVDA_data = NVDA.download()
nvdaTable = getTable(NVDA_data)
plotTable('NVDA', nvdaTable)

LEA = Data("LEA", start, end, 'iex')
LEA_data = LEA.download()
leaTable = getTable(LEA_data)
plotTable('LEA', leaTable)

MEI = Data("MEI", start, end, 'iex')
MEI_data = MEI.download()
meiTable = getTable(MEI_data)
plotTable('MEI', meiTable)

HMC = Data("HMC", start, end, 'iex')
HMC_data = HMC.download()
hmcTable = getTable(HMC_data)
plotTable('HMC', hmcTable)

RACE = Data("RACE", start, end, 'iex')
RACE_data = RACE.download()
raceTable = getTable(RACE_data)
plotTable('RACE', raceTable)

CMI = Data("CMI", start, end, 'iex')
CMI_data = CMI.download()
cmiTable = getTable(CMI_data)
plotTable('CMI', cmiTable)

PAG = Data("PAG", start, end, 'iex')
PAG_data = PAG.download()
PAG_10 = PAG_data.rolling(window=10).mean()
PAG_50 = PAG_data.rolling(window=50).mean()
pagTable = pd.concat([PAG_data, PAG_10, PAG_50], axis=1)
pagTable.columns = ['Close Price', '10-day SMA', '50-day SMA']


# Plot
plt.style.use('dark_background') # For plotting
pagTable.plot(lw=2, colormap='Paired', markersize=3, grid = "on", title='%s 3-year Moving Average' % "PAG", figsize = (12,5))
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.show()


#================================================Portfolio Performance Prediction================================================#
## Monte Carlo Simulation for the portfolio

def MC_portfolio(assets, rf, N):
    
    # 'BOSCHLTD', 'PCAR', 'TEN', 'NVDA', 'LEA', 'MEI', 'HMC', 'RACE', 'CMI', 'PAG'
    table = pd.concat([BOSCHLTD_data, PCAR_data, TEN_data, NVDA_data, LEA_data, MEI_data, HMC_data, RACE_data, CMI_data, PAG_data], axis=1)
    table.columns = assets

    # return
    daily_return = table/table.shift(1) - 1
    daily_return = daily_return.fillna(0)
    daily_return = daily_return.replace(np.inf, 0)
    annual_mean_return = daily_return.mean()*252

    # covariance
    daily_cov = daily_return.cov()
    annual_cov = daily_cov*252
    
    # Monte Carlo simulation
    nofasset = len(assets)
    nofasset = len(pool)
    E_p = []; S_p = []; W_p = []; 

    for single in range(N):
    
        weight = np.random.random(nofasset)
        weight /= sum(weight)
        port_return = np.dot(weight, annual_mean_return)
        port_variance = np.dot(weight.T, np.dot(annual_cov, weight))
        port_vol = np.sqrt(port_variance)
    
        E_p.append(port_return)
        S_p.append(port_vol)
        W_p.append(weight)

    portfolio_table = {'Portfolio Return': E_p,
                       'Volatility': S_p
                       }

    for count, ticker in enumerate(assets):
        portfolio_table[ticker + ' weight'] = [w[count] for w in W_p]

    extended_table = pd.DataFrame(portfolio_table)

    return extended_table


result = MC_portfolio(assets=pool, rf=0.0208, N=100000)

#================================================Convert to CSV================================================#
## the high and low stocks last 3 years and 30 days 
## pull out the 10 year high and low as well
## csv format     
historical_price_Table = pd.concat([BOSCHLTD_data, CMI_data, HMC_data, LEA_data, MEI_data, NVDA_data, PAG_data, PCAR_data, RACE_data, TEN_data], axis=1)
historical_price_Table.columns = ['BOSCH', 'CMI', 'HMC', 'LEA', 'MEI', 'NVDA', 'PAG', 'PCAR', 'RACE', 'TEN']
historical_price_Table.to_csv("Historical Prices Table Utilized for Analysis.csv")

boschTable.to_csv("boschMV.csv")
cmiTable.to_csv("cmiMV.csv")
hmcTable.to_csv("hmcMV.csv")
leaTable.to_csv("leaMV.csv")
meiTable.to_csv("meiMV.csv")
nvdaTable.to_csv("nvdaMV.csv")
pagTable.to_csv("pagMV.csv")
raceTable.to_csv("raceMV.csv")
tenTable.to_csv("tenMV.csv")
pcarTable.to_csv("pcarMV.csv")

result.to_csv("Monte Carlo result.csv")
