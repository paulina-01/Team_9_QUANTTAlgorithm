import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Normalizer, MinMaxScaler,StandardScaler

class MVAproject(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2021, 5, 1)  # Set Start Date
        self.SetCash(100000)  # Set Strategy Cash
        #self.AddEquity("SPY", Resolution.Daily)
        self.SetEndDate(2022, 1, 29)
        self.stock_data = np.array([])
        self.AddEquity("APA", Resolution.Daily)
        self.AddEquity("OXY", Resolution.Daily)
        self.AddEquity("CTRA", Resolution.Daily)
        self.AddEquity("HAL", Resolution.Daily)
        self.apaMomentum = self.MOMP("APA", 30, Resolution.Daily)
        self.oxyMomentum = self.MOMP("OXY", 30, Resolution.Daily)
        self.crtaMomentum = self.MOMP("CTRA", 30, Resolution.Daily)
        self.halMomentum = self.MOMP("HAL", 30, Resolution.Daily)
        self.stock_data_apa = np.array([])
        self.stock_data_oxy = np.array([])
        self.stock_data_ctra = np.array([])
        self.stock_data_hal = np.array([])
        def OnSecuritiesChanged(self, changes):
            for security in changes.AddedSecurities:
                security.SetLeverage(5)

    def OnData(self, data):
        '''OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.
            Arguments:
                data: Slice object keyed by symbol containing the stock data
        '''
        '''
        if not self.Portfolio.Invested: #portfolio strategy can be added
            self.SetHoldings("APA", 0.25)
            self.SetHoldings("OXY", 0.25)
            self.SetHoldings("CTRA",0.25)
            self.SetHoldings("HAL", 0.25)
        '''
        #dict to store the momentum of each security that has a positive momentum
        securityMomDict = {}
        '''
        dict to store the proportion that each security’s momentum has of the sum of all of the securities’ momentums (sum only includes securities w/ positive momentums
        '''
        securityProportionDict = {}
        '''
        for each security with a positive momentum, add its momentum to securityMomDict
        '''
        for holding in self.Portfolio.Values:
            ticker = holding.Symbol.Value
            if data.ContainsKey(ticker):
                if not data.Bars.ContainsKey(ticker): return
            
                if 'APA' in ticker:
                    self.stock_data_apa = np.append(self.stock_data_apa, data[ticker].Close)
                if 'OXY' in ticker:
                    self.stock_data_oxy = np.append(self.stock_data_oxy, data[ticker].Close)
                if 'CTRA' in ticker:
                    self.stock_data_ctra = np.append(self.stock_data_ctra, data[ticker].Close)
                if 'HAL' in ticker:
                    self.stock_data_hal = np.append(self.stock_data_hal, data[ticker].Close)
                
            #if (self.apaMomentum.Current.Value > 0) and decision(self.stock_data_apa, 'APA',holding):
            if len(self.stock_data_apa) > 30:
                if bullish(self.stock_data_apa):
                    if self.apaMomentum.Current.Value > 0:#π
                        securityMomDict['APA'] = self.apaMomentum.Current.Value
            if len(self.stock_data_oxy) > 30:        
                if bullish(self.stock_data_oxy):
                    if self.oxyMomentum.Current.Value > 0:#π
                        securityMomDict['OXY'] = self.oxyMomentum.Current.Value
            if len(self.stock_data_ctra) > 30:     
                if bullish(self.stock_data_ctra):
                    if self.crtaMomentum.Current.Value > 0:#π
                        securityMomDict['CTRA'] = self.crtaMomentum.Current.Value
            if len(self.stock_data_hal) > 30:     
                if bullish(self.stock_data_hal):
                    if self.halMomentum.Current.Value > 0:#π
                        securityMomDict['HAL'] = self.halMomentum.Current.Value
                
            #liquidate full portfolio if all securities have negative momentums
            if len(securityMomDict) == 0:
                self.Liquidate('APA')
                self.Liquidate('OXY')
                self.Liquidate('CTRA')
                self.Liquidate('HAL')
            else:
                if "APA" not in securityMomDict:#π
                    self.Liquidate('APA')#π
                if "OXY" not in securityMomDict:#π
                    self.Liquidate('OXY')#π
                if "CTRA" not in securityMomDict:#π
                    self.Liquidate('CTRA')#π
                if "HAL" not in securityMomDict:#π
                    self.Liquidate('HAL')#π
                #add the proportion of each security’s momentum (proportion of the sum of all positive momentums)
                totalMom = sum(securityMomDict.values())
                for security in securityMomDict:
                    securityProportionDict[security] = securityMomDict[security]/(totalMom)
                #set holdings based on the proportion of total momentum of each security
                for security in securityProportionDict:
                    if security == 'APA':
                        self.SetHoldings('APA', securityProportionDict[security])
                    elif security == 'OXY':
                        self.SetHoldings('OXY', securityProportionDict[security])
                
                    elif security == 'CTRA':
                        self.SetHoldings('CTRA', securityProportionDict[security])
                    elif security == 'HAL':
                        self.SetHoldings('HAL', securityProportionDict[security])

def bullish(series) -> bool: #Bullish functions are one of the implementations (but there can be more complex with things other than regressions* interesting)
    lb = 30
    peaks,_ = find_peaks(series)
    troughs,_ = find_peaks(-series)
    
    #Optima peaks and troughs
    optima_data_top = regress_optima(series, peaks, lb) # estimated peak for next purchase
    optima_data_bottom = regress_optima(series, troughs, lb) # estimated troughs for next purchase
    estimate_price = regress_next(series, lb)
    
    if((estimate_price > optima_data_top)):
        return True

    elif((estimate_price < optima_data_bottom)):
        return False
        
    else:
        return None

def regress_optima(data: np.ndarray, optima: np.ndarray, lb: int) -> (int): #lb = lookback period
    """
    Output is estimated optima for next purchase
    """
    
    optimas = optima[-lb:] #array of indexes where its either peak or trough
    optima_predict = np.array(len(data)) # next purchase time's index
    y_data = [] #middleman
    for val in optimas:
        y_data.append(data[val])
    y_data = np.array(y_data) #converting list into numpy array
    #model = LinearRegression()
    model = KNeighborsRegressor(n_neighbors=4) #138.44 %
    #model = XGBRegressor()# slow
    #model = SVR()
    #model = RandomForestRegressor()
    #Standardization = StandardScaler()
    Standardization = MinMaxScaler()

    Standardization.fit(y_data.reshape(-1,1))
    model.fit(optimas.reshape(-1,1),Standardization.transform(y_data.reshape(-1,1)))
    next_optima = model.predict(optima_predict.reshape(-1,1))
    #next_optima = Standardization.inverse_transform(next_optima.reshape(-1,1))
    return next_optima
    
def regress_next(data: np.ndarray, lb: int) -> (int):
    """
    Output is estimated data for next purchase
    """
    #model = LinearRegression()#80%, scaler no change
    model = KNeighborsRegressor(n_neighbors=4)
    #model = XGBRegressor()# slow, 77.42%
    #model = SVR()#96.12%, minmax 86.73%
    #model = RandomForestRegressor()# 88.41%, slow
    #Standardization = StandardScaler()
    Standardization = MinMaxScaler()

    Standardization.fit(data[-lb:].reshape(-1,1))
    model.fit(np.arange(0,lb).reshape(-1,1),Standardization.transform(data[-lb:].reshape(-1,1)))
    next_price = model.predict(np.array(lb).reshape(-1,1))
    next_price= Standardization.inverse_transform(next_price.reshape(-1,1))
    return next_price
