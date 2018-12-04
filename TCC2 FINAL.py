# -*- coding: utf-8 -*-
"""
PMR 3550 - TRABALHO DE CONCLUSÃO DE CURSO II EM ENGENHARIA MECATRÔNICA
SISTEMA AUTOMÁTICO DE INVESTIMENTO BASEADO EM MACHINE LEARNING

RODRIGO LAZZARIS FERNANDEZ 
NUSP 8582940

"""


#import relevant libraries
import pandas as pd
import xgboost as xgb
import talib
import matplotlib.pyplot as plt


#variables
data_path = 'C:\Users\Rodrigo\Desktop\Poli 10o Semestre\PMR3550 - TCC2\Dados\\' #dataset location
ahead = 46 #number of days ahead for xgboost model prediction and portfolio reconstruction
pct_train = 0.6 #fraction of dataset used for training, remainder used for testing
n_best_stocks = 10 #number of stocks chosen for portfolio on each portfolio reconstruction


def main(data_path, ahead, pct_train, n_best_stocks):

    #close all previously generated images
    plt.close('all')
    
    #read historical data
    df_open  = pd.read_csv(data_path+'px_open.csv', sep=";", decimal=",", index_col='Date')
    df_high  = pd.read_csv(data_path+'px_high.csv', sep=";", decimal=",", index_col='Date')
    df_low   = pd.read_csv(data_path+'px_low.csv', sep=";", decimal=",", index_col='Date')
    df_close = pd.read_csv(data_path+'px_close.csv', sep=";", decimal=",", index_col='Date')
    
    
    #create prediction dataframe
    df_total_pred = pd.DataFrame()
    
    
    #create xgboost model for each stock and plot panel
    for stock in df_close.columns[:10]:
        
        print stock
        
        df_data, df_x, df_y, df_x_train, df_y_train, df_x_test, df_y_test = create_dataset(stock, df_open, df_high, df_low, df_close, pct_train)
        
        #XGBoost
        model = xgb.XGBRegressor(seed=1, max_depth=3, n_estimators= 50, silent=True, min_child_weight=1, colsample_bytree=0.75)
        model.fit(df_x_train, df_y_train, eval_set=[(df_x_train, df_y_train)], eval_metric='rmse', verbose=False)
        results = model.evals_result()
        plt.figure()
        plt.plot(results['validation_0']['rmse'],  marker='o', markersize=3)
        plt.title(stock + " - Erro vs. Numero de Arvores")
        plt.xlabel("Numero de Arvores")
        plt.ylabel("Erro quadratico medio")
           
        #Accuracy Test
        tp_train, tn_train, fp_train, fn_train = get_accuracy(df_y_train.values, model.predict(df_x_train))
        tp_test, tn_test, fp_test, fn_test = get_accuracy(df_y_test.values, model.predict(df_x_test))
    
        #Predictions
        df_total_pred[stock] = model.predict(df_x_test) #include model predictions to dataframe with all model predictions
        df_total_pred.index = df_x_test.index
              
        #Plot panel
        plot_panel(stock, model, pct_train, df_data, df_x_train, df_x_test, df_y_train, df_y_test, tp_train, tn_train, tp_test, tn_test)
    
    
    #Historical Backtest
    relative_perf = backtest(df_close, df_total_pred, n_best_stocks)
    print relative_perf


##############################################################################################################
##############################################################################################################
def create_dataset(stock, df_open, df_high, df_low, df_close, pct_train):
    
    #Get stock prices
    df_data = pd.DataFrame()
    df_data['Open']  = df_open[stock]
    df_data['High']  = df_high[stock]
    df_data['Low']   = df_low[stock]
    df_data['Close'] = df_close[stock]
    
    #Generate features from imported data
    df_data['Mom5'] = df_data.Close/df_data.Close.shift(5) - 1
    df_data['Mom15'] = df_data.Close/df_data.Close.shift(15) - 1    
    df_data['Mom25'] = df_data.Close/df_data.Close.shift(25) - 1
    df_data['Mom35'] = df_data.Close/df_data.Close.shift(35) - 1
    df_data['Mom50'] = df_data.Close/df_data.Close.shift(50) - 1 
    df_data['Mom100'] = df_data.Close/df_data.Close.shift(100) - 1
    df_data['ADX'] = talib.ADX(df_data.High, df_data.Low, df_data.Close)
    df_data['RSI'] = talib.RSI(df_data.Close)
    df_data['ATR'] = talib.ATR(df_data.High, df_data.Low, df_data.Close)
    df_data['SAR'] = talib.SAR(df_data.High, df_data.Low)/df_data.Close

    #Generate classification/regression
    df_data['Ahead'] = df_data.Close.shift(-ahead)/df_data.Close - 1
    df_data = df_data.dropna()
    
    #Slice dataframe to obtain x and y 
    df_x = df_data.drop(columns=['Open', 'High', 'Low', 'Close', 'Ahead']).copy()
    df_y = df_data[['Ahead']].copy()
    #Slice x and y between train and test
    df_x_train = df_x.head(int(pct_train*len(df_data))).copy()
    df_y_train = df_y.head(int(pct_train*len(df_data))).copy()
    df_x_test  = df_x.tail(int((1-pct_train)*len(df_data)))[ahead:].copy()
    df_y_test  = df_y.tail(int((1-pct_train)*len(df_data)))[ahead:].copy()
    
    return df_data, df_x, df_y, df_x_train, df_y_train, df_x_test, df_y_test

##############################################################################################################
##############################################################################################################


def get_accuracy(data, pred):
    df_acc = pd.DataFrame()
    df_acc['pred'] = pred
    df_acc['data'] = data    
    for i in range(len(df_acc)):
        if df_acc.data[i]>0 and df_acc.pred[i]>0:
            df_acc.at[i, 'Class'] = "True Positive"
        if df_acc.data[i]<0 and df_acc.pred[i]>0:
            df_acc.at[i, 'Class'] = "False Positive"
        if df_acc.data[i]<0 and df_acc.pred[i]<0:
            df_acc.at[i, 'Class'] = "True Negative"
        if df_acc.data[i]>0 and df_acc.pred[i]<0:
            df_acc.at[i, 'Class'] = "False Negative"    
    acc_tp = float(len(df_acc[df_acc.Class == 'True Positive']))/len(df_acc)
    acc_tn = float(len(df_acc[df_acc.Class == 'True Negative']))/len(df_acc)
    acc_fp = float(len(df_acc[df_acc.Class == 'False Positive']))/len(df_acc)
    acc_fn = float(len(df_acc[df_acc.Class == 'False Negative']))/len(df_acc)    
    return acc_tp, acc_tn, acc_fp, acc_fn

##############################################################################################################
##############################################################################################################

def plot_panel(stock, model, pct_train, df_data, df_x_train, df_x_test, df_y_train, df_y_test, tp_train, tn_train, tp_test, tn_test):

    plt.figure() #open new figure
    plt.gcf().set_size_inches(12.45, 17.55) #plot size definition
    
    
    xgb.plot_importance(model, ax=plt.subplot(233))
    plt.title(stock + " - Relevancia dos Indicadores")
    xgb.plot_tree(model, num_trees=0, ax=plt.subplot(212))
    
    #plot training time series data
    plt.subplot(431)
    plt.title(stock + " - Dados (Treino)")
    plt.xlabel("Data")
    plt.ylabel("Preco")
    plt.scatter(x=range(int(pct_train*len(df_data))), y=df_data.head(int(pct_train*len(df_data))).Close, s=1.5, marker='.', c=['g' if pred>=0 else 'r' for pred in model.predict(df_x_train)])
    plt.xticks(range(int(pct_train*len(df_data)))[::100], df_data.head(int(pct_train*len(df_data))).index[::100], rotation=90, fontsize=6)
    plt.margins(x=0)
    
    #plot testing time series data
    plt.subplot(434)
    plt.title(stock + " - Dados (Teste)")
    plt.xlabel("Data")
    plt.ylabel("Preco")
    plt.scatter(x=range(int((1-pct_train)*len(df_data))-ahead), y=df_data.tail(int((1-pct_train)*len(df_data))-ahead).Close, s=1.5, marker='.', c=['g' if pred>=0 else 'r' for pred in model.predict(df_x_test)])
    plt.xticks(range(int((1-pct_train)*len(df_data))-ahead)[::100], df_data.tail(int((1-pct_train)*len(df_data))-ahead).index[::100], rotation=90, fontsize=6)
    plt.margins(x=0)
       
    #scatter plot for training data predictions
    plt.subplot(432)
    plt.title(stock + " - Previsoes (Treino) - Acc. %.1f" %(100*(tp_train+tn_train)) + "%")
    plt.axhline(y=0, color='k', lw=0.8)
    plt.axvline(x=0, color='k', lw=0.8)
    plt.xlabel("Variacao Realizada")
    plt.ylabel("Variacao Prevista")
    plt.scatter(df_y_train.values, model.predict(df_x_train), s=0.6, c=['g' if pred>=0 else 'r' for pred in model.predict(df_x_train)])
     
    #scatter plot for testing data predictions
    plt.subplot(435)
    plt.title(stock + " - Previsoes (Teste) - Acc. %.1f" %(100*(tp_test+tn_test)) + "%")
    plt.axhline(y=0, color='k', lw=0.8)
    plt.axvline(x=0, color='k', lw=0.8)
    plt.xlabel("Variacao Realizada")
    plt.ylabel("Variacao Prevista")
    plt.scatter(df_y_test.values, model.predict(df_x_test), s=0.6, c=['g' if pred>=0 else 'r' for pred in model.predict(df_x_test)])
    
    #subplots layout format
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.99, top=0.98, bottom=0.01)
    plt.savefig('plots//'+stock+'.jpg')

##############################################################################################################
##############################################################################################################

def backtest(df_close, df_total_pred, n_best_stocks):
    
    #create market and porftolio accumulated performance variables
    acc_portfolio = 0
    acc_mkt = 0
    acc_portfolio_l = [0]
    acc_mkt_l = [0]
      
    #create return dataframe
    df_daily_ret = (df_close/df_close.shift(1) - 1).dropna()
    
    i=0
    while i < len(df_total_pred):    
        
        #Porfolio reconstruction
        if i%ahead == 0:
            portfolio = df_total_pred.iloc[i].sort_values(ascending=False)[:n_best_stocks].index.sort_values()
            #print "Portolio Update"
            #print df_total_pred.index[i]
            #print portfolio.tolist()
            
        ret_portfolio = df_daily_ret[portfolio].loc[df_total_pred.index[i]].mean() #calculate daily average return for the portfolio
        ret_mkt = df_daily_ret.loc[df_total_pred.index[i]].mean() #calculate daily average return for the market
        
        acc_portfolio = (1+acc_portfolio)*(1+ret_portfolio)-1 #compounded performance of the portfolio
        acc_mkt = (1+acc_mkt)*(1+ret_mkt) - 1 #compounded performance of the market
        
        acc_portfolio_l.append(acc_portfolio)
        acc_mkt_l.append(acc_mkt)
              
        i+=1
    
    #plot backtest
    plt.figure()
    plt.plot(acc_portfolio_l, label="Portfolio")
    plt.plot(acc_mkt_l, label="Mercado")
    plt.legend()
    plt.title("Teste Historico")
    plt.xlabel("Data")
    plt.ylabel("Rentabilidade")
    plt.xticks(range(len(df_total_pred.index))[::50], df_total_pred.index[::50], rotation=90, fontsize=8)
    plt.margins(x=0)
    plt.tight_layout()
    
    
    print acc_portfolio
    print acc_mkt
    relative_perf = acc_portfolio/acc_mkt #portfolio performance relative to market
    
    return relative_perf

##############################################################################################################
##############################################################################################################


main(data_path, ahead, pct_train, n_best_stocks)
