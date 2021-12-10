# -*- coding: utf-8 -*-
"""
Created on 20200523

@author: mchen

version:
1.2    add error handler, illiquid ratio changes to percentage
1.3    add list of VaRs, illiquid ratio
1.4    remove package check/installation to simplify dependency management,
       replace with pyinstaller packaging

"""

import pandas as pd
import numpy as np
import logging
import os
import time
import getpass
from carnbrea_tools import start_logger
import sys
import PySimpleGUI as sg
import warnings
warnings.filterwarnings('ignore')

def read_asset_class_file(file_asset_class_assumption, 
                          asset_class_names):
    
    """
    read asset class file (forward-looking)
    
    args:
        
        file_asset_class_assumption (str): file name. 
            file must contain tabs: 'return', 'stdev', 'correlation'
        
        asset_class_names (list of str): names of asset classes used in portfolios
    
    returns:
        
        df_asset_class_return (df): asset class annualised return
        df_asset_class_stdev (df): asset class annualised stdev
        df_asset_class_corr (df): asset class daily return correlation
        
    """
    
    df_asset_class_return = pd.read_excel(file_asset_class_assumption, 
                                           sheet_name = 'return',
                                           index_col = 0)

    assert (all(elem in list(df_asset_class_return.index)  for elem in asset_class_names)), "missing asset return"
    
    df_asset_class_return = df_asset_class_return.loc[asset_class_names].fillna(0)

    df_asset_class_stdev = pd.read_excel(file_asset_class_assumption, 
                                          sheet_name = 'stdev',
                                          index_col = 0)

    assert (all(elem in list(df_asset_class_stdev.index)  for elem in asset_class_names)), "missing asset stdev"
    
    df_asset_class_stdev = df_asset_class_stdev.loc[asset_class_names].fillna(0)
    
    df_asset_class_corr = pd.read_excel(file_asset_class_assumption, 
                                         sheet_name = 'correlation',
                                         index_col = 0)
    
    assert (all(elem in list(df_asset_class_corr.index) for elem in asset_class_names)), "missing asset corr row"
    assert (all(elem in list(df_asset_class_corr.columns) for elem in asset_class_names)), "missing asset corr column"
    assert (df_asset_class_corr.shape[0] == df_asset_class_corr.shape[1]), "corr is not symetric"
    
    df_asset_class_corr = (df_asset_class_corr.loc[asset_class_names][asset_class_names]).fillna(0)
    
    return df_asset_class_return, df_asset_class_stdev, df_asset_class_corr

def monte_carlo_simulation (dict_asset_class_weights, 
                            asset_class_names,
                            df_asset_class_return, 
                            df_asset_class_stdev,
                            df_asset_class_corr,
                            num_simulation,
                            VaR_levels,
                            df_illiquid_ratio):
    
    """
    conduct monte carlo simulation based on given distribution
    
    args:

        dict_asset_class_weights (dict of lists), 
        asset_class_names (list of str)
        df_asset_class_return (df): annualised asset class returns
        df_asset_class_stdev (df): annualised asset class stdev
        df_asset_class_corr (df): asset class return correlation
        num_simulation (int/float)
        VaR_levels (list): list of var levels
        df_illiquid_ratio (df)
        
    returns:
        
        df_var (df): portfolio ann return at var level
        df_asset_class_weights_at_var (df): asset class weights when portfolio ann return at var level
        df_avg (df): average portfolio ann return across simulations
        df_asset_class_weights_at_avg (df): asset class weights at average ann return
        df (df): simulated annualised asset class returns
    
    """
    
    dict_var = dict(zip(dict_asset_class_weights.keys(),[np.nan for x in dict_asset_class_weights.keys()]))
    dict_avg = dict(zip(dict_asset_class_weights.keys(),[np.nan for x in dict_asset_class_weights.keys()]))
    dict_var_asset_class_weights = dict(zip(dict_asset_class_weights.keys(),[np.nan for x in asset_class_names]))
    dict_avg_asset_class_weights = dict(zip(dict_asset_class_weights.keys(),[np.nan for x in asset_class_names]))
    
    stdev = np.diag(df_asset_class_stdev[df_asset_class_stdev.columns[0]].tolist())
    mean = df_asset_class_return[df_asset_class_return.columns[0]].tolist()
    corr = df_asset_class_corr.values
    
    covariance = (stdev.dot(corr)).dot(stdev)
    df = pd.DataFrame(np.random.multivariate_normal(mean, covariance, num_simulation),
                      columns=asset_class_names)
    
    df_var_all = pd.DataFrame()
    df_asset_class_weights_at_var_all = pd.DataFrame()
    df_avg_all = pd.DataFrame()
    df_asset_class_weights_at_avg_all = pd.DataFrame()
    
    for VaR_level in VaR_levels:   
        
        for port in dict_var.keys():
                    
            df_asset_class_weights = pd.Series(dict_asset_class_weights[port])
            df_asset_class_weights.index = asset_class_names
            df_temp = df.copy()
            df_temp['simulated port return'] = df_temp.mul(df_asset_class_weights, axis=1).sum(axis=1)                         
            df_temp['pct_rank(ascend)'] = df_temp['simulated port return'].rank(ascending=True, pct=True)
            df_temp = df_temp.sort_values(by = ['pct_rank(ascend)'])
            dict_var[port] = (df_temp[df_temp['pct_rank(ascend)'] <= VaR_level].iloc[-1])['simulated port return']
            dict_avg[port] = df_temp['simulated port return'].mean(axis = 0)
    
            df_asset_class_var_return = (df_temp[df_temp['pct_rank(ascend)'] <= VaR_level].iloc[-1])[asset_class_names]
            df_weights_var = (1+df_asset_class_var_return).mul(dict_asset_class_weights[port])
            df_weights_var_norm = df_weights_var/df_weights_var.sum()
            dict_var_asset_class_weights[port] = df_weights_var_norm
            
            df_asset_class_avg_return = (df_temp.mean(axis=0))[asset_class_names]
            df_weights_avg = (1+df_asset_class_avg_return).mul(dict_asset_class_weights[port])
            df_weights_avg_norm = df_weights_avg/df_weights_avg.sum()
            dict_avg_asset_class_weights[port] = df_weights_avg_norm          

        df_var = pd.DataFrame(dict_var, index = ['portfolio_VaR_at_%s'%VaR_level])
        df_var_all = df_var_all.append(df_var)
        
        df_asset_class_weights_at_var = pd.DataFrame(dict_var_asset_class_weights).T
        df_asset_class_weights_at_var['illiquid_ratio'] = (df_illiquid_ratio.set_index(['asset_class_names']).T*df_asset_class_weights_at_var).sum(axis=1)
        df_asset_class_weights_at_var['VaR'] = VaR_level
        df_asset_class_weights_at_var = df_asset_class_weights_at_var[['VaR','illiquid_ratio'] + asset_class_names]
        df_asset_class_weights_at_var_all = df_asset_class_weights_at_var_all.append(df_asset_class_weights_at_var)
        
        df_avg = pd.DataFrame(dict_avg, index = ['portfolio_Avg'])
        df_avg_all = df_avg_all.append(df_avg)
        
        df_asset_class_weights_at_avg = pd.DataFrame(dict_avg_asset_class_weights).T
        df_asset_class_weights_at_avg['illiquid_ratio'] = (df_illiquid_ratio.set_index(['asset_class_names']).T*df_asset_class_weights_at_avg).sum(axis=1)
        df_asset_class_weights_at_avg = df_asset_class_weights_at_avg[['illiquid_ratio'] + asset_class_names]
        df_asset_class_weights_at_avg_all = df_asset_class_weights_at_avg_all.append(df_asset_class_weights_at_avg)        
    
    return df_var_all, df_asset_class_weights_at_var_all, df_avg_all.drop_duplicates(), df_asset_class_weights_at_avg_all.drop_duplicates(), df

def get_historical_distribution (df_data_full,
                                 _period,
                                 asset_class_names,
                                 num_days_in_a_year):
    
    """
    calculate (non-compound) annualised return, stdev and corr based on given price/return data
    
    notes:
        
        cash is annual return in price data whereas PE is quarterly return
    
    args:
        
        df_data_full (df): assset class index daily prices
         _period (tuple of dates): select period
         asset_class_names (list of str): asset classes used in portfolios
         num_days_in_a_year (int/float): used to annualise daily returns and stdev
    
    returns:
        
        df_ann_return (df): annualised asset class returns
        df_ann_stdev (df): annualised asset class stdev
        df_corr (df): asset class daily return correlation
        
    """
    
    df_data = df_data_full.loc[_period[0] : _period[1]].fillna(method='ffill')
    df_data = df_data[asset_class_names]
            
    df_return = (df_data/df_data.shift(1) - 1)

    if 'Cash' in df_return.columns:

        df_return['Cash'] = df_data['Cash']/(100 * num_days_in_a_year)

    if 'PE' in df_return.columns:

        df_return['PE'] = df_data['PE']/(100 * num_days_in_a_year * 0.25)
        
    # return
    df_ann_return = df_return.mean(axis=0) * num_days_in_a_year
    df_ann_return = pd.DataFrame(df_ann_return).loc[asset_class_names]
    
    # stdev
    df_ann_stdev = df_return.std(axis=0) * num_days_in_a_year ** (0.5)
    df_ann_stdev = pd.DataFrame(df_ann_stdev).loc[asset_class_names]

    # corr
    df_corr = df_return.corr()
    
    return df_ann_return, df_ann_stdev, df_corr

def test_5_2 (  input_data_file_name,
                asset_class_names,
                dict_asset_class_weights,
                VaR_levels,
                num_simulation,
                list_periods,
                num_days_in_a_year,
                df_illiquid_ratio):
    
    """
    perform 5_2 test, stochastic simulation based on dstribution derived from historical data
    
    args:
        
        input_data_file_name (str): file name of the input asset class price data
        asset_class_names (list of str)
        dict_asset_class_weights (dict of lists)
        VaR_levels (list)
        num_simulation (int/float)
        list_periods (list of tuples)
        num_days_in_a_year (int/float)    
    
    returns:
        
        df_port_return_all_periods (df): table of portfolio return at var based on simulation
        df_asset_class_weights_all_periods(df): table of asset class weights when portfolio return at var
        list_df_simulation (list of df): simulated annulaised return from each period
        
    """
    
    df_data_full = pd.read_csv(input_data_file_name, index_col = 0, parse_dates = True, dayfirst = True)
    df_port_return_all_periods = pd.DataFrame()
    df_asset_class_weights_all_periods = pd.DataFrame()
    list_df_simulation = []
    
    for _period in list_periods:
        
        df_asset_class_return, df_asset_class_stdev, \
        df_asset_class_corr = get_historical_distribution ( df_data_full,
                                                            _period,
                                                            asset_class_names,
                                                            num_days_in_a_year)
            
        _,_,df_avg, df_asset_class_weights_at_avg, df_simulation= monte_carlo_simulation (dict_asset_class_weights, 
                                                                             asset_class_names,
                                                                             df_asset_class_return.fillna(0), 
                                                                             df_asset_class_stdev.fillna(0),
                                                                             df_asset_class_corr.fillna(0),
                                                                             num_simulation,
                                                                             VaR_levels,
                                                                             df_illiquid_ratio)

        df_avg['period'] = '{}:{}'.format(_period[0], _period[1])
        df_asset_class_weights_at_avg['period'] = '{}:{}'.format(_period[0], _period[1])

        df_port_return_all_periods = df_port_return_all_periods.append(df_avg)
        df_asset_class_weights_all_periods = df_asset_class_weights_all_periods.append(df_asset_class_weights_at_avg)
        list_df_simulation.append(df_simulation)
        
    df_port_return_all_periods.index.names = ['']        
    df_port_return_all_periods = df_port_return_all_periods.reset_index().set_index(['', 'period'])
    df_asset_class_weights_all_periods.index.names = ['portfolio']
    df_asset_class_weights_all_periods = df_asset_class_weights_all_periods.reset_index(). \
                                                set_index(['portfolio', 'period'])

    return df_port_return_all_periods, df_asset_class_weights_all_periods, list_df_simulation

def cum_compound(df_daily_return):
    
    """
    calculate compound return 
    """
    
    df_cum_compound = ((df_daily_return + 1).cumprod() - 1)
    
    return df_cum_compound

def test_5_3(input_data_file_name, 
             asset_class_names,
             dict_asset_class_weights,
             list_periods,
             num_days_in_a_year,
             df_illiquid_ratio):
    
    """
    perform 5_3 test, portfolio return and asset class weights during peak to trough drawdown.
    similar to 5_2 test except this test is not based on stochastic simulation
    
    args:
        
        input_data_file_name (str): file name of the input asset class price data
        asset_class_names (list of str)
        dict_asset_class_weights (dict of lists)
        list_periods (list of tuples)
        num_days_in_a_year (int/float)    
        df_illiquid_ratio (df)
        
    returns:
        
        df_min_port_return_all_portfolios (df): table of portfolio peak-trough return
        df_min_port_return_date_all_portfolios (df): table of portfolio peak-trough dates
        df_asset_class_weights_all_periods (df): table of asset class weights when portfolio reach peak-trough return

    """
    
    df_data_full = pd.read_csv(input_data_file_name, index_col = 0, parse_dates = True, dayfirst = True)
    df_asset_class_weights_all_periods = pd.DataFrame()
    df_weights = pd.DataFrame(index = asset_class_names, data = dict_asset_class_weights)

    df_min_port_return_all_portfolios = pd.DataFrame(index = ['{}:{}'.format(x[0],x[1]) for x in list_periods],
                                                     columns = dict_asset_class_weights.keys(),
                                                     data = np.nan)

    df_min_port_return_date_all_portfolios = pd.DataFrame(index = ['{}:{}'.format(x[0],x[1]) for x in list_periods],
                                                     columns = dict_asset_class_weights.keys(),
                                                     data = '')
    for _period in list_periods:

        df_data = df_data_full.loc[_period[0] : _period[1]].fillna(method='ffill')
        df_data = df_data[asset_class_names]
        df_return = (df_data/df_data.shift(1) - 1)

        if 'Cash' in df_return.columns:

            df_return['Cash'] = df_data['Cash']/(100 * num_days_in_a_year)
            
        if 'PE' in df_return.columns:

            df_return['PE'] = df_data['PE']/(100 * num_days_in_a_year * 0.25)
              
        for _port in dict_asset_class_weights.keys():
            
            print ('5.3', _port)
            df_return_cum = cum_compound(df_return)
            df_port_return_cum = df_return_cum.mul(df_weights[_port], axis=1).sum(axis=1)
            df_port_return_cum_max = df_port_return_cum.cummax()
            df_dd = df_port_return_cum - df_port_return_cum_max
            bottom_port_return = df_dd.min()
            bottom_port_return_date = df_dd.idxmin().strftime("%Y-%m-%d")
            peak_port_return_date = df_dd[df_dd == 0].loc[:bottom_port_return_date].index[-1].strftime("%Y-%m-%d")
            
            df_min_port_return_all_portfolios.loc['{}:{}'.format(_period[0],_period[1])][_port] = bottom_port_return
            
            peak_trough_dates = "%s:%s"%(peak_port_return_date, bottom_port_return_date)
            df_min_port_return_date_all_portfolios.loc['{}:{}'.format(_period[0],_period[1])][_port] = peak_trough_dates

            df_weights_max_dd = df_weights[_port] * (1 + \
                                         cum_compound(df_return.loc[peak_port_return_date:bottom_port_return_date]).iloc[-1])
            df_weights_max_dd_norm = df_weights_max_dd / df_weights_max_dd.sum()
            df_asset_class_weights_max_dd = pd.DataFrame(df_weights_max_dd_norm).T

            df_asset_class_weights_max_dd.index.names = ['portfolio']
            df_asset_class_weights_max_dd.index = [_port]
            df_asset_class_weights_max_dd['illiquid_ratio'] = ((df_illiquid_ratio.set_index(['asset_class_names'])[_port])*df_asset_class_weights_max_dd.iloc[0].fillna(0)).sum()
            df_asset_class_weights_max_dd['period'] = '{}:{}'.format(_period[0],_period[1])
            
            df_asset_class_weights_all_periods = df_asset_class_weights_all_periods.append(df_asset_class_weights_max_dd)

    df_asset_class_weights_all_periods.index.names = ['portfolio']
    df_asset_class_weights_all_periods = df_asset_class_weights_all_periods.reset_index() \
                                                                           .set_index(['portfolio', 'period'])[asset_class_names]
    
    return df_min_port_return_all_portfolios, df_min_port_return_date_all_portfolios, df_asset_class_weights_all_periods

def test_7 (df_5_1_asset_class_weights_at_var, df_config_saa, redemption_levels, df_illiquid_ratio, VaR_levels):

    """
    test combination of poor portfolio return and redemption
    
    args:
        
        df_5_1_asset_class_weights_at_var (df): result from test 5_1, asset class weights when portfolio return at var
        df_config_saa (df): df of saa including illiquid ratio for each asset class
        redemption_levels (list of float): list of redemption ratios
        df_illiquid_ratio (df)
        VaR_levels (list)
        
    returns:
        
        df_asset_class_redemp_all (df): asset class weights
    
    """
    
    df_asset_class_weights_at_var = df_5_1_asset_class_weights_at_var.drop(['illiquid_ratio'],axis=1)    
    df_asset_class_redemp_all = pd.DataFrame()
    df_illiquid_ratio = df_illiquid_ratio.set_index(['asset_class_names'])
    df_config_saa = df_config_saa.set_index(['asset_class_names'])
    
    for redemption_level in redemption_levels:

        for portfolio in df_config_saa.columns:
            
            for VaR_level in VaR_levels:
            
                df_asset_class_weights_at_var_temp = df_asset_class_weights_at_var[df_asset_class_weights_at_var['VaR']==VaR_level].drop(['VaR'],axis=1).ix[portfolio]
                liquid_total_asset = (df_asset_class_weights_at_var_temp*(1-df_illiquid_ratio[portfolio])).sum()
                
                df_asset_class_weights_redem = df_asset_class_weights_at_var_temp * (1-df_illiquid_ratio[portfolio]) * (1 - redemption_level / liquid_total_asset) + df_asset_class_weights_at_var_temp*df_illiquid_ratio[portfolio]
                df_asset_class_weights_redem = df_asset_class_weights_redem.to_frame().T
                df_asset_class_weights_redem_norm = (df_asset_class_weights_redem. \
                    div(df_asset_class_weights_redem.sum(axis=1),axis=0))
                
                df_asset_class_weights_redem_norm['illiquid ratio'] = (df_asset_class_weights_redem_norm.ix[portfolio]*df_illiquid_ratio[portfolio]).sum()
                df_asset_class_weights_redem_norm.index.names = ['portfolio']
                df_asset_class_weights_redem_norm['redemption ratio'] = redemption_level
                df_asset_class_weights_redem_norm['VaR'] = VaR_level
                
                df_asset_class_redemp_all = df_asset_class_redemp_all.append(df_asset_class_weights_redem_norm.reset_index())

    df_asset_class_redemp_all= df_asset_class_redemp_all.set_index(['portfolio', 'VaR', 'redemption ratio'])
    
    return df_asset_class_redemp_all

def save_result(path_output,
                df_5_1_var,
                df_5_1_asset_class_weights_at_var,
                df_5_1_simulation,
                df_5_2_avg,
                df_5_2_asset_class_weights_at_avg,
                list_5_2_simulation,
                df_min_port_return,
                df_min_port_return_date,
                df_asset_class_weights,
                df_asset_class_redemp_all):
    
    """
    save result to excel
    
    args:
        
        save_all_result (str, Yes or No): save result to file or not
        path_output (str): path of output
        
    """

    with pd.ExcelWriter(path_output) as writer: 

        df_5_1_var.to_excel(writer, sheet_name='5_1_var')
        df_5_1_asset_class_weights_at_var.to_excel(writer, sheet_name='5_1_asset_class_weights_at_var')
        df_5_1_simulation.to_excel(writer, sheet_name='5_1_simulation')
        df_5_2_avg.to_excel(writer, sheet_name='5_2_avg')
        df_5_2_asset_class_weights_at_avg.to_excel(writer, sheet_name='5_2_asset_class_weights_at_avg')
        
        for i, simulation in enumerate(list_5_2_simulation):

            simulation.to_excel(writer, sheet_name='simulation_period_%s'%(i+1))

        df_min_port_return.to_excel(writer, sheet_name='5_3_min_port_return')
        df_min_port_return_date.to_excel(writer, sheet_name='5_3_min_port_return_dates')
        df_asset_class_weights.to_excel(writer, sheet_name='5_3_asset_class_weights')
        df_asset_class_redemp_all.to_excel(writer, sheet_name='7_redemption')
                                    
    return

def merge(list1, list2): 
      
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))] 
    
    return merged_list 

def main():
    
    """
    main work function
    
    """

    """
    initialising
    """
    ctime = time.strftime("%Y%m%d%H%M")
    username = getpass.getuser()
    logger = logging.getLogger(__name__)
    log_file = os.getcwd() + '\\log\\stress_test_{}_{}.log'.format(username, ctime)
    start_logger(log_file)

    """
    UI
    """    
    sg.theme('DarkAmber')
    
    layout = [ [sg.Text('Configuration File',size = (17,1)), sg.InputText(), sg.FileBrowse()],
               [sg.Text('Output Directory',size = (17,1)), sg.InputText(), sg.FolderBrowse()],
               [sg.Button('Ok'), sg.Button('Cancel')]]
    
    window = sg.Window('Carnbrea Stress Test', layout, default_element_size = (50, 40), grab_anywhere = False)
    
    event, values = window.read()
    window.close()
    
    if event in (None, 'Cancel'):
        
        logger.info('process cancelled by user')
        sys.exit("exit")
        
    """
    main process
    """
    logger.info("process starts\n")
    
    path_config = str(values[0]).strip().replace('/','\\')
    dir_output = (str(values[1]).strip()).replace('/','\\')
    if dir_output[-1] != '\\': 
        dir_output = dir_output + '\\'
    
    logger.info("path_config: %s"%path_config)    
    logger.info("dir_output: %s"%dir_output)   
    logger.info("reading config")
    
    # read config 
    df_config_basic = pd.read_excel(path_config, sheet_name = "basic_parameters").dropna(how='all',axis=0).dropna(how='all',axis=1)
    df_config_saa = pd.read_excel(path_config, sheet_name = "asset_allocation").dropna(how='all',axis=0).dropna(how='all',axis=1)
    df_config_events = pd.read_excel(path_config, sheet_name = "hist_events").dropna(how='all',axis=0).dropna(how='all',axis=1)
    df_illiquid_ratio = pd.read_excel(path_config, sheet_name = "illiquid_ratio").dropna(how='all',axis=0).dropna(how='all',axis=1)

    dict_config_basic = dict(zip(df_config_basic['parameter'],df_config_basic['value']))
    
    saa_names = [x for x in df_config_saa.columns if 'asset_class_names' not in x  and 'illiquid' not in x]
    saa_allocations = [df_config_saa[x].tolist() for x in saa_names]
    dict_asset_class_weights = dict(zip(saa_names, saa_allocations))
    
    asset_class_names = df_config_saa['asset_class_names'].str.strip().tolist()
    VaR_levels = [float(x) for x in str(dict_config_basic['VaR_level']).split(',')]
    num_simulation = int(dict_config_basic['num_simulation'])
    num_days_in_a_year = float(dict_config_basic['num_days_in_a_year'])
    redemption_levels = [float(x) for x in str(dict_config_basic['redemption_levels']).split(',')]
    list_hist_events = merge(df_config_events['start_date'].tolist(), df_config_events['end_date'].tolist())
    
    # validate input
    assert df_config_saa.shape == df_illiquid_ratio.shape, "illiquid_ratio shape doesnt match asset allocation"
    
    
    # 5.1
    logger.info("running test 5.1")
    logger.info("""file_asset_class_assumption: {}
    asset_class_names: {}""".format(dict_config_basic['asset_assumption_file'], asset_class_names))    
    
    try:
        df_asset_class_return, df_asset_class_stdev, df_asset_class_corr = read_asset_class_file(file_asset_class_assumption = dict_config_basic['asset_assumption_file'], 
                                                                                                 asset_class_names = asset_class_names )
    
        df_5_1_var, df_5_1_asset_class_weights_at_var, _, \
        _, df_5_1_simulation = monte_carlo_simulation (  dict_asset_class_weights, 
                                                         asset_class_names,
                                                         df_asset_class_return, 
                                                         df_asset_class_stdev,
                                                         df_asset_class_corr,
                                                         num_simulation,
                                                         VaR_levels,
                                                         df_illiquid_ratio)
        
        pass_5_1 = True
        
    except Exception as error:
        logger.error("error running test 5.1, %s"%error)
        pass_5_1 = False
        
    # 5.2
    logger.info("""running test 5.2
    input data: {},
    asset_class_names: {}
    dict_asset_class_weights: {}
    VaR_level: {}
    num_simulation: {}
    list_hist_events: {}
    num_days_in_a_year: {}""".format(dict_config_basic['asset_class_price'],asset_class_names,dict_asset_class_weights,
                                     VaR_levels,num_simulation,list_hist_events,num_days_in_a_year))
    
    try:
        df_5_2_avg, df_5_2_asset_class_weights_at_avg, list_5_2_simulation = test_5_2(  dict_config_basic['asset_class_price'],
                                                                                        asset_class_names,
                                                                                        dict_asset_class_weights,
                                                                                        VaR_levels,
                                                                                        num_simulation,
                                                                                        list_hist_events,
                                                                                        num_days_in_a_year,
                                                                                        df_illiquid_ratio)
        pass_5_2 = True
        
    except Exception as error:
        logger.error("error running test 5.2, %s"%error)   
        pass_5_2 = False
        
    # 5.3
    logger.info("""running test 5.3
    input data: {},
    asset_class_names: {}
    dict_asset_class_weights: {}
    list_hist_events: {}
    num_days_in_a_year: {}""".format(dict_config_basic['asset_class_price'],asset_class_names,
                                     dict_asset_class_weights,list_hist_events,num_days_in_a_year))
    
    try:
        df_min_port_return, df_min_port_return_date, df_asset_class_weights = test_5_3( dict_config_basic['asset_class_price'], 
                                                                                        asset_class_names,
                                                                                        dict_asset_class_weights,
                                                                                        list_hist_events,
                                                                                        num_days_in_a_year,
                                                                                        df_illiquid_ratio)
        pass_5_3 = True
        
    except Exception as error:
        logger.error("error running test 5.3, %s"%error)   
        pass_5_3 = False
        
    # 7
    logger.info("""running test 7
    redemption_levels: {}        
    """.format(redemption_levels))
    
    try:
        df_asset_class_redemp_all = test_7(df_5_1_asset_class_weights_at_var,
                                           df_config_saa,
                                           redemption_levels,
                                           df_illiquid_ratio,
                                           VaR_levels)
        pass_7 = True
        
    except Exception as error:
        logger.error("error running test 7, %s"%error)   
        pass_7 = False
        
    # save result
    logger.info("saving result")
    path_output = dir_output + 'stress_test_result_{}_{}.xlsx'.format(username, ctime)
    
    with pd.ExcelWriter(path_output) as writer: 
        
        if pass_5_1:
            df_5_1_var.to_excel(writer, sheet_name='5_1_var')
            df_5_1_asset_class_weights_at_var.to_excel(writer, sheet_name='5_1_asset_class_weights_at_var')
            df_5_1_simulation.to_excel(writer, sheet_name='5_1_simulation')
            
        if pass_5_2:
            df_5_2_avg.to_excel(writer, sheet_name='5_2_avg')
            df_5_2_asset_class_weights_at_avg.to_excel(writer, sheet_name='5_2_asset_class_weights_at_avg')
        
            for i, simulation in enumerate(list_5_2_simulation):
    
                simulation.to_excel(writer, sheet_name='simulation_period_%s'%(i+1))
                
        if pass_5_3:
            df_min_port_return.to_excel(writer, sheet_name='5_3_min_port_return')
            df_min_port_return_date.to_excel(writer, sheet_name='5_3_min_port_return_dates')
            df_asset_class_weights.to_excel(writer, sheet_name='5_3_asset_class_weights')
            
        if pass_7:
            df_asset_class_redemp_all.to_excel(writer, sheet_name='7_redemption')
    
    logger.info("process finsihed")
    
    return values

if __name__=="__main__":
        
    pass
    values = main()

