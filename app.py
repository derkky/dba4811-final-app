import numpy as np      
import pandas as pd   
import datetime as dt
import math

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestRegressor

from sklearn.multioutput import MultiOutputRegressor

import sklearn.metrics
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_squared_error
from sklearn import tree
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import gradio as gr

pd.set_option('display.max_columns', None)

from numpy.random import seed
seed(1)

import warnings
warnings.filterwarnings('ignore')


## load dataset and models
raw_df = pd.read_csv('bike_sharing.csv').drop(columns = ['instant'])
tree_model = pickle.load(open('rf_model.sav', 'rb'))

# helper functions for maching learning step

def unnormalise_data(raw_df):

    df = raw_df.copy()
    df['temp'] = df['temp'].apply(lambda x: x*(39-(-8)) + (-8))
    df['atemp'] = df['atemp'].apply(lambda x: x*(50-(-16)) + (-16))
    df['hum'] = df['hum'].apply(lambda x: x*100)
    df['windspeed'] = df['windspeed'].apply(lambda x: x*67)
    df['yr'] = df['yr'].apply(lambda x: 2012 if x==1 else 2011)
    df['dteday_hr'] = pd.to_datetime(df['dteday'] + "-" + df['hr'].apply(lambda x: str(x)), format='%Y-%m-%d-%H')
    df['dteday'] = pd.to_datetime(df['dteday'], format='%Y-%m-%d')

    return df

def convert_categorical(old_df):
    df = old_df.copy()
    
    season_list = ['Spring', 'Summer', 'Fall', 'Winter']
    season_keys = np.arange(1,5)
    season_map = {season_keys[i]: season_list[i] for i in range(len(season_keys))}
    df['season'] = df['season'].transform(lambda x: season_map[x])
    
    weekday_list = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    weekday_keys = np.arange(0,7)
    weekday_map = {weekday_keys[i]: weekday_list[i] for i in range(len(weekday_keys))}
    df['weekday'] = df['weekday'].transform(lambda x: weekday_map[x])
    

    weathersit_map = {
    1: 'Clear / Partly Cloudy',    #Clear, Few clouds, Partly cloudy, Partly cloudy'
    2: 'Mist / Cloudy',            # 'Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist',
    3: 'Light Rain / Light Snow',    #'Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds',
    4: 'Heavy Rain / Ice Pallets / Fog' # 'Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog'
    }
    df['weathersit'] = df['weathersit'].transform(lambda x: weathersit_map[x])
    

    month_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 
                  'August', 'September', 'October', 'November', 'December']
    month_keys = np.arange(1,13)
    month_map = {month_keys[i]: month_list[i] for i in range(len(month_keys))}
    df['mnth'] = df['mnth'].transform(lambda x: month_map[x])
    
    return df


def data_split(df):
    
    df['dteday'] = pd.to_datetime(df['dteday'], format='%Y-%m-%d')

    train = df[df['dteday'] < '2012-07-01'].drop(columns = ['dteday'])
    val = df[(df['dteday'] >= '2012-07-01') & (df['dteday'] < '2012-10-01')].drop(columns = ['dteday'])
    test = df[df['dteday'] >= '2012-10-01'].drop(columns = ['dteday'])

    y_train = train[['casual', 'registered']].to_numpy()
    y_val = val[['casual', 'registered']].to_numpy()
    y_test = test[['casual', 'registered']].to_numpy()

    X_train = train.drop(columns = ['casual', 'registered'])
    X_val = val.drop(columns = ['casual', 'registered'])
    X_test = test.drop(columns = ['casual', 'registered'])
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# function to get lagged variables

def get_lag_var(old_df):
    df = old_df.copy()
    
    lagged_variables = ['temp', 'atemp', 'hum', 'windspeed', 'weathersit']
    
    for var in lagged_variables:
        column_name = str(var) + "_lag"
        df[column_name] = df[var].shift(1)
    
    df.drop(columns = lagged_variables, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True)
    
    df.drop(columns = ['cnt'], inplace=True)
        
    return df    

def preprocess_tree(raw_df):
    
    df = raw_df.copy()
    df = unnormalise_data(df).drop(columns = ['dteday_hr'])
    df = convert_categorical(df)
    df = get_lag_var(df)
    df = pd.get_dummies(df, columns=['season', 'yr', 'mnth', 'weekday','weathersit_lag'], drop_first=False)
    X_train, X_val, X_test, y_train, y_val, y_test = data_split(df)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

X_train_tree, X_val_tree, X_test_tree, y_train_tree, y_val_tree, y_test_tree = preprocess_tree(raw_df)

# helper function to get optimal price based on predicted demand

def get_pct_chg_demand_casual(pct_chg_price):
    pct_chg_demand = -1.5639 * math.atan(pct_chg_price)
    return pct_chg_demand

def get_pct_chg_demand_registered(pct_chg_price):
    pct_chg_demand = -3.4488 * math.atan(pct_chg_price)
    return pct_chg_demand

def get_optimal_pct_chg_price_casual(pred_demand, supply):
    max_revenue_chg = 0
    best_pct_chg_price = 0
    
    if pred_demand > supply:
        best_q_demand = supply
        
    else:
        best_q_demand = pred_demand

    for pct_chg_price in np.arange(-0.99, 1.01, 0.01):
        pct_chg_demand = get_pct_chg_demand_casual(pct_chg_price)
        q_demand = (pct_chg_demand + 1) * pred_demand

        # If the quantity demanded at the new price point exceeds the current supply or is negative, ignore and 
        # check next price point
        if  q_demand > supply or q_demand < 0:
            continue

        if pred_demand > supply:        
            old_revenue = supply * 1

        else:
            old_revenue = pred_demand * 1

        new_revenue = (1+pct_chg_demand)*pred_demand * (1+pct_chg_price)
        revenue_chg = (new_revenue - old_revenue) / old_revenue


        if revenue_chg > max_revenue_chg:
            max_revenue_chg = revenue_chg
            best_pct_chg_price = pct_chg_price
            best_q_demand =  (pct_chg_demand + 1) * pred_demand
    
    return(max_revenue_chg, best_pct_chg_price, round(best_q_demand))

def get_optimal_pct_chg_price_registered(pred_demand, supply):
    max_revenue_chg = 0
    best_pct_chg_price = 0
    
    if pred_demand > supply:
        best_q_demand = supply
        
    else:
        best_q_demand = pred_demand

    for pct_chg_price in np.arange(-0.99, 1.01, 0.01):
        pct_chg_demand = get_pct_chg_demand_registered(pct_chg_price)
        q_demand = (pct_chg_demand + 1) * pred_demand

        # If the quantity demanded at the new price point exceeds the current supply or is negative, ignore and 
        # check next price point
        if  q_demand > supply or q_demand < 0:
            continue

        if pred_demand > supply:        
            old_revenue = supply * 1

        else:
            old_revenue = pred_demand * 1

        new_revenue = (1+pct_chg_demand)*pred_demand * (1+pct_chg_price)
        revenue_chg = (new_revenue - old_revenue) / old_revenue

        if revenue_chg > max_revenue_chg:
            max_revenue_chg = revenue_chg
            best_pct_chg_price = pct_chg_price
            best_q_demand =  (pct_chg_demand + 1) * pred_demand

    return(max_revenue_chg, best_pct_chg_price, round(best_q_demand))

def get_optimal_price_chg(pred_casual, pred_registered, total_supply):
    supply_registered = round(total_supply * (pred_registered/(pred_registered + pred_casual)))
    res_registered = get_optimal_pct_chg_price_registered(pred_registered, supply_registered)
    
    supply_casual = total_supply - supply_registered
    res_casual = get_optimal_pct_chg_price_casual(pred_casual, supply_casual)
    
    return (res_registered, res_casual)

fixed_total_supply = 977

def model_pred(hour, season, year, month, holiday, day_of_week, working_day, previous_weather_situation,
               previous_temperature, previous_feeling_temperature, previous_humidity, previous_windspeed):
    
    # map 'yes' and 'no' back to 0 and 1

    holiday_mapped =  1 if holiday is 'Yes' else 0
    working_day_mapped = 1 if working_day is 'Yes' else 0

    # create dictionary and dataframe of mapping

    dict_variable = {'hr': hour,
                     'season': season,
                     'yr': year,
                     'mnth': month,
                     'holiday': holiday_mapped,
                     'weekday': day_of_week,
                     'workingday': working_day_mapped,
                     'weathersit_lag': previous_weather_situation,
                     'temp_lag': previous_temperature,
                     'atemp_lag': previous_feeling_temperature,
                     'hum_lag': previous_humidity, 
                     'windspeed_lag': previous_windspeed}

    X_input = pd.DataFrame(dict_variable, index=[0])

    # prepare dataframe in right format for prediction

    X_input_dummy = pd.get_dummies(X_input, columns=['season', 'yr', 'mnth', 'weekday','weathersit_lag'], drop_first=False)
    X_tree_app = X_train_tree.copy()
    combined_df = pd.concat([X_tree_app, X_input_dummy]).fillna(0)
    X_values = combined_df.iloc[-1:]

    # predict using Random Forest
    preds = tree_model.predict(X_values)

    predicted_casual = round(preds[0][0])
    predicted_registered = round(preds[0][1])
    
    df_return = pd.DataFrame()
    
    df_return['User Type'] =['Casual', 'Registered']
    df_return['Original Demand (Predicted Users)'] = [predicted_casual, predicted_registered]
    
    # get best optimal price 
    res_registered, res_casual = get_optimal_price_chg(predicted_casual, predicted_registered, fixed_total_supply)
    
    casual_revenue, casual_price, casual_best_demand = res_casual
    registered_revenue, registered_price, registered_best_demand = res_registered
    
    df_return['Optimal Change in Price'] = [casual_price, registered_price]
    df_return['New Demand (given optimal price)'] = [casual_best_demand, registered_best_demand]
    df_return['Percentage Increase in Revenue'] = [casual_revenue, registered_revenue]
    
    df_return['Optimal Change in Price'] = pd.Series(["{0:+.2f}%".format(val * 100) for val in df_return['Optimal Change in Price']], index = df_return.index)
    df_return['Percentage Increase in Revenue'] = pd.Series(["{0:+.2f}%".format(val * 100) for val in df_return['Percentage Increase in Revenue']], index = df_return.index)
    
    return df_return

iface = gr.Interface(
    model_pred,
    [
        gr.inputs.Slider(0,23,1),
        gr.inputs.Radio(['Spring', 'Summer', 'Fall', 'Winter'], type='value'),
         gr.inputs.Radio(['2011', '2012'], type='value'),
         gr.inputs.Radio(['January', 'February', 'March', 'April', 'May', 'June', 'July', 
                  'August', 'September', 'October', 'November', 'December'], type='value'),
        gr.inputs.Radio(["Yes", "No"], type="value"),
        gr.inputs.Radio(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], type='value'),
        gr.inputs.Radio(["Yes", "No"], type="value"),
        gr.inputs.Radio(["Clear / Partly Cloudy", "Mist / Cloudy", "Light Rain / Light Snow", "Heavy Rain / Ice Pallets / Fog"], type="value"),
        gr.inputs.Slider(-16, 40),
        gr.inputs.Slider(-16, 50),
        gr.inputs.Slider(0, 100),
        gr.inputs.Slider(0, 60),
    ],
    "dataframe",
    title = "Capital Bike Share Dynamic Pricing Model",
    description = "Input day/time and current hour's weather information to get next hour's predicted demand and optimal pricing strategy"
)

iface.launch()