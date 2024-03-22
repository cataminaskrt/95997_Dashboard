# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 00:09:22 2024

@author: Usuario
"""
#Libraries necessary for the implementation of the dashboard
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import pickle
from sklearn import  metrics
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AutoReg
from dash.exceptions import PreventUpdate

#CSS style
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

white_text_style = {'color': 'white'}

#Loading the Raw Data
df_total = pd.read_csv("file_total.csv", index_col=0, parse_dates=True)
columns = df_total.columns.tolist()
start_date = df_total.index.min()
end_date = df_total.index.max()

# Cut-off date for the 2019 set
testing_cutoff_date = '2019-01-01'

#Splitting the Raw Data between training and testing data
training_data = df_total.loc[df_total.index < testing_cutoff_date] #dataset with values from 2017 and 2018
testing_data = df_total.loc[df_total.index >= testing_cutoff_date] #dataset with values from 2019

#Cleaning the training set
training_data = training_data.dropna() 

#Creating and cleaning a separate dataset for meteorological data in 2017,2018
df_meteo = training_data.copy()
df_meteo = df_meteo.drop("Power_kW", axis=1)

#Loading the real values for 2019
df_real = pd.read_csv('data_2019.csv')
y=df_real['Central (kWh)'].values

#Creating a separate dataset just with meteorological data for 2019
df_meteo_2019 = testing_data.drop('Power_kW', axis=1)

#Cleaning the meteo dataset for 2019
dates_to_drop = df_meteo_2019[~df_meteo_2019.index.isin(df_real['Date'])].index
df_meteo_2019 = df_meteo_2019.drop(dates_to_drop)

#Cleaning and adding a new dataset to visualize raw data of 2019
df_real_aux = df_real.set_index('Date')
dates_to_drop = df_real_aux[~df_real_aux.index.isin(df_real_aux.index)].index
df_real_2019 = df_real_aux.drop(dates_to_drop)

#Changing the names of the columns in all datasets
df_total = df_total.rename(columns={
    'Power_kW': 'Power (kW)',
    'HR': 'Relative Humidity',
    'temp_C': 'Temperature (ºC)',
    'windSpeed_m/s': 'Wind Speed (m/s)',
    'windGust_m/s': 'Wind Gust (m/s)',
    'pres_mbar': 'Pressure (mbar)',
    'solarRad_W/m2': 'Solar Radiation (W/m2)',
    'rain_mm/h': 'Rain (mm/h)',
    'rain_day': 'Rain day',
})

df_meteo = df_meteo.rename(columns={
    'HR': 'Relative Humidity',
    'temp_C': 'Temperature (ºC)',
    'windSpeed_m/s': 'Wind Speed (m/s)',
    'windGust_m/s': 'Wind Gust (m/s)',
    'pres_mbar': 'Pressure (mbar)',
    'solarRad_W/m2': 'Solar Radiation (W/m2)',
    'rain_mm/h': 'Rain (mm/h)',
    'rain_day': 'Rain day',
})

df_meteo_2019 = df_meteo_2019.rename(columns={
    'HR': 'Relative Humidity',
    'temp_C': 'Temperature (ºC)',
    'windSpeed_m/s': 'Wind Speed (m/s)',
    'windGust_m/s': 'Wind Gust (m/s)',
    'pres_mbar': 'Pressure (mbar)',
    'solarRad_W/m2': 'Solar Radiation (W/m2)',
    'rain_mm/h': 'Rain (mm/h)',
    'rain_day': 'Rain day',
})

training_data = training_data.rename(columns={
    'Power_kW': 'Power (kW)',
    'temp_C': 'Temperature (ºC)',
    'HR': 'Relative Humidity',
    'windSpeed_m/s': 'Wind Speed (m/s)',
    'windGust_m/s': 'Wind Gust (m/s)',
    'pres_mbar': 'Pressure (mbar)',
    'solarRad_W/m2': 'Solar Radiation (W/m2)',
    'rain_mm/h': 'Rain (mm/h)',
    'rain_day': 'Rain day',
})

df_real = df_real.rename(columns={
    'temp_C': 'Temperature (ºC)',
    'HR': 'Relative Humidity',
    'windSpeed_m/s': 'Wind Speed (m/s)',
    'windGust_m/s': 'Wind Gust (m/s)',
    'pres_mbar': 'Pressure (mbar)',
    'solarRad_W/m2': 'Solar Radiation (W/m2)',
    'rain_mm/h': 'Rain (mm/h)',
    'rain_day': 'Rain day',
})

df_real_2019 = df_real_2019.rename(columns={
    'temp_C': 'Temperature (ºC)',
    'HR': 'Relative Humidity',
    'windSpeed_m/s': 'Wind Speed (m/s)',
    'windGust_m/s': 'Wind Gust (m/s)',
    'pres_mbar': 'Pressure (mbar)',
    'solarRad_W/m2': 'Solar Radiation (W/m2)',
    'rain_mm/h': 'Rain (mm/h)',
    'rain_day': 'Rain day',
})

#Initializing the variables
X = None
Y = None

X_train = None
X_test = None
y_train = None
y_test = None

X_2019 = None
y_pred2019 = None

#By default, plot Central (kWh) vs. Date
fig2 = px.line(df_real, x='Date', y='Central (kWh)')

#Auxiliary functions
def generate_table(dataframe, max_rows=10):
    # Apply some CSS styles to the table
    table_style = {
        'borderCollapse': 'collapse',
        'borderSpacing': '0',
        'width': '100%',
        'border': '1px solid #ddd',
        'fontFamily': 'Arial, sans-serif',
        'fontSize': '14px'
    }
    
    th_style = {
        'border': '1px solid #ddd',
        'padding': '8px',
        'textAlign': 'left',
        'backgroundColor': '#f2f2f2',
        'fontWeight': 'bold',
        'color': '#333'
    }
    
    td_style = {
        'border': '1px solid #ddd',
        'padding': '8px',
        'textAlign': 'left'
    }
    
    return html.Table(
        # Apply the table style
        style=table_style,
        children=[
            # Add the table header
            html.Thead(
                html.Tr([
                    *[html.Th(col, style=th_style) for col in dataframe.columns]
                ])
            ),
            # Add the table body
            html.Tbody([
                html.Tr([
                    *[html.Td(dataframe.iloc[i][col], style=td_style) for col in dataframe.columns]
                ])
                for i in range(min(len(dataframe), max_rows))
            ])
        ]
    )

def generate_graph(df, columns, start_date, end_date):
    filtered_df = df.loc[start_date:end_date, columns]
    
    # Define a list to hold the y-axis configurations
    y_axis_config = []
    
    # Loop through each column and define a new y-axis configuration
    for i, column in enumerate(columns):
        y_axis_config.append({'title': column, 'overlaying': 'y', 'side': 'right', 'position': i * 0.1})
    
    # Define the data and layout of the figure
    data = [go.Scatter(x=filtered_df.index, y=filtered_df[column], name=column) for column in filtered_df.columns]
    layout = go.Layout(title=', '.join(columns), xaxis_title='Date')
    
    # Update the layout to include the y-axis configurations
    layout.update({'yaxis{}'.format(i + 1): y_axis_config[i] for i in range(len(y_axis_config))})
    
    # Create the figure with the data and layout
    fig = go.Figure(data=data, layout=layout)
    
    return fig


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

#Defining the layout of the dashboard, with the respective tabs and their descriptions
app.layout = html.Div(style={'backgroundColor': 'white'}, children=[
    html.H1('IST Central Building Forecast Energy'),
    html.Div(id='df_total', children=df_total.to_json(orient='split'), style={'display': 'none'}),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Raw Data 2017/2018', value='tab-1', children=[
            html.Div([
                html.H2("Raw Data 2017/2018"),
                html.P('Check graphically the raw data of Central Building in IST for the years 2017 and 2018. Select as many features as you want and adjust the date range as needed.'),
                dcc.Dropdown(
                    id='column-dropdown',
                    options=[{'label': i, 'value': i} for i in training_data.columns],
                    value=[training_data.columns[0]],
                    multi=True
                ),
                dcc.DatePickerRange(
                    id='date-picker',
                    min_date_allowed=training_data.index.min(),
                    max_date_allowed=training_data.index.max(),
                    start_date=training_data.index.min(),
                    end_date=training_data.index.max()
                ),
                dcc.Graph(id='graph'),
            ])
        ]),
        dcc.Tab(label='Raw Data 2019', value='tab-2', children=[
            html.Div([
                html.H2("Raw Data 2019"),
                html.P('Check graphically the raw data of Central Building in IST for the year 2019. Select the variables you want to plot and adjust the date range as needed.'),
                dcc.Dropdown(
                id='column-dropdown-2019',
                options=[{'label': i, 'value': i} for i in df_real_2019.columns],
                value=[df_real_2019.columns[0]],
                multi=True
                ),
                dcc.DatePickerRange(
                id='date-picker-2019',
                min_date_allowed=df_real_2019.index.min(),
                max_date_allowed=df_real_2019.index.max(),
                start_date=df_real_2019.index.min(),
                end_date=df_real_2019.index.max()
                ),
                dcc.Graph(id='graph-2019'),
            ])
        ]),
        dcc.Tab(label='Exploratory Data Analysis', value='tab-3', children=[
            html.Div([
                html.H2("Exploratory Data Analysis"),
                html.P('Here you have two types of graphical analysis to visualize your data. The first option is a scatter plot where you can select two features and check the relationship between them, which can be a powerful tool for feature selection. The second option is a box plot where you can select only one feature and see the distribution of their values, which is a powerful tool to check for possible outliers.'),
                dcc.Dropdown(
                    id='feature1',
                    options=[{'label': col, 'value': col} for col in training_data.columns],
                    value=training_data.columns[0]
                ),
                dcc.Dropdown(
                    id='feature2',
                    options=[{'label': col, 'value': col} for col in training_data.columns],
                    value=training_data.columns[1]
                ),
                dcc.Graph(id='scatter-plot'),
                dcc.Dropdown(
                    id='feature-boxplot',
                    options=[{'label': col, 'value': col} for col in training_data.columns],
                    value=training_data.columns[0]
                ),
                dcc.Graph(id='box-plot')
            ])
        ]),
        dcc.Tab(label='Feature Selection', value='tab-4', children=[
            html.Div([
                html.H2("Feature Selection"),
                html.P('Select the features that you wish to use for your model, and when its done, remember to confirm your selection!'),
                dcc.Dropdown(
                    id='feature-dropdown',
                    options=[{'label': col, 'value': col} for col in df_meteo.columns],
                    value=[df_meteo.columns[0]],
                    multi=True
                ),
                html.Div(id='feature-table-div'),
                html.Button('Confirm your selection', id='split-button'),
                html.Div(id='split-values'),
                html.Div([
                    html.H6(""),
                    html.Pre(id="x-values", style=white_text_style)
                ]),
                html.Div([
                    html.H6(""),
                    html.Pre(id="y-values", style=white_text_style)
                ]),
                html.Div([
                    html.H6(""),
                    html.Pre(id="x-2019-values", style=white_text_style)
                ]),
            ])
        ]),
        dcc.Tab(label='Models', value='tab-5', children=[
            html.Div([
                html.H2("Models"),
                html.P('Select the type of model you want to train with the features that you previously selected and press the button to train your model. If you dont select anything in the tab Feature Selection, it is impossible to train the model, so try again.'),
                dcc.Dropdown(
                    id='model-dropdown',
                    options=[
                        {'label': 'Auto Regressive', 'value': 'auto_regressive'},
                        {'label': 'Linear Regression', 'value': 'linear'},
                        {'label': 'Random Forests', 'value': 'random_forests'},
                        {'label': 'Bootstrapping', 'value': 'bootstrapping'},
                        {'label': 'Decision Tree Regressor', 'value': 'decision_trees'},
                        {'label': 'Gradient Boosting', 'value': 'gradient_boosting'}
                    ],
                    value='linear'
                ),
                html.Button('Train Model', id='train-model-button'),
            ]),
            html.Div([
                html.H2(""),
                dcc.Loading(
                    id="loading-1",
                    children=[html.Div([dcc.Graph(id="lr-graph")])]
                )
            ]),
        ]),
        dcc.Tab(label='Results of Forecast', value='tab-6', children=[
            html.Div([
                html.H2('Results of Forecast'),
                html.P('Here you can find by default the raw values of 2019. If you wish to check the results of your predictions vs. the actual real values and the performance metrics, press the button. It is worth mentioning that you need to do the steps for Feature Selection and Models to actually have predictions.'),
                dcc.Graph(id='time-series-plot', figure=fig2),
                dcc.Graph(id='scatter-plot-real-predicted'),
                html.Button('Run', id='button_model'),
                html.Div(id='model-performance-table')
            ])
        ]),
    ]),
    html.Div(id='tabs-content')
])


@app.callback(Output('graph', 'figure'),
              Input('column-dropdown', 'value'),
              Input('date-picker', 'start_date'),
              Input('date-picker', 'end_date')
)

def update_figure(columns, start_date, end_date):
    
    filtered_df = training_data.loc[start_date:end_date, columns]
    
    # Define a list to hold the y-axis configurations
    y_axis_config = []
    
    # Loop through each column and define a new y-axis configuration
    for i, column in enumerate(columns):
        y_axis_config.append({'overlaying': 'y', 'side': 'right', 'position': 1 - i * 0.1})
    
    # Define the data and layout of the figure
    data = [{'x': filtered_df.index, 'y': filtered_df[column], 'type': 'line', 'name': column} for column in filtered_df.columns]
    layout = {'title': {'text': ', '.join(columns)}, 'xaxis': {'title': 'Date'}}
    
    # Update the layout to include the y-axis configurations
    layout.update({'yaxis{}'.format(i + 1): y_axis_config[i] for i in range(len(y_axis_config))})
    
    # Create the figure with the data and layout
    fig = {'data': data, 'layout': layout}
    
    return fig

@app.callback(Output('graph-2019', 'figure'),
              Input('column-dropdown-2019', 'value'),
              Input('date-picker-2019', 'start_date'),
              Input('date-picker-2019', 'end_date')
)

def update_figure_2019(columns, start_date, end_date):
    filtered_df = df_real_2019.loc[start_date:end_date, columns]
    
    # Define a list to hold the y-axis configurations
    y_axis_config = []
    
    # Loop through each column and define a new y-axis configuration
    for i, column in enumerate(columns):
        y_axis_config.append({'overlaying': 'y', 'side': 'right', 'position': 1 - i * 0.1})
    
    # Define the data and layout of the figure
    data = [{'x': filtered_df.index, 'y': filtered_df[column], 'type': 'line', 'name': column} for column in filtered_df.columns]
    layout = {'title': 'Raw Data 2019', 'xaxis': {'title': 'Date'}}
    
    # Update the layout to include the y-axis configurations
    layout.update({'yaxis{}'.format(i + 1): y_axis_config[i] for i in range(len(y_axis_config))})
    
    # Create the figure with the data and layout
    fig = {'data': data, 'layout': layout}
    
    return fig

@app.callback(
    Output('scatter-plot', 'figure'),
    Input('feature1', 'value'),
    Input('feature2', 'value')
)

def update_scatter_plot(feature1, feature2):
    fig = px.scatter(df_total, x=feature1, y=feature2, title=f'{feature1} vs {feature2}')
    return fig

@app.callback(
    Output('box-plot', 'figure'),
    Input('feature-boxplot', 'value')
)

def update_box_plot(feature_boxplot):
    fig = go.Figure()
    fig.add_trace(go.Box(y=df_total[feature_boxplot], name=feature_boxplot))
    fig.update_layout(title=f"Box Plot for {feature_boxplot}", title_x=0.5)
    return fig

@app.callback(
    Output('feature-table-div', 'children'),
    Input('feature-dropdown', 'value')
)

def update_feature_table(selected_features):
    if selected_features:
        global df_model
        df_model = df_meteo[selected_features]
        table = generate_table(df_model)
        return table
    else:
        return html.Div()
    
@app.callback(
    Output('x-values', 'children'),
    Output('y-values', 'children'),
    Output('x-2019-values', 'children'),
    Input('feature-dropdown', 'value')
)
def update_x_y(selected_features):
    global X, Y, X_2019
    if selected_features:
        # Concatenate selected features with the target variable
        selected_features_with_target = selected_features + ['Power (kW)']
        
        # Filter the DataFrame based on selected features
        df_selected_features = training_data[selected_features_with_target]
        
        # Split the data into features (X) and target variable (Y)
        X = df_selected_features.drop(columns=['Power (kW)']).values
        Y = df_selected_features['Power (kW)'].values 
        
        # Extract features for the year 2019
        X_2019 = df_meteo_2019[selected_features].values
        
        return str(X), str(Y), str(X_2019)
    else:
        return "", "", ""

    
@app.callback(
    Output('split-values', 'children'),
    Input('split-button', 'n_clicks')
)

def generate_train_test_split(n_clicks):
    global X_train, X_test, y_train, y_test
    if n_clicks:
        X_train, X_test, y_train, y_test = train_test_split(X, Y)
        return 'Done! Continue to the next tab if you wish!'
    else:
        return ""
    
#Define global variables
y_pred_list = []
y_pred2019 = []

@app.callback(
    Output('lr-graph', 'figure'),
    Input('train-model-button', 'n_clicks'),
    State('model-dropdown', 'value')
)

def train_and_predict(n_clicks, model_type):
    global y_pred_list, y_pred2019  # access global variable

    if n_clicks is None:
        return dash.no_update 
    else:

        if model_type == 'linear':
            # Linear Regression
            model = LinearRegression()
        elif model_type == 'random_forests':
            # Random Forests
            parameters = {'bootstrap': True,
                          'min_samples_leaf': 3,
                          'n_estimators': 200, 
                          'min_samples_split': 15,
                          'max_features': 'sqrt',
                          'max_depth': 20,
                          'max_leaf_nodes': None}
            model = RandomForestRegressor(**parameters)
        elif model_type == 'bootstrapping':
            # Bootstrapping
            model = BaggingRegressor()
        elif model_type == 'decision_trees':
            # Decision Trees
            model = DecisionTreeRegressor()
        elif model_type == 'auto_regressive':
            # Auto-Regressive (AR) Model
            model = AutoReg(endog=y_train, lags=5)
        elif model_type == 'gradient_boosting':
            # Gradient Boosting
            model = GradientBoostingRegressor()
        
        if model_type == 'auto_regressive':
            ar_model = model.fit()
            # Save the trained model
            with open('ar_model.pkl', 'wb') as file:
                pickle.dump(ar_model, file)
                file.close()

            # Make predictions
            y_pred = ar_model.predict(start=len(y_train), end=len(y_train)+len(X_test)-1, dynamic=False)
            y_pred_list.append(y_pred)
            y_pred2019 = ar_model.predict(start=len(Y), end=len(Y)+len(X_2019)-1, dynamic=False)

            # Generate scatter plot of predicted vs actual values
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers'))
            fig.update_layout(title=f'AutoRegressive Predictions')
        else:
            # Train the model using the training sets
            model.fit(X_train, y_train)

            # Save the trained model
            with open('model.pkl', 'wb') as file:
                pickle.dump(model, file)
                file.close()

            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_list.append(y_pred)
        
            y_pred2019 = model.predict(X_2019)
        
            # Generate scatter plot of predicted vs actual values
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers'))
            fig.update_layout(title=f'{model_type.capitalize()} Predictions')
        
        return fig

@app.callback(
    Output('time-series-plot', 'figure'),
    Output('scatter-plot-real-predicted', 'figure'),
    Output('model-performance-table', 'children'),
    Input('button_model', 'n_clicks')
)

def run_model(n_clicks):
    global y_pred2019 
    if n_clicks is None:
        raise PreventUpdate
    else:
        #Plot of Real vs. Predicted Power - time series
        fig = go.Figure(layout=go.Layout(title='Real vs Predicted Power Consumption'))
        fig.add_scatter(x = df_real['Date'].values, y = df_real['Central (kWh)'].values, name='Real Power')
        fig.add_scatter(x = df_real['Date'].values, y = y_pred2019, name='Predicted Power') 

        #Plot of Real vs. Predicted Power - scatter plot
        scatter_plot_fig = go.Figure()
        scatter_plot_fig.add_trace(go.Scatter(x=df_real['Central (kWh)'], y=y_pred2019, mode='markers'))
        scatter_plot_fig.update_layout(
        title='Real vs Predicted Power Consumption',
        xaxis_title='Real Power',
        yaxis_title='Predicted Power'
        )

        # Calculate model performance metrics
        MAE = metrics.mean_absolute_error(df_real['Central (kWh)'].values, y_pred2019)
        MBE = np.mean(df_real['Central (kWh)'].values - y_pred2019)
        MSE = metrics.mean_squared_error(df_real['Central (kWh)'].values, y_pred2019)
        RMSE = np.sqrt(MSE)
        cvrmse = RMSE / np.mean(df_real['Central (kWh)'].values)
        nmbe = MBE / np.mean(df_real['Central (kWh)'].values)

        # Format the metrics as percentages with two decimal places
        cvRMSE_perc = "{:.2f}%".format(cvrmse * 100)
        NMBE_perc = "{:.2f}%".format(nmbe * 100)
        
        # Create the table with the metrics
        d = {'MAE': [MAE],'MBE': [MBE], 'MSE': [MSE], 'RMSE': [RMSE],'cvMSE': [cvRMSE_perc],'NMBE': [NMBE_perc]}
        df_metrics = pd.DataFrame(data=d)
        table = generate_table(df_metrics)
        
    return fig, scatter_plot_fig, table

    
if __name__ == '__main__':
    app.run_server()

