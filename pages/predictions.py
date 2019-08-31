import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import shap
import plotly

from app import app

from joblib import load
pipeline = load('assets/pipeline.joblib')

column1 = dbc.Col(
     [
        html.H2('Play Prediction', className='mb-4', style={'textAlign': 'center'}), 
        html.Div(id='prediction-content', className='mb-3', style={'textAlign': 'center', 'font-weight': 'bold'}),
        html.Div(id='prediction-image') 
    ],
    
    md = 4
)

col1 = dbc.Col(    
    [
     # Shotgun checkbox
        dcc.Markdown('##### Is the offense in shotgun formation?'),
        dcc.Dropdown(
            id='shotgun', 
            options= [
                {'label': 'No', 'value': 0},
                {'label': 'Yes', 'value': 1},
#                 {'label': 'Unknown', 'value': pd.np.NaN},
            ],
            className = 'mb-3',
            value=0
        ),
    # Down dropdown menu
        dcc.Markdown('##### Down'), 
        dcc.Dropdown(
            id='down', 
            options= [
                {'label': '1st down', 'value': 1},
                {'label': '2nd down', 'value': 2},
                {'label': '3rd down', 'value': 3},
                {'label': '4th down', 'value': 4}
            ],
            className = 'mb-3',
            value=1
        ), 
     # Yards to go field
        dcc.Markdown('##### Yards to 1st down'), 
        dcc.Input(
            id = 'ydstogo',
            type='number',
            min = 0,
            max = 'yardline_100',
            step = 1,
            value=10,
            className = 'mb-3',
            debounce = True
        ),
     # Minutes remaining field
        dcc.Markdown('##### Minutes remaining in quarter'), 
        dcc.Input(
            id = 'minutes',
            type='number',
            min = 0,
            max = 15,
            value=14.5,
            className = 'mb-3',
            debounce = True
        ),
                       
     # Yardline slider
        html.Div(  
            [
            dcc.Markdown('##### Yards to touchdown'), 
            daq.Slider(              
                id='yardline_100', 
                min=1, 
                max=99, 
                step=1, 
                value=75, 
                marks={n: str(n) for n in range(10,91,10)}, 
                className='mb-3',
                handleLabel={
                    'label': 'Current',
                    'showCurrentValue': True
                    },
                ),
            ],
            style={'marginTop': 10, 'marginBottom': 10},            
        )                     
#         html.Div(id='slider-output-container'),
    ],        
    # setting column width                  
)

col2 = dbc.Col(
     [    # Quarter dropdown menu
        dcc.Markdown('##### Quarter'), 
        dcc.Dropdown(
            id='qtr', 
            options= [
                {'label': '1st quarter', 'value': 1},
                {'label': '2nd quarter', 'value': 2},
                {'label': '3rd quarter', 'value': 3},
                {'label': '4th quarter', 'value': 4},
                {'label': 'Overtime', 'value': 5}
            ],
            className = 'mb-3',
            value=1
        ),
      # Offensive score field
         html.Div([
            dcc.Markdown('##### Score of offensive team'), 
            dcc.Input(
                id = 'posteam_score',
                type='number',
                min = 0,
                step = 1,
                value=0,
                className = 'mb-3',
                debounce = True
            ),
         ],
         style={'marginTop': 10, 'marginBottom': 10}
         ),   
     # Defensive score field
        dcc.Markdown('##### Score of defensive team'), 
        dcc.Input(
            id = 'defteam_score',
            type='number',
            min = 0,
            step = 1,
            value=0,
            className = 'mb-3',
            debounce = True
        ),
     # Offensive team timeouts remaining menu
        dcc.Markdown('##### Offensive team timeouts remaining'), 
        dcc.Dropdown(
            id='posteam_timeouts_remaining', 
            options= [
                {'label': '3 timeouts', 'value': 3},
                {'label': '2 timeouts', 'value': 2},
                {'label': '1 timeout', 'value': 1},
                {'label': 'No timeouts', 'value': 0},
            ],
            className = 'mb-3',
            value=3
        ),
      # Defensive team timeouts remaining menu
        dcc.Markdown('##### Defensive team timeouts remaining'), 
        dcc.Dropdown(
            id='defteam_timeouts_remaining', 
            options= [
                {'label': '3 timeouts', 'value': 3},
                {'label': '2 timeouts', 'value': 2},
                {'label': '1 timeout', 'value': 1},
                {'label': 'No timeouts', 'value': 0},
            ],
            className = 'mb-3',
            value=3
        )
        
        
    ],
   
    
)

@app.callback(
    Output('prediction-content', 'children'),
    [
     Input('shotgun', 'value'), 
     Input('down', 'value'),
     Input('ydstogo', 'value'),
     Input('qtr', 'value'),
     Input('minutes', 'value'),
     Input('yardline_100', 'value'),
     Input('posteam_score', 'value'),
     Input('defteam_score', 'value'),
     Input('posteam_timeouts_remaining', 'value'),
     Input('defteam_timeouts_remaining', 'value'),    
    ]
)
def predict(shotgun, down, ydstogo, qtr, minutes,
            yardline_100, posteam_score, defteam_score,
            posteam_timeouts_remaining, defteam_timeouts_remaining):
    qtr_secs = 15*60
    full_qtrs = 4-qtr
    full_qtr_secs = full_qtrs*qtr_secs
    minute_secs = minutes*60
    game_seconds_remaining = full_qtr_secs+minute_secs
    posteam_score = int(posteam_score)
    defteam_score = int(defteam_score)
    score_differential = posteam_score-defteam_score
    df = pd.DataFrame(
        columns=['shotgun', 'down', 'ydstogo', 'game_seconds_remaining', 'yardline_100', 
                 'score_differential', 'posteam_timeouts_remaining', 
                 'defteam_timeouts_remaining', 'defteam_score', 'posteam_score'], 
        data=[[shotgun, down, ydstogo, game_seconds_remaining, yardline_100, 
               score_differential, posteam_timeouts_remaining, 
               defteam_timeouts_remaining, defteam_score, posteam_score]]
    )
      
    y_pred = pipeline.predict(df)[0]
    if y_pred == 'pass':
        y_pred_proba = pipeline.predict_proba(df)[0][0]
        return f'{y_pred_proba*100:.0f}% chance of a {y_pred}'
    else:
        y_pred_proba = pipeline.predict_proba(df)[0][1]
        return f'{y_pred_proba*100:.0f}% chance of a {y_pred}'
        
#     # getting shapley plot
    
#     explainer = shap.TreeExplainer(pipeline.named_steps.gradientboostingclassifier)
#     df_encoded = pipeline.named_steps.ordinalencoder.transform(df)
#     df_imputed = pipeline.named_steps.iterativeimputer.transform(df_encoded)
#     shap_values = explainer.shap_values(df_imputed)

#     shap.initjs()
#     mpl_plot = shap.force_plot(
#         base_value = explainer.expected_value,
#         shap_values = shap_values,
#         features = df,
#         matplotlib = True
#     )
#     from plotly.tools import mpl_to_plotly

@app.callback(
    Output('prediction-image', 'children'),
    [
     Input('shotgun', 'value'), 
     Input('down', 'value'),
     Input('ydstogo', 'value'),
     Input('qtr', 'value'),
     Input('minutes', 'value'),
     Input('yardline_100', 'value'),
     Input('posteam_score', 'value'),
     Input('defteam_score', 'value'),
     Input('posteam_timeouts_remaining', 'value'),
     Input('defteam_timeouts_remaining', 'value'),    
    ]
)
def predict_image(shotgun, down, ydstogo, qtr, minutes,
            yardline_100, posteam_score, defteam_score,
            posteam_timeouts_remaining, defteam_timeouts_remaining):
    qtr_secs = 15*60
    full_qtrs = 4-qtr
    full_qtr_secs = full_qtrs*qtr_secs
    minute_secs = minutes*60
    game_seconds_remaining = full_qtr_secs+minute_secs

    posteam_score = int(posteam_score)
    defteam_score = int(defteam_score)
    score_differential = posteam_score-defteam_score
    print('score diff', score_differential)
    df = pd.DataFrame(
        columns=['shotgun', 'down', 'ydstogo', 'game_seconds_remaining', 'yardline_100', 
                 'score_differential', 'posteam_timeouts_remaining', 
                 'defteam_timeouts_remaining', 'defteam_score', 'posteam_score'], 
        data=[[shotgun, down, ydstogo, game_seconds_remaining, yardline_100, 
               score_differential, posteam_timeouts_remaining, 
               defteam_timeouts_remaining, defteam_score, posteam_score]]
    )
      
    y_pred = pipeline.predict(df)[0]
    if y_pred == 'pass':
        return html.Img(src='assets/pass_image.jpeg',className='img-fluid', style = {'height': '400px'})
    else:
        return html.Img(src='assets/run_image.jpeg',className='img-fluid', style = {'height': '400px'})


@app.callback(
    dash.dependencies.Output('slider-output-container', 'children'),
    [dash.dependencies.Input('yardline_100', 'value')])
def update_output(yardline_100):
    if yardline_100 > 1:
        return f'{yardline_100} yards from the end zone'
    else:
        return f'{yardline_100} yard from the end zone'

column2_title = dbc.Row([html.H2('Game Situation', className='mb-6')], className = 'mb-3') 
    
column2_content = dbc.Row([col1, col2])
column2 = dbc.Col([column2_title, column2_content])
layout = dbc.Row([column1, column2])
