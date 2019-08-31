import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app

row1 = dbc.Row(
    [
        dcc.Markdown(
            """     
            ## About Robo Romo
            """,
            className = 'mb-3'
           
          
        ),
        dcc.Markdown(
            """
            #### **Dataset and model choice**
            
            ##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Robo Romo is built on a [dataset](https://www.kaggle.com/maxhorowitz/nflplaybyplay2009to2016#NFL%20Play%20by%20Play%202009-2018%20(v5).csv) of all of the NFL plays from 2009 to 2018. Tree-based models were the most promising model type because of the likely non-monotonic relationships between the features and outcomes. For example, teams may pass a lot before the end of the second or fourth quarters to score before time runs out, but the end of the first and third quarters are unlikely to have similar effects. I fit both random forest and gradient boosting models, settling on a gradient boosting model because it achieved the best ROC AUC score on my test set.
            """
        ),
    ]
)
row2 = dbc.Row(
    [
        dcc.Markdown(
            """
            #### **Model performance**
            """
        )
    ],
    className = 'mt-5'
)
col1 = dbc.Col(
    [  
        html.Img(src='assets/ROC_curve2.png',
                         className='img-fluid',
                         style={'display': 'block', 'margin-left': 'auto','margin-right': 'auto','width' : '90%'}
                        ),
    ],
    md = 3
)
col2 = dbc.Col(
    [
        html.Div(
            [
        
                dcc.Markdown(
                    """

                    ##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ROC AUC scores are a measure of how well models rank probabilities. This model achieves an ROC AUC score of 0.81, meaning that 81% of the graph on the right is under the curve. A perfect predictor would have an ROC AUC score of 1, because it would have a 100% true positive rate with a 0% false positive rate. This model correctly guessed the play on the test set 74% of the time. For comparison, a mean baseline (predicting pass every time) would have been correct 61% of the time.

                    """,
                className ='mt-5'
                )
            ],
            style={'vertical-align': 'center'}
        )
   ]
)
row3 = dbc.Row([col1,col2])

row4 = dbc.Row(
    [
        dcc.Markdown(
            """
            #### **Understanding the model**
            """
        )
    ],
    className = 'mt-5'
)

col3 = dbc.Col(
    [
        html.Img(src='assets/feature_importances.png',
                         className='img-fluid',
                         style={'display': 'block', 'margin-left': 'auto','margin-right': 'auto','width' : '90%'}
                        ),
        dcc.Markdown(
            """
            ##### **Permutation Importances**
            """,
            style={'textAlign': 'center'}
        )
    ]
)
col4 = dbc.Col(
    [
        html.Img(src='assets/down_pdp.png',
                         className='img-fluid',
                         style={'display': 'block',
                                'margin-left': 'auto',
                                'margin-right': 'auto',
                                'margin-top':'2%',
                                'width' : '90%'}
                        ),
        dcc.Markdown(
            """
            ##### **Partial Dependence - Down**
            """,
            style={'textAlign': 'center'},
            className = 'mt-2'
            
        )
    ]
)

row5 = dbc.Row([col3, col4], className = 'mt-3')
row6 = dbc.Row(
    [
        dcc.Markdown(
            """
            
            ##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The figure on the left shows the relative importance of the different inputs to the model overall. As football fans might expect, whether the team is in a shotgun formation makes a significant difference, along with the down and distance to the first down line. The number of timeouts that teams have remaining only impacts the model in rare end-of-half situations. Still, I chose to include those features in the model because those situations are likely of the most interest.
            ##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The figure on the right shows how the likelihood of a run play changes as the down changes. Teams tend to pass more as they get to later downs. Interestingly though, the model does not seem to distinguish between third down plays and fourth down plays.
            ##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Below, you can see how play calls change as the time remaining and score change. Lighter colors indicate a higher likelihood of run plays. There is a clear tendency towards running at the end of a game when a team is in the lead and a clear but less distinctive tendency to pass when teams are behind at the end of the game. In addition, there is a clear tendency to pass more right before halftime as teams are more aggressive trying to score before time runs out.
            """
        ),
       
        html.Img(src='assets/calls_game_state2.png',
                         className='img-fluid',
                         style={'display': 'block','margin-left': 'auto','margin-right': 'auto','width' : '60%'}
                        ),
        

        
        
        
        dcc.Markdown(
            """ 
            
            #### **Potential improvements**
            
            ##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;One straightforward way to improve this model would be through hyperparameter optimization. Running automated hyperparameter search implementations proved impractical with this dataset on my 2012 Macbook Air, but there are surely better parameters for the gradient boosting model. Another way the model could likely be improved is by re-fitting the model on all of the data once I selected a model. I have not done this in order for my model and the model performance information shown above to be consistent, but the model would likely perform better on unseen data by using all of the past data. I also made a conscious tradeoff not to include many features in the original dataset, or engineer more complicated features, in order to keep the model easily useable. The accuracy could certainly be improved by including or engineering more features.
             
            """,
            
        )
    ],
    className = 'mt-3'
    
)

layout = dbc.Col([row1,row2,row3,row4,row5,row6])
