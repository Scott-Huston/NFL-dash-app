import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app

column1 = dbc.Col(
    [
        dcc.Markdown(
            """
        
            ## The 1-Yard Line


            """,
            className='mb-3'
        ),
        
        dcc.Markdown(
            """
            
            ##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;It's the most controversial run or pass decision in recent memory. The Seahawks were down 24-28 in the Super Bowl with 24 seconds remaining in the game. They lined up in a shotgun formation from the 1-yard line ... and promptly threw an interception to lose the game. Naturally, fans and commentators questioned the decision heavily, but what does the model have to say about that situation?
            ##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If we enter the same situation fans saw when the Seahawks lined up on the 1-yard line, the model gives an 82% likelihood that the Seahawks had called a pass play. Below is a visual representation of how the model weights the factors in the game to come up with that prediction.
            
            """
        ),
        
        
    ],
    md=5,
)


column2 = dbc.Col(
    [
        html.Img(src='assets/interception.jpg', className='img-fluid'),
    ]
)

row2 = dbc.Col(
    [
    dcc.Markdown('#### Pass - 82%', className='mb-2', style={'textAlign': 'center', 'color': '#2888E5'}), 
    
      # Shapley plot here
        html.Img(src='assets/interception_shotgun_shapley.png', 
                 className='img-fluid', 
                 style={'display': 'block', 'margin-left': 'auto','margin-right': 'auto','width' : '60%'}), 
        dcc.Markdown(
            """
            
            ##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The red arrows indicate parts of the situation that led the model to be more likely to predict a pass, and information with blue arrows made the model more likely to predict a run. The length of the arrows indicates the relative importance of the information. As we would intuitively expect, being on the 1-yard line made a run more likely, while being down late in the game with only one timeout remaining made a pass more likely.
            ##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The feature that had the biggest impact on the model predicting an 82% likelihood of a pass play being called was the fact that the team was in a shotgun formation. They didn't have to call a play from that formation though. If we re-enter the same situation into the model but without the Seahawks in a shotgun formation the model gives a 58% chance of a run play being called.
            
            """,
            className = 'mt-3',
        ),
        
        dcc.Markdown('#### Run - 58%', className='mt-3', style={'textAlign': 'center', 'color': '#E94153'}), 
        
        # Shapley plot here (without shotgun formation)
        html.Img(src='assets/interception_noShotgun_shapley.png', 
                 className='img-fluid', 
                 style={'display': 'block', 'margin-left': 'auto','margin-right': 'auto','width' : '60%'}), 
      
         dcc.Markdown(
            """
            
            ##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If the Seahawks hadn't chosen to line up in a shotgun formation, the model does predict a run play. Even in that hypothetical scenario though, the model is only slightly more confident that a run play has been called than a pass play. It's important to note that the model doesn't account for the fact that Seahawks running back Marshawn Lynch was considered to be one of the best running backs in the league, and especially good in short-yardage situations. A model that took that into account would likely change the results to make a run play a bit more likely. Still, it doesn't seem that the decision to pass in that scenario was particularly unusual.
            
            """,
            className = 'mt-3',
        ),
        
        
    ]
)
row1 = dbc.Row([column1,column2], className= 'mb-2')

layout = dbc.Col([row1, row2])
