#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data = pd.read_csv('/Users/scotthuston/Desktop/NFL Play by Play 2009-2018 (v5).csv')
data.head()


# In[2]:


future_columns = ['quarter_end', 'sp','ydsnet', 'desc', 'yards_gained', 
                'qb_dropback', 'qb_kneel', 'qb_spike', 'qb_scramble',
                'pass_length', 'pass_location', 'air_yards', 'yards_after_catch', 
                'run_location', 'field_goal_result', 'kick_distance', 'extra_point_result', 'run_gap', 'two_point_conv_result', 'timeout', 
               'timeout_team', 'td_team', 'posteam_score_post', 'defteam_score_post',
               'score_differential_post', 'epa', 'total_home_epa', 
                'total_away_epa', 'total_home_pass_epa', 'total_away_pass_epa', 
                'air_epa', 'yac_epa', 'comp_air_epa', 'comp_yac_epa', 
                'total_home_comp_air_epa', 'total_away_comp_air_epa', 
                'total_home_comp_yac_epa', 'total_home_raw_air_epa', 
                'total_away_raw_air_epa', 'total_home_raw_yac_epa', 
                'total_away_raw_yac_epa', 'wpa', 'home_wp_post', 'away_wp_post',
                'total_home_rush_wpa', 'total_away_rush_wpa', 
                'total_home_pass_wpa', 'total_away_pass_wpa', 'air_wpa', 
                'yac_wpa', 'comp_air_wpa', 'comp_yac_wpa', 
                'total_home_comp_air_wpa', 'total_away_comp_air_wpa', 
                'total_home_comp_yac_wpa', 'total_away_comp_yac_wpa', 
                'total_home_raw_air_wpa', 'total_away_raw_air_wpa', 
                'total_home_raw_yac_wpa', 'total_away_raw_yac_wpa', 
                'punt_blocked', 'first_down_rush', 'first_down_pass', 
                'third_down_converted', 'third_down_failed', 
                'fourth_down_converted', 'fourth_down_failed', 
                'incomplete_pass', 'interception', 'fumble_forced', 
                'fumble_not_forced', 'fumble_out_of_bounds', 'solo_tackle', 
                'safety', 'penalty', 'tackled_for_loss', 'fumble_lost', 
                'qb_hit', 'rush_attempt', 'pass_attempt', 'sack', 'touchdown', 
                'pass_touchdown', 'rush_touchdown', 'fumble', 'complete_pass', 
                'assist_tackle', 'lateral_reception', 'lateral_rush', 
                'lateral_return', 'lateral_recovery', 'passer_player_id', 
                'passer_player_name', 'receiver_player_id', 
                'receiver_player_name', 'rusher_player_id', 
                'rusher_player_name', 'lateral_receiver_player_id', 
                'lateral_receiver_player_name', 'lateral_rusher_player_name', 
                'lateral_sack_player_id', 'lateral_sack_player_name', 
                'interception_player_id', 'interception_player_name', 
                'lateral_interception_player_id', 
                'lateral_interception_player_name', 
                'tackle_for_loss_1_player_id', 'tackle_for_loss_1_player_name', 
                'tackle_for_loss_2_player_id', 'tackle_for_loss_2_player_name', 
                'qb_hit_1_player_id', 'qb_hit_1_player_name', 
                'qb_hit_2_player_id', 'qb_hit_2_player_name', 
                'forced_fumble_player_1_team', 'forced_fumble_player_1_player_id', 
                'forced_fumble_player_1_player_name', 'forced_fumble_player_2_team', 
                'forced_fumble_player_2_player_id', 'forced_fumble_player_2_player_name',
                'solo_tackle_1_player_id', 'solo_tackle_1_player_id', 
                'solo_tackle_1_player_name', 'solo_tackle_2_player_name', 
                'assist_tackle_1_player_id', 'assist_tackle_1_player_name',
                'assist_tackle_1_team', 'assist_tackle_2_player_id', 
                'assist_tackle_2_player_name', 'assist_tackle_2_team', 
                'assist_tackle_3_player_id', 'assist_tackle_3_player_name', 
                'assist_tackle_3_team', 'assist_tackle_4_player_id', 
                'assist_tackle_4_player_name', 'assist_tackle_4_team', 
                'pass_defense_1_player_id', 'pass_defense_1_player_name', 
                'pass_defense_2_player_id', 'pass_defense_2_player_name',
                'fumbled_1_team', 'fumbled_1_player_id', 'fumbled_1_player_name', 
                'fumbled_2_player_id', 'fumbled_2_player_name', 'fumbled_2_team', 
                'fumble_recovery_1_team', 'fumble_recovery_1_yards', 
                'fumble_recovery_1_player_id', 'fumble_recovery_1_player_name',
                'fumble_recovery_2_team', 'fumble_recovery_2_yards', 
                'fumble_recovery_2_player_id', 'fumble_recovery_2_player_name', 
                'penalty_team', 'penalty_player_id', 'penalty_player_name', 
                'penalty_yards', 'replay_or_challenge', 
                'replay_or_challenge_result', 'penalty_type', 
                'defensive_two_point_attempt', 'defensive_two_point_conv', 
                'defensive_extra_point_attempt', 'defensive_extra_point_conv', 
                'lateral_rusher_player_id', 'punt_returner_player_id', 
                'punt_returner_player_name', 'lateral_punt_returner_player_id', 
                'lateral_punt_returner_player_name', 'kickoff_returner_player_name', 
                'kickoff_returner_player_id', 'lateral_kickoff_returner_player_name',
                'punter_player_id', 'punter_player_name', 'kicker_player_name', 
                'kicker_player_id', 'own_kickoff_recovery_player_id', 
                'own_kickoff_recovery_player_name', 'blocked_player_id', 
                'blocked_player_name', 'solo_tackle_1_team', 'solo_tackle_2_team', 
                'solo_tackle_2_player_id', 'return_team', 'punt_inside_twenty', 
                'punt_in_endzone', 'punt_out_of_bounds', 'punt_downed', 
                'punt_fair_catch', 'kickoff_inside_twenty', 'kickoff_in_endzone',
                'kickoff_out_of_bounds', 'kickoff_downed', 'kickoff_fair_catch', 
                'own_kickoff_recovery', 'own_kickoff_recovery_td', 'return_touchdown', 
                'extra_point_attempt', 'two_point_attempt', 'kickoff_attempt', 
                'punt_attempt', 'lateral_kickoff_returner_player_id', 'return_yards', 
                'first_down_penalty', 'total_home_rush_epa', 'total_away_rush_epa',
                'total_away_comp_yac_epa'
              ]     


target = 'play_type'


# In[3]:


# Changing the play type to pass if the qb scrambled as I'm trying to predict play calls, not play results

def scramble_to_pass(row):
  if row['qb_scramble']==1:
    row['play_type'] = 'pass'
  return row

data1=data.apply(scramble_to_pass, axis = 1)


# In[4]:


# Checking that my scramble_to_pass function worked correctly
data1[data1.qb_scramble == 1]


# In[5]:


# Removing columns about the result of the play
data1 = data1.drop(future_columns, axis = 'columns')
data1.head().T


# In[6]:


# Filtering to only look at pass and run plays
df = data1[(data1['play_type']=='pass') | (data1['play_type']=='run')]
df.head()


# In[7]:


# Splitting data into training, validation, and test sets
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(df.drop('play_type', axis = 'columns'), df['play_type'], train_size = .8, test_size = .2, stratify = df['play_type'])
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size = .75, test_size = .25, stratify = y_train)


# In[8]:


# Making pipeline and fitting basic random forest

from sklearn.pipeline import make_pipeline
import category_encoders as ce
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# In[9]:


from sklearn.ensemble import GradientBoostingClassifier
GDB_pipeline = make_pipeline(ce.OrdinalEncoder()
                            ,IterativeImputer()
                            ,GradientBoostingClassifier())

GDB_pipeline.fit(X_train, y_train)


# In[10]:


from sklearn.metrics import roc_auc_score

y_val_pred_proba = GDB_pipeline.predict_proba(X_val)[:,1]
roc_auc_score(y_val, y_val_pred_proba)


# In[11]:


# plotting feature importances
import matplotlib.pyplot as plt

encoder = GDB_pipeline.named_steps['ordinalencoder']
encoded_columns = encoder.fit_transform(X_train).columns
importances = pd.Series(GDB_pipeline.named_steps.gradientboostingclassifier.feature_importances_, encoded_columns)
plt.figure(figsize = (10,40))
importances.sort_values().plot.barh(color = 'gray');
plt.show()


# In[12]:


# Rerunning model with only select features

def cut_features(df):
    df = df.copy()
    df = df[['shotgun', 'down', 
            'ydstogo', 'game_seconds_remaining', 
            'yardline_100', 'score_differential', 
            'posteam_timeouts_remaining', 
            'defteam_timeouts_remaining', 'defteam_score',
            'posteam_score']]
    return df

# creating new X_train and X_val with just those features
X_train_cut = cut_features(X_train)
X_val_cut = cut_features(X_val)


# In[13]:


# refitting pipeline
GDB_pipeline = make_pipeline(ce.OrdinalEncoder()
                            ,IterativeImputer()
                            ,GradientBoostingClassifier(verbose = 1))

GDB_pipeline.fit(X_train_cut, y_train)


# In[14]:


# trimming whole df in advance of using CV methods
X_df = cut_features(df)
y_df = df[target]
df.head()


# In[15]:


X_df.head()


# In[16]:


X_df.shape


# In[17]:


y_df.head()


# In[18]:


# Using randomizedsearchCV to optimize hyperparameters

from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

param_distributions = {
    'gradientboostingclassifier__learning_rate': uniform(.001, .2),
#     'gradientboostingclassifier__n_estimators': randint(30,3000),
    'gradientboostingclassifier__min_samples_leaf': randint(1,50),
    'gradientboostingclassifier__max_depth': randint(2,30)
}

search = RandomizedSearchCV(
    GDB_pipeline, 
    param_distributions=param_distributions, 
    n_iter=1, 
    cv=2, 
    scoring='accuracy', 
    verbose=10, 
    return_train_score=True, 
    n_jobs=-1
)

search.fit(X_df, y_df);


# In[ ]:


pd.DataFrame(search.cv_results_).sort_values(by='rank_test_score')


# In[19]:


# Checking ROC_AUC of new smaller model
# The new model is almost as good which is what I was hoping
y_val_pred_proba = GDB_pipeline.predict_proba(X_val_cut)[:,1]
roc_auc_score(y_val, y_val_pred_proba)


# In[20]:


encoder = GDB_pipeline.named_steps['ordinalencoder']
encoded_columns = encoder.fit_transform(X_train_cut).columns
importances = pd.Series(GDB_pipeline.named_steps.gradientboostingclassifier.feature_importances_, encoded_columns)
plt.figure(figsize = (10,40))
importances.sort_values().plot.barh(color = 'gray');
plt.show()


# In[ ]:


# Pickling the model for future use
from joblib import dump

dump(GDB_pipeline, 'pipeline.joblib')


# In[21]:


X_train_cut[X_train_cut.yardline_100<10]


# In[22]:


X_train_cut.iloc[[0]]


# In[24]:


# getting shapley plot for Seahawks situation
import shap
row = X_train_cut.iloc[[0]]
row['shotgun'] = 1
row['down']=2
row['ydstogo']=1
row['game_seconds_remaining']=24
row['yardline_100']=1
row['score_differential']=-4
row['posteam_timeouts_remaining']=1
row['defteam_timeouts_remaining']=2
row['defteam_score']=28
row['posteam_score']=24

    
explainer = shap.TreeExplainer(GDB_pipeline.named_steps.gradientboostingclassifier)
df_encoded = GDB_pipeline.named_steps.ordinalencoder.transform(row)
df_imputed = GDB_pipeline.named_steps.iterativeimputer.transform(df_encoded)
shap_values = explainer.shap_values(df_imputed)

shap.initjs()
shap.force_plot(
    base_value = explainer.expected_value,
    shap_values = shap_values,
    features = row,
    feature_names = ['Shotgun Formation', 'Down', 'Yards to 1st Down', 
                     'Seconds Remaining', 'Yards from end zone', 
                     'Score Differential', 'SEA timeouts',
                    'NE timeouts', 'NE score', 'SEA score'],
)

plt.figure(figsize=(40,20))
plt.show()


# In[25]:


# Getting shapley plot for Seahawks situation if not in shotgun formation
row['shotgun'] = 0

explainer = shap.TreeExplainer(GDB_pipeline.named_steps.gradientboostingclassifier)
df_encoded = GDB_pipeline.named_steps.ordinalencoder.transform(row)
df_imputed = GDB_pipeline.named_steps.iterativeimputer.transform(df_encoded)
shap_values = explainer.shap_values(df_imputed)

# shap.initjs()
shap.force_plot(
    base_value = explainer.expected_value,
    shap_values = shap_values,
    features = row,
    feature_names = ['Shotgun Formation', 'Down', 'Yards to 1st Down', 
                     'Seconds Remaining', 'Yards from end zone', 
                     'Score Differential', 'SEA timeouts',
                    'NE timeouts', 'NE score', 'SEA score'],
    matplotlib = True
)
plt.figure(figsize=(10,5))
plt.show()


# In[26]:


GDB_pipeline.predict_proba(row)


# In[27]:


# ROC_AUC_Score
from sklearn.metrics import roc_auc_score
y_pred_proba = GDB_pipeline.predict_proba(X_val_cut)[:,1]
roc_auc_score(y_val, y_pred_proba)


# In[57]:


from sklearn.metrics import accuracy_score
y_pred = GDB_pipeline.predict(X_val_cut)
accuracy_score(y_val, y_pred)


# In[58]:


y_val.describe()


# In[59]:


38730/63945


# In[56]:


# plot ROC curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
import matplotlib.style as style
from matplotlib.pyplot import figure
figure(num=None, figsize=(10, 10), dpi=300, facecolor='#013F8E')
# fig = plt.figure()
# fig.patch.set_facecolor('#013F8E')
# plt.style.use('dark_background')
fpr, tpr, thresholds = roc_curve(y_val=='run', y_pred_proba)
fpr = pd.Series(fpr)
tpr = pd.Series(tpr)
plt.plot(fpr,tpr, color = '#FFFFFF')
ax = plt.gca()
ax.set_facecolor('#013F8E')
plt.title('ROC Curve', fontweight = 'bold', fontsize = 40)
plt.xlabel('False Positive Rate', fontsize = 35)
plt.ylabel('True Positive Rate', fontsize = 35)

plt.show()


# In[48]:


# getting feature importances

import eli5
from eli5.sklearn import PermutationImportance

encoder = GDB_pipeline.named_steps.ordinalencoder
X_train_encoded = encoder.fit_transform(X_train_cut)
X_val_encoded = encoder.transform(X_val_cut)

imputer = GDB_pipeline.named_steps.iterativeimputer
X_train_imputed = imputer.fit_transform(X_train_encoded)
X_val_imputed = imputer.fit_transform(X_val_encoded)

model = GDB_pipeline.named_steps.gradientboostingclassifier
# model.fit(X_train_imputed,y_train)

permuter = PermutationImportance(model
                ,scoring = 'accuracy'
                ,n_iter = 2)
permuter.fit(X_val_imputed, y_val)
feature_names = X_val_encoded.columns.tolist()
eli5.show_weights(permuter, top = None, feature_names=feature_names)


# In[78]:


from pdpbox import pdp

plt.style.use('seaborn-dark-palette')
feature = 'down'
model = GDB_pipeline.named_steps['gradientboostingclassifier']
model_features = X_train_cut.columns
X_train_imputed = pd.DataFrame(X_train_imputed)
X_train_imputed.columns = X_train_cut.columns
pdp_dist = pdp.pdp_isolate(model = model, dataset = X_train_imputed, model_features = model_features, feature = feature)
pdp.pdp_plot(pdp_dist,
             feature_name = feature, 
             plot_lines = True, 
             frac_to_plot = 100,
             plot_params = {
    'title': '',
    'subtitle': ''
             }
            ); # if fract_to_plot > 1, that's just the number of lines to plot
plt.xticks([1,2,3,4], fontsize = 30, color ='white')
plt.yticks([-.5,-.4,-.3,-.2,-.1,0],fontsize = 30, color = 'white')
plt.ylabel('Effect on likelihood of run', fontsize = 30)
plt.xlabel('Down', fontsize = 30)
plt.title('')
plt.suptitle('')


# In[98]:



# plotting feature interactions with partial dependence plots
from pdpbox.pdp import pdp_interact, pdp_interact_plot
features = ['score_differential', 'game_seconds_remaining']
plt.style.use('default')
# figure(num=None, figsize=(10, 10), dpi=300)
interaction = pdp_interact(
    model = model,
    dataset = X_train_imputed,
    model_features = X_train_imputed.columns,
    features = features,
    
)


fig, ax = pdp_interact_plot(interaction, plot_type = 'contour', feature_names = features);


ax.xlabel('Score differential')
ax.ylabel('Seconds remaining in game')




# In[81]:





# In[ ]:




