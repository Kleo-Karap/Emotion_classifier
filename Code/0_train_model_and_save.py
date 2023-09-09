# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 23:52:40 2023

@author: kleop
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
# %% load prepared data
df = pd.read_pickle('C:/Users/kleop/Documents/repos/Ergasia/prepared_dataframe.pickle')

# %% calm/angry
act_c = df[ (df['emotion'] == 'calm')]
act_a = df[ (df['emotion'] == 'angry')]

# %% isolate features and labels (I chose mfccs as the feature based on which i'll train the model, because in the PCA plot the separation of the 2 classes is clear.
title = "MFCC Features"
print(f"\n{title}")

act_c_features = np.stack( act_c['mfcc_profile'].to_numpy() )
act_a_features = np.stack( act_a['mfcc_profile'].to_numpy() )
all_features = np.vstack((act_c_features, act_a_features))

act_c_labels = 0*np.ones( ( act_c.shape[0] , 1 ) )  #0 denotes the calm-class, 1 denotes the angry-class
act_a_labels = 1*np.ones( ( act_a.shape[0] , 1 ) )
all_labels = np.r_[act_c_labels , act_a_labels]

# %% train - test split
train_set , test_set = train_test_split( np.c_[ all_features , all_labels] , test_size=0.2 , random_state=0)

train_input = train_set[:, :-1]
train_label = train_set[:, -1]
test_input = test_set[:, :-1]
test_label = test_set[:, -1]


#%% Start trying out several tests
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit( train_input , train_label )
# make predictions from training data
preds = lin_reg.predict( test_input )
preds_binary = np.array( preds >= 0.5 ).astype(int)
comparison_check = np.c_[ preds , preds_binary , test_label ]
accuracy_linear = np.sum( test_label == preds_binary ) / preds.size

#%%
from sklearn.ensemble import RandomForestClassifier

forest_class = RandomForestClassifier()
forest_class.fit( train_input , train_label )
# make predictions from training data
preds_binary = forest_class.predict( test_input )
comparison_check = np.c_[ preds_binary , test_label ]
accuracy_forest = np.sum( test_label == preds_binary ) / preds.size

#%%
from sklearn.svm import SVC

svm_class = SVC()
svm_class.fit( train_input , train_label )
# make predictions from training data
preds_binary = svm_class.predict( test_input )
comparison_check = np.c_[ preds_binary , test_label ]
accuracy_svm = np.sum( test_label == preds_binary ) / preds.size


#%% 
from sklearn.linear_model import LogisticRegression
log_reg= LogisticRegression()


# train classifier
log_reg.fit( train_input , train_label )
# make predictions from training data
preds_binary = log_reg.predict( test_input )
comparison_check = np.c_[ preds_binary , test_label ]
accuracy_logistic = np.sum( test_label == preds_binary ) / preds_binary.size

#%%
from sklearn.metrics import make_scorer

def binary_accuracy( y_true , y_pred ):
    bin_pred = np.array( y_pred >= 0.5 ).astype(int)
    return np.sum( y_true == bin_pred ) / y_true.size

my_scorer = make_scorer(binary_accuracy, greater_is_better=True)

# %% cross validation

scores_lin = cross_val_score( lin_reg, all_features, all_labels,
                         scoring=my_scorer, cv=10 )

scores_forest = cross_val_score( forest_class, all_features, all_labels.ravel(),
                         scoring=my_scorer, cv=10 )

scores_svm = cross_val_score( svm_class, all_features, all_labels.ravel(),
                         scoring=my_scorer, cv=10 )

scores_logistic=cross_val_score( log_reg, all_features, all_labels.ravel(),
                         scoring=my_scorer, cv=10 )


def present_scores( s , algorithm='method' ):
    print(30*'-')
    print( algorithm + ' accuracy in 10-fold cross validation:' )
    print('mean: ' + str( np.mean(s) ))
    print('std: ' + str( np.std(s) ))
    print('median: ' + str( np.median(s) ))

present_scores( scores_lin , algorithm='linear regression' )
present_scores( scores_forest , algorithm='random forest' )
present_scores( scores_svm , algorithm='SVM' )
present_scores(scores_logistic ,algorithm='Logistic Regression')

# %% save model
# %% save model
filename = 'random_forest_emotion.sav'
pickle.dump(forest_class, open(filename, 'wb'))

