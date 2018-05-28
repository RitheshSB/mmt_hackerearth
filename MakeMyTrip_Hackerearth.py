
# coding: utf-8

# # Importing Required Packages

# In[1]:


import os
import pandas as pd
import numpy as np
import xgboost
import sklearn


# # Setting the working directory

# In[2]:


os.chdir(r'C:\Users\Rithesh\Desktop\MMT_Hackerearth')


# # Importing the Train and Test Files

# In[3]:


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


# In[274]:


(train_data['id'].reset_index(drop = False)['id'] - train_data['id'].reset_index(drop = False)['index']).sum()


# In[278]:


(test_data['id'].reset_index(drop = False)['id'] - test_data['id'].reset_index(drop = False)['index'] - 552).sum()


# In[277]:


test_data['id'].reset_index(drop = False).head()


# # Data Exploration

# ## Column Types: Continuous, Categorical and Target

# In[4]:


col_types = train_data.dtypes
col_types


# In[5]:


categorical_cols = col_types[col_types=='object'].index.values
categorical_cols


# In[6]:


continuous_cols = col_types[col_types!='object'].index.values
continuous_cols = continuous_cols[[i not in ['P','id'] for i in continuous_cols]]
continuous_cols


# In[7]:


target_col = ['P']
target_col


# # Column Exploration

# In[8]:


train_data[continuous_cols].describe()


# In[9]:


test_data[continuous_cols].describe()


# In[10]:


train_data[categorical_cols].apply(pd.Series.nunique)


# In[11]:


test_data[categorical_cols].apply(pd.Series.nunique)


# ## Missing values exploration

# In[12]:


train_data[continuous_cols].isnull().sum(axis = 0)


# In[13]:


train_data[categorical_cols].isnull().sum(axis = 0)


# In[14]:


test_data[continuous_cols].isnull().sum(axis = 0)


# In[15]:


test_data[categorical_cols].isnull().sum(axis = 0)


# # Data Preparation - Train Data

# In[229]:


categorical_append_train_test = pd.concat([train_data[np.append(categorical_cols,['id'])],test_data[np.append(categorical_cols,['id'])]],axis = 0).reset_index(drop = True)


# In[238]:


dummies_categorical_append_train_test = pd.concat([pd.get_dummies(categorical_append_train_test.drop('id',axis = 1)),categorical_append_train_test['id']], axis = 1)


# In[243]:


dummies_categorical_append_train_test.loc[dummies_categorical_append_train_test['id']<=552,:].drop('id',axis = 1).shape


# ## Creating Dummy Variables for categorical cols

# In[245]:


train_data_after_dummies = dummies_categorical_append_train_test.loc[dummies_categorical_append_train_test['id']<=552,:].drop('id',axis = 1)
train_data_after_dummies.describe()


# ## Outliers Manipulation

# In[246]:


train_data[continuous_cols].describe()


# In[247]:


def rm_outliers(pd_series):
    quantile_75 = pd_series.quantile(0.75)
    quantile_25 = pd_series.quantile(0.25)
    iqr = 1.5*(quantile_75-quantile_25)
    conditions = [pd_series > quantile_75 + iqr,
                  pd_series < quantile_25 - iqr
                 ]
    choices = [quantile_75+iqr, quantile_25-iqr]
    pd_series_after = pd.Series(np.select(conditions,choices, default=pd_series))
    return pd_series_after


# In[248]:


train_data_after_outliers = train_data[continuous_cols].apply(rm_outliers)


# In[249]:


train_data_after_outliers.describe()


# ## Missing Values Interpretation

# In[250]:


train_data_after_nans =train_data_after_outliers.fillna(train_data_after_outliers.mean())
train_data_after_nans.describe()


# ## Normalization

# In[251]:


def normalization(pd_series):
    pd_series_new = (pd_series - pd_series.mean())/pd_series.std()
    return pd_series_new


# In[252]:


train_data_after_normalization = train_data_after_nans.apply(normalization)
train_data_after_normalization.describe()


# ## Final Train Dataset after Cleaning

# In[253]:


train_data_after_clean = pd.concat([train_data_after_normalization,train_data_after_dummies,train_data[target_col]],axis = 1)


# # Data Preparation - Test Data

# ## Creating Dummy Variables for categorical cols

# In[265]:


test_data_after_dummies = dummies_categorical_append_train_test.loc[dummies_categorical_append_train_test['id']>552,:].drop('id',axis = 1).reset_index(drop = True)
test_data_after_dummies.describe()


# ## Missing Values Interpretation

# In[266]:


test_data[continuous_cols].describe()


# In[267]:


test_data_after_nans =test_data[continuous_cols].fillna(train_data_after_outliers.mean())
test_data_after_nans.describe()


# ## Normalization

# In[268]:


def normalization_using_train(pd_series):
    name_series = pd_series.name
    pd_series_new = (pd_series - train_data_after_nans[name_series].mean())/train_data_after_nans[name_series].std()
    return pd_series_new


# In[269]:


test_data_after_normalization = test_data_after_nans.apply(normalization_using_train)
test_data_after_normalization.describe()


# ## Final Test Dataset after Cleaning

# In[270]:


test_data_after_clean = pd.concat([test_data['id'],test_data_after_normalization,test_data_after_dummies],axis = 1)
test_data_after_clean.shape


# # XGBoost - Modelling

# In[300]:


xgb_model = xgboost.XGBClassifier(seed = 1, nthread=1)


# In[302]:


kfold = sklearn.model_selection.StratifiedKFold(n_splits=4, random_state=1)
results = sklearn.model_selection.cross_val_score(xgb_model, train_data_after_clean.drop(target_col, axis = 1), train_data_after_clean[target_col[0]], cv=kfold, scoring = 'neg_log_loss')
print("Accuracy: %.2f (%.2f)" % (results.mean()*100, results.std()*100))


# In[303]:


xgb_model.get_params()


# In[69]:


predictors = train_data_after_clean.columns.values[train_data_after_clean.columns.values!=target_col[0]]
predictors


# In[164]:


train_data_after_clean[target_col[0]].sum()/train_data_after_clean.shape[0]


# In[132]:


gridsearch_sample = sklearn.model_selection.GridSearchCV(estimator = xgboost.XGBClassifier(seed = 1
                                                                                           , base_score = train_data_after_clean[target_col[0]].sum()/train_data_after_clean.shape[0]
                                                                                           , nthread=1
                                                                                           , max_depth = 5
                                                                                           , min_child_weight = 1
                                                                                           , gamma = 0
                                                                                           , subsample = 0.8
                                                                                           , colsample_bytree = 0.8
                                                                                           , scale_pos_weight = 1),
                                     param_grid={
                                         'learning_rate':[i/20.0 for i in range(6,0,-1)],
                                         'n_estimators':range(10,100,10)
                                     },         
                                     scoring = 'neg_log_loss',
                                     n_jobs= 1,
                                     iid = False,
                                     cv = 4)
gridsearch_sample.fit(train_data_after_clean[predictors],train_data_after_clean[target_col[0]])
gridsearch_sample.grid_scores_, gridsearch_sample.best_params_, gridsearch_sample.best_score_


# In[137]:


gridsearch_sample_1 = sklearn.model_selection.GridSearchCV(estimator = xgboost.XGBClassifier(seed = 1
                                                                                           , learning_rate = 0.1
                                                                                           , n_estimators = 70
                                                                                           , base_score = train_data_after_clean[target_col[0]].sum()/train_data_after_clean.shape[0]
                                                                                           , nthread=1
#                                                                                            , max_depth = 5
#                                                                                            , min_child_weight = 1
                                                                                           , gamma = 0
                                                                                           , subsample = 0.8
                                                                                           , colsample_bytree = 0.8
                                                                                           , scale_pos_weight = 1),
                                     param_grid={
                                         'max_depth':range(1,10),
                                         'min_child_weight': [i/10 for i in range(9,12)]
                                     },         
                                     scoring = 'neg_log_loss',
                                     n_jobs= 1,
                                     iid = False,
                                     cv = 4)
gridsearch_sample_1.fit(train_data_after_clean[predictors],train_data_after_clean[target_col[0]])
gridsearch_sample_1.grid_scores_, gridsearch_sample_1.best_params_, gridsearch_sample_1.best_score_


# In[139]:


gridsearch_sample_1 = sklearn.model_selection.GridSearchCV(estimator = xgboost.XGBClassifier(seed = 1
                                                                                           , learning_rate = 0.1
                                                                                           , n_estimators = 70
                                                                                           , base_score = train_data_after_clean[target_col[0]].sum()/train_data_after_clean.shape[0]
                                                                                           , nthread=1
                                                                                           , max_depth = 5
                                                                                           , min_child_weight = 0.9
#                                                                                            , gamma = 0
                                                                                           , subsample = 0.8
                                                                                           , colsample_bytree = 0.8
                                                                                           , scale_pos_weight = 1),
                                     param_grid={
                                         'gamma': [i/10.0 for i in range(0,11)]
                                     },         
                                     scoring = 'neg_log_loss',
                                     n_jobs= 1,
                                     iid = False,
                                     cv = 4)
gridsearch_sample_1.fit(train_data_after_clean[predictors],train_data_after_clean[target_col[0]])
gridsearch_sample_1.grid_scores_, gridsearch_sample_1.best_params_, gridsearch_sample_1.best_score_


# In[141]:


gridsearch_sample_1 = sklearn.model_selection.GridSearchCV(estimator = xgboost.XGBClassifier(seed = 1
                                                                                           , learning_rate = 0.1
                                                                                           , n_estimators = 80
                                                                                           , base_score = train_data_after_clean[target_col[0]].sum()/train_data_after_clean.shape[0]
                                                                                           , nthread=1
                                                                                           , max_depth = 5
                                                                                           , min_child_weight = 0.9
                                                                                           , gamma = 0.7
                                                                                           , subsample = 0.8
                                                                                           , colsample_bytree = 0.8
                                                                                           , scale_pos_weight = 1),
                                     param_grid={
                                         'n_estimators': range(50,100,5),
                                         'learning_rate':[i/20.0 for i in range(6,0,-1)],
                                     },         
                                     scoring = 'neg_log_loss',
                                     n_jobs= 1,
                                     iid = False,
                                     cv = 4)
gridsearch_sample_1.fit(train_data_after_clean[predictors],train_data_after_clean[target_col[0]])
gridsearch_sample_1.grid_scores_, gridsearch_sample_1.best_params_, gridsearch_sample_1.best_score_


# In[144]:


gridsearch_sample_1 = sklearn.model_selection.GridSearchCV(estimator = xgboost.XGBClassifier(seed = 1
                                                                                           , learning_rate = 0.1
                                                                                           , n_estimators = 75
                                                                                           , base_score = train_data_after_clean[target_col[0]].sum()/train_data_after_clean.shape[0]
                                                                                           , nthread=1
                                                                                           , max_depth = 5
                                                                                           , min_child_weight = 0.9
                                                                                           , gamma = 0.7
#                                                                                            , subsample = 0.8
#                                                                                            , colsample_bytree = 0.8
                                                                                           , scale_pos_weight = 1),
                                     param_grid={
                                         'subsample': [i/20.0 for i in range(4,20)],
                                         'colsample_bytree': [i/20.0 for i in range(8,21)]
                                     },         
                                     scoring = 'neg_log_loss',
                                     n_jobs= 1,
                                     iid = False,
                                     cv = 4)
gridsearch_sample_1.fit(train_data_after_clean[predictors],train_data_after_clean[target_col[0]])
gridsearch_sample_1.grid_scores_, gridsearch_sample_1.best_params_, gridsearch_sample_1.best_score_


# In[149]:


gridsearch_sample_1 = sklearn.model_selection.GridSearchCV(estimator = xgboost.XGBClassifier(seed = 1
                                                                                           , learning_rate = 0.1
                                                                                           , n_estimators = 75
                                                                                           , base_score = train_data_after_clean[target_col[0]].sum()/train_data_after_clean.shape[0]
                                                                                           , nthread=1
                                                                                           , max_depth = 5
                                                                                           , min_child_weight = 0.9
                                                                                           , gamma = 0.7
                                                                                           , subsample = 0.6
                                                                                           , colsample_bytree = 0.9
                                                                                           , scale_pos_weight = 1),
                                     param_grid={
                                         'reg_alpha':[1e-10,1e-8,1e-6,1e-5,1e-4,1e-3,1e-2, 0.1, 1, 100]
                                     },         
                                     scoring = 'neg_log_loss',
                                     n_jobs= 1,
                                     iid = False,
                                     cv = 4)
gridsearch_sample_1.fit(train_data_after_clean[predictors],train_data_after_clean[target_col[0]])
gridsearch_sample_1.grid_scores_, gridsearch_sample_1.best_params_, gridsearch_sample_1.best_score_


# In[152]:


gridsearch_sample_1 = sklearn.model_selection.GridSearchCV(estimator = xgboost.XGBClassifier(seed = 1
                                                                                           , learning_rate = 0.01
                                                                                           , n_estimators = 750
                                                                                           , base_score = train_data_after_clean[target_col[0]].sum()/train_data_after_clean.shape[0]
                                                                                           , nthread=1
                                                                                           , max_depth = 5
                                                                                           , min_child_weight = 0.9
                                                                                           , gamma = 0.7
                                                                                           , subsample = 0.6
                                                                                           , colsample_bytree = 0.9
                                                                                           , scale_pos_weight = 1
                                                                                           , reg_alpha = 0.001),
                                     param_grid={
                                         'n_estimators':range(600,710,10)
                                     },         
                                     scoring = 'neg_log_loss',
                                     n_jobs= 1,
                                     iid = False,
                                     cv = 4)
gridsearch_sample_1.fit(train_data_after_clean[predictors],train_data_after_clean[target_col[0]])
gridsearch_sample_1.grid_scores_, gridsearch_sample_1.best_params_, gridsearch_sample_1.best_score_


# In[153]:


xgb_model_tuned = xgboost.XGBClassifier(seed = 1
                      , learning_rate = 0.01
                      , n_estimators = 750
                      , base_score = train_data_after_clean[target_col[0]].sum()/train_data_after_clean.shape[0]
                      , nthread=1
                      , max_depth = 5
                      , min_child_weight = 0.9
                      , gamma = 0.7
                      , subsample = 0.6
                      , colsample_bytree = 0.9
                      , scale_pos_weight = 1
                      , reg_alpha = 0.001)


# In[163]:


kfold = sklearn.model_selection.StratifiedKFold(n_splits=4, random_state=1)
results = sklearn.model_selection.cross_val_score(xgb_model_tuned, train_data_after_clean.drop(target_col, axis = 1), train_data_after_clean[target_col[0]], cv=kfold,scoring='neg_log_loss')
# print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
print("Log Loss: %.3f (%.2f)" % (results.mean(), results.std()))


# In[166]:


fitted_xgb_tuned = xgb_model_tuned.fit(train_data_after_clean.drop(target_col, axis = 1),train_data_after_clean[target_col[0]])


# In[167]:


predicted_prob = fitted_xgb_tuned.predict_proba(train_data_after_clean.drop(target_col, axis = 1))[:,1]


# In[169]:


print(predicted_prob.mean())
print(predicted_prob.max())
print(predicted_prob.min())


# In[181]:


final_df = pd.concat([pd.Series(predicted_prob),train_data_after_clean[target_col[0]]],axis = 1)
final_df.columns = ['predicted_prob','actuals']


# In[182]:


final_df.head()


# In[183]:


fpr, tpr, thresholds = sklearn.metrics.roc_curve(train_data_after_clean[target_col[0]],predicted_prob)


# In[184]:


fpr


# In[185]:


tpr


# In[188]:


cut_off = thresholds[np.argmax(tpr - fpr)]


# In[189]:


final_df['Prediction_Flag'] = final_df['predicted_prob'].map(lambda x: 1 if x > cut_off else 0)


# In[191]:


sklearn.metrics.accuracy_score(final_df['actuals'],final_df['Prediction_Flag'])


# In[192]:


sklearn.metrics.confusion_matrix(final_df['actuals'],final_df['Prediction_Flag'])


# In[194]:


tn, fp, fn, tp = sklearn.metrics.confusion_matrix(final_df['actuals'],final_df['Prediction_Flag']).ravel()


# In[279]:


predicted_prob_test = fitted_xgb_tuned.predict_proba(test_data_after_clean.drop('id', axis = 1))[:,1]


# In[298]:


predictions_df = pd.concat([test_data['id'],pd.Series(predicted_prob_test).map(lambda x: 1 if x > cut_off else 0)], axis = 1)
predictions_df.columns = ['id','P']
test_data.shape[0] == predictions_df.shape[0]


# In[299]:


predictions_df.to_csv('predictions.csv',index = False)

