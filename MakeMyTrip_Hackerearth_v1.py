
# coding: utf-8

# # Importing Required Packages

# In[1]:


import os
import pandas as pd
import numpy as np
import xgboost
import sklearn
import matplotlib.pyplot as plt


# # Setting the working directory

# In[2]:


os.chdir(r'C:\Users\Rithesh\Desktop\MMT_Hackerearth')


# # Importing the Train and Test Files

# In[3]:


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


# In[4]:


(train_data['id'].reset_index(drop = False)['id'] - train_data['id'].reset_index(drop = False)['index']).sum()


# In[5]:


(test_data['id'].reset_index(drop = False)['id'] - test_data['id'].reset_index(drop = False)['index'] - 552).sum()


# In[6]:


test_data['id'].reset_index(drop = False).head()


# # Data Exploration

# ## Column Types: Continuous, Categorical and Target

# In[7]:


col_types = train_data.dtypes
col_types


# In[8]:


categorical_cols = col_types[col_types=='object'].index.values
categorical_cols


# In[9]:


continuous_cols = col_types[col_types!='object'].index.values
continuous_cols = continuous_cols[[i not in ['P','id'] for i in continuous_cols]]
continuous_cols


# In[10]:


target_col = ['P']
target_col


# # Column Exploration

# In[11]:


train_data[continuous_cols].describe()


# In[12]:


test_data[continuous_cols].describe()


# In[13]:


train_data[categorical_cols].apply(pd.Series.nunique)


# In[14]:


test_data[categorical_cols].apply(pd.Series.nunique)


# ## Missing values exploration

# In[15]:


train_data[continuous_cols].isnull().sum(axis = 0)


# In[16]:


train_data[categorical_cols].isnull().sum(axis = 0)


# In[17]:


test_data[continuous_cols].isnull().sum(axis = 0)


# In[18]:


test_data[categorical_cols].isnull().sum(axis = 0)


# # Plots

# ## Continuous vs Target Variable

# In[19]:


for i in continuous_cols:
    train_data[[i,target_col[0]]].boxplot(by = target_col[0])


# # Data Preparation - Train Data

# In[20]:


categorical_append_train_test = pd.concat([train_data[np.append(categorical_cols,['id'])],test_data[np.append(categorical_cols,['id'])]],axis = 0).reset_index(drop = True)


# In[21]:


dummies_categorical_append_train_test = pd.concat([pd.get_dummies(categorical_append_train_test.drop('id',axis = 1)),categorical_append_train_test['id']], axis = 1)


# In[22]:


dummies_categorical_append_train_test.loc[dummies_categorical_append_train_test['id']<=552,:].drop('id',axis = 1).shape


# ## Creating Dummy Variables for categorical cols

# In[23]:


train_data_after_dummies = dummies_categorical_append_train_test.loc[dummies_categorical_append_train_test['id']<=552,:].drop('id',axis = 1)
train_data_after_dummies.describe()


# ## Outliers Manipulation

# In[25]:


train_data[continuous_cols].describe()


# In[26]:


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


# In[39]:


def rm_outliers_1(pd_series):
    quantile_95 = pd_series.quantile(0.95)
#     quantile_25 = pd_series.quantile(0.25)
#     iqr = 1.5*(quantile_75-quantile_25)
    conditions = [pd_series > quantile_95]
    choices = [quantile_95]
    pd_series_after = pd.Series(np.select(conditions,choices, default=pd_series))
    return pd_series_after


# In[41]:


train_data_after_outliers = train_data[continuous_cols[continuous_cols!='O']].apply(rm_outliers)
train_data_after_outliers['O'] = rm_outliers_1(train_data['O'])


# In[42]:


train_data_after_outliers.describe()


# ## Missing Values Interpretation

# In[43]:


train_data_after_nans =train_data_after_outliers.fillna(train_data_after_outliers.mean())
train_data_after_nans.describe()


# ## Normalization

# In[44]:


def normalization(pd_series):
    pd_series_new = (pd_series - pd_series.mean())/pd_series.std()
    return pd_series_new


# In[45]:


train_data_after_normalization = train_data_after_nans.apply(normalization)
train_data_after_normalization.describe()


# ## Final Train Dataset after Cleaning

# In[46]:


train_data_after_clean = pd.concat([train_data_after_normalization,train_data_after_dummies,train_data[target_col]],axis = 1)


# In[47]:


for i in continuous_cols:
    train_data_after_clean[[i,target_col[0]]].boxplot(by = target_col[0])


# # Data Preparation - Test Data

# ## Creating Dummy Variables for categorical cols

# In[48]:


test_data_after_dummies = dummies_categorical_append_train_test.loc[dummies_categorical_append_train_test['id']>552,:].drop('id',axis = 1).reset_index(drop = True)
test_data_after_dummies.describe()


# ## Missing Values Interpretation

# In[49]:


test_data[continuous_cols].describe()


# In[50]:


test_data_after_nans =test_data[continuous_cols].fillna(train_data_after_outliers.mean())
test_data_after_nans.describe()


# ## Normalization

# In[51]:


def normalization_using_train(pd_series):
    name_series = pd_series.name
    pd_series_new = (pd_series - train_data_after_nans[name_series].mean())/train_data_after_nans[name_series].std()
    return pd_series_new


# In[52]:


test_data_after_normalization = test_data_after_nans.apply(normalization_using_train)
test_data_after_normalization.describe()


# ## Final Test Dataset after Cleaning

# In[53]:


test_data_after_clean = pd.concat([test_data['id'],test_data_after_normalization,test_data_after_dummies],axis = 1)
test_data_after_clean.shape


# # XGBoost - Modelling

# In[54]:


xgb_model = xgboost.XGBClassifier(seed = 1, nthread=1)


# In[57]:


kfold = sklearn.model_selection.StratifiedKFold(n_splits=4, random_state=1)
results = sklearn.model_selection.cross_val_score(xgb_model, train_data_after_clean.drop(target_col, axis = 1), train_data_after_clean[target_col[0]], cv=kfold, scoring = 'neg_log_loss')
print("Accuracy: %.2f (%.2f)" % (results.mean(), results.std()))


# In[58]:


xgb_model.get_params()


# In[59]:


predictors = train_data_after_clean.columns.values[train_data_after_clean.columns.values!=target_col[0]]
predictors


# In[60]:


train_data_after_clean[target_col[0]].sum()/train_data_after_clean.shape[0]


# In[61]:


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


# In[63]:


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
                                         'min_child_weight': [i/10 for i in range(5,12)]
                                     },         
                                     scoring = 'neg_log_loss',
                                     n_jobs= 1,
                                     iid = False,
                                     cv = 4)
gridsearch_sample_1.fit(train_data_after_clean[predictors],train_data_after_clean[target_col[0]])
gridsearch_sample_1.grid_scores_, gridsearch_sample_1.best_params_, gridsearch_sample_1.best_score_


# In[64]:


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


# In[65]:


gridsearch_sample_1 = sklearn.model_selection.GridSearchCV(estimator = xgboost.XGBClassifier(seed = 1
                                                                                           , learning_rate = 0.1
                                                                                           , n_estimators = 80
                                                                                           , base_score = train_data_after_clean[target_col[0]].sum()/train_data_after_clean.shape[0]
                                                                                           , nthread=1
                                                                                           , max_depth = 5
                                                                                           , min_child_weight = 0.9
                                                                                           , gamma = 0
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


# In[66]:


gridsearch_sample_1 = sklearn.model_selection.GridSearchCV(estimator = xgboost.XGBClassifier(seed = 1
                                                                                           , learning_rate = 0.1
                                                                                           , n_estimators = 70
                                                                                           , base_score = train_data_after_clean[target_col[0]].sum()/train_data_after_clean.shape[0]
                                                                                           , nthread=1
                                                                                           , max_depth = 5
                                                                                           , min_child_weight = 0.9
                                                                                           , gamma = 0
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


# In[67]:


gridsearch_sample_1 = sklearn.model_selection.GridSearchCV(estimator = xgboost.XGBClassifier(seed = 1
                                                                                           , learning_rate = 0.1
                                                                                           , n_estimators = 70
                                                                                           , base_score = train_data_after_clean[target_col[0]].sum()/train_data_after_clean.shape[0]
                                                                                           , nthread=1
                                                                                           , max_depth = 5
                                                                                           , min_child_weight = 0.9
                                                                                           , gamma = 0
                                                                                           , subsample = 0.8
                                                                                           , colsample_bytree = 0.8
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


# In[68]:


gridsearch_sample_1 = sklearn.model_selection.GridSearchCV(estimator = xgboost.XGBClassifier(seed = 1
                                                                                           , learning_rate = 0.01
                                                                                           , n_estimators = 750
                                                                                           , base_score = train_data_after_clean[target_col[0]].sum()/train_data_after_clean.shape[0]
                                                                                           , nthread=1
                                                                                           , max_depth = 5
                                                                                           , min_child_weight = 0.9
                                                                                           , gamma = 0
                                                                                           , subsample = 0.8
                                                                                           , colsample_bytree = 0.8
                                                                                           , scale_pos_weight = 1
                                                                                           , reg_alpha = 0.0001),
                                     param_grid={
                                         'n_estimators':range(600,710,10)
                                     },         
                                     scoring = 'neg_log_loss',
                                     n_jobs= 1,
                                     iid = False,
                                     cv = 4)
gridsearch_sample_1.fit(train_data_after_clean[predictors],train_data_after_clean[target_col[0]])
gridsearch_sample_1.grid_scores_, gridsearch_sample_1.best_params_, gridsearch_sample_1.best_score_


# In[69]:


xgb_model_tuned = xgboost.XGBClassifier(seed = 1
                      , learning_rate = 0.01
                      , n_estimators = 620
                      , base_score = train_data_after_clean[target_col[0]].sum()/train_data_after_clean.shape[0]
                      , nthread=1
                      , max_depth = 5
                      , min_child_weight = 0.9
                      , gamma = 0
                      , subsample = 0.8
                      , colsample_bytree = 0.8
                      , scale_pos_weight = 1
                      , reg_alpha = 0.0001)


# In[70]:


kfold = sklearn.model_selection.StratifiedKFold(n_splits=4, random_state=1)
results = sklearn.model_selection.cross_val_score(xgb_model_tuned, train_data_after_clean.drop(target_col, axis = 1), train_data_after_clean[target_col[0]], cv=kfold,scoring='neg_log_loss')
# print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
print("Log Loss: %.3f (%.2f)" % (results.mean(), results.std()))


# In[71]:


fitted_xgb_tuned = xgb_model_tuned.fit(train_data_after_clean.drop(target_col, axis = 1),train_data_after_clean[target_col[0]])


# In[72]:


predicted_prob = fitted_xgb_tuned.predict_proba(train_data_after_clean.drop(target_col, axis = 1))[:,1]


# In[73]:


print(predicted_prob.mean())
print(predicted_prob.max())
print(predicted_prob.min())


# In[74]:


final_df = pd.concat([pd.Series(predicted_prob),train_data_after_clean[target_col[0]]],axis = 1)
final_df.columns = ['predicted_prob','actuals']


# In[75]:


final_df.head()


# In[76]:


fpr, tpr, thresholds = sklearn.metrics.roc_curve(train_data_after_clean[target_col[0]],predicted_prob)


# In[77]:


fpr


# In[78]:


tpr


# In[79]:


cut_off = thresholds[np.argmax(tpr - fpr)]


# In[80]:


final_df['Prediction_Flag'] = final_df['predicted_prob'].map(lambda x: 1 if x > cut_off else 0)


# In[81]:


sklearn.metrics.accuracy_score(final_df['actuals'],final_df['Prediction_Flag'])


# In[82]:


sklearn.metrics.confusion_matrix(final_df['actuals'],final_df['Prediction_Flag'])


# In[83]:


tn, fp, fn, tp = sklearn.metrics.confusion_matrix(final_df['actuals'],final_df['Prediction_Flag']).ravel()


# In[84]:


predicted_prob_test = fitted_xgb_tuned.predict_proba(test_data_after_clean.drop('id', axis = 1))[:,1]


# In[85]:


predictions_df = pd.concat([test_data['id'],pd.Series(predicted_prob_test).map(lambda x: 1 if x > cut_off else 0)], axis = 1)
predictions_df.columns = ['id','P']
test_data.shape[0] == predictions_df.shape[0]


# In[86]:


predictions_df.to_csv('predictions_v1.csv',index = False)

