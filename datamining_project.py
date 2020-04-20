#%%
import numpy as np
import pandas as pd

#%%
# Standard quick checks
def dfChkBasics(dframe, valCnt = False): 
  cnt = 1
  print('\ndataframe Basic Check function -')
  
  try:
    print(f'\n{cnt}: info(): ')
    cnt+=1
    print(dframe.info())
  except: pass

  print(f'\n{cnt}: describe(): ')
  cnt+=1
  print(dframe.describe())

  print(f'\n{cnt}: dtypes: ')
  cnt+=1
  print(dframe.dtypes)

  try:
    print(f'\n{cnt}: columns: ')
    cnt+=1
    print(dframe.columns)
  except: pass

  print(f'\n{cnt}: head() -- ')
  cnt+=1
  print(dframe.head())

  print(f'\n{cnt}: shape: ')
  cnt+=1
  print(dframe.shape)

  if (valCnt):
    print('\nValue Counts for each feature -')
    for colname in dframe.columns :
      print(f'\n{cnt}: {colname} value_counts(): ')
      print(dframe[colname].value_counts())
      cnt +=1

# examples:
# dfChkBasics(df)


#%% [markdown]
import os
import pandas as pd
#os.chdir('../dataminingproject')  
# sometime when I opened the workspace from another folder, the 
# working directory getcwd() will be in the wrong place. 
# You can change it with chdir()
dirpath = os.getcwd() # print("current directory is : " + dirpath)
filepath = os.path.join( dirpath ,'cepr_org_2019.csv')
cols_list=['age','female','wbho','forborn','citizen','vet','married', 'marstat','ownchild','empl','unem','nilf','uncov','state','educ','centcity','suburb','rural','smsastat14','ind_m03','agric','manuf', 'hourslw','rw', 'multjobn']
df= pd.read_csv(filepath, usecols=cols_list)
dfChkBasics(df, True)
print(df.dtypes)

# Data dict
# age - age (Numeric)
# female - sex (0 = male, 1 = female)
# wbho - Race (white, Hispanic, Black, Other)
# forborn - Foreign born (0 = foriegn born, 1 = US born)
# citizen - US citizen (0 = No-US citizen, 1 = US citizen)
# vet - Veteran (0 = No-vateren, 1 = veteran)
# married - Married (0 = Never married, 1 = married)
# marstat - Marital status (Married, Never Married, Divorced, Widowed, Separated)
# ownchild - Number of children (Numeric)
# empl - Employed (0 = employed, 1 = unemployed)
# unem - Unemployed (0 = employed, 1 = unemployed)
# nilf - Not in labor force (0 = Not in labor force, 1 = in labor force)
# uncov - Union coverage (0 - non-Union coverage, 1 = Union coverage)
# state - state (50 states)
# educ - Education level (HS, Some college, College, Advanced, LTHS)
# centcity - Central city (0 = no Central city, 1 = Central city)
# suburb - suburbs (0 = no suburbs area, 1 = suburbs area)
# rural - rural (0 = no rural area, 1 = rural area)
# smsastat14 - Metro CBSA FIPS Code
# ind_m03 - Major Industry Recode (Educational and health services, Wholesale and retail trade, Professional and business services, Manufacturing, Leisure and hospitality, Construction, Financial activities, Other  services, Transportation and utilities, Public administration, Agriculture, forestry, fishing, and hunting, Information, Mining, and Armed Forces)
# agric - Agriculture (0 = Non-Argiculture job, 1 = Non-Agriculture job)
# manuf - Manufacturing (0 = Non-Manufacturing job, 1 = Non-Manufacturing job)
# hourslw - Hours last week, all jobs (Numeric)
# rw - Real hourly wage, 2019$ (Numeric)
# multjobn - Number of jobs (Numeric)


# %%
# clean numuric data (3 variables)
# Step 1 - plot the boxplot to investigate the outlier in "age", "rw", and "houtslw" (Numuric)
import matplotlib.pyplot as plt 
import numpy as np
boxplot = df.boxplot(column=['age', 'rw', 'hourslw'])
dfnew = df[['age', 'rw', 'hourslw']]
df.drop(columns=['age', 'rw', 'hourslw'])

# According to the plot, there are a little bit outlier in age. In addition, rw and hourslw have so many outlier, so we need to remove them out. 

#%%
#Step 2 - remove outlier by calculated the interquartile range (IQR). 
# IQR is calculated as the difference between the 75th and 25th percentiles or IQR = Q3 − Q1. 
# So, the numbers that are out of IQR are outlier. 
Q1 = dfnew.quantile(0.25)
Q3 = dfnew.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

df_new = dfnew[~((dfnew < (Q1 - 1.5 * IQR)) |(dfnew > (Q3 + 1.5 * IQR))).any(axis=1)]
print(df_new.shape)


# %%
# Step 3
# merge the data (3 numuric) without outlier to original dataset by index of both the dataframes.
cleaned_df = df.merge(df_new, left_index=True, right_index=True).dropna()

# reset index for new dataset
cleaned_df = cleaned_df.reset_index()

#rename age_x, rw_x, hourlw_x to age, rw, and hourlw, and drop age_y, rw_y, and hourlw_y
cleaned_df = cleaned_df.rename(columns={"age_x": "age", "rw_x": "rw", "hourslw_x": "hourslw"})
cleaned_df = cleaned_df.drop(columns=['age_y', 'rw_y', 'hourslw_y','index'])

#Cleaned_data
dfChkBasics(cleaned_df, True)
print(cleaned_df.dtypes)

#%%
#data exploration and graphing
#table on quantitative variables
quant_variables=cleaned_df[['age','ownchild','multjobn','hourslw','rw']]
table=quant_variables.describe()
table.columns=['Age','Number of Children','Number of Jobs','Hours Worked Last Week','Real Wage']
print(table)

#%%
#age and real wage
import seaborn as sns
kde=sns.kdeplot(cleaned_df.age, cleaned_df.rw, cmap="Blues", shade=True)
kde.set_xlabel('Age',fontsize=10)
kde.set_ylabel('Real Wage',fontsize=10)
kde.set_title('Age and Real Wage',fontsize=15)

# %%
#gender by education level
countplot=sns.countplot(x='educ', hue='female', order=['HS','LTHS','Some college','College','Advanced'],data=cleaned_df,palette=['skyblue','lightsalmon'])
leg = countplot.get_legend()
leg.set_title("Gender")
labs = leg.texts
labs[0].set_text("Male")
labs[1].set_text("Female")
countplot.set_xlabel('Education Level',fontsize=10)
countplot.set_ylabel('Count',fontsize=10)
countplot.set_title('Education Level by Gender',fontsize=15)

# %%
#education level by race
fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(12, 6))
plt.suptitle('Education Level by Race',fontsize=20)
plt.subplots_adjust(hspace=.4)
white = cleaned_df[cleaned_df['wbho']=='White']
hispanic = cleaned_df[cleaned_df['wbho']=='Hispanic']
black = cleaned_df[cleaned_df['wbho']=='Black']
other = cleaned_df[cleaned_df['wbho']=='Other']

ax = sns.countplot(x='educ',order=['HS','LTHS','Some college','College','Advanced'],data=white,ax = axes[0,0],palette='BuGn_r')
ax.set_xlabel('')
ax.set_title('White')
ax = sns.countplot(x='educ',order=['HS','LTHS','Some college','College','Advanced'],data=hispanic,ax = axes[0,1],palette='BuGn_r')
ax.set_xlabel('')
ax.set_title('Hispanic')
ax = sns.countplot(x='educ',order=['HS','LTHS','Some college','College','Advanced'],data=black,ax = axes[1,0],palette='BuGn_r')
ax.set_xlabel('')
ax.set_title('Black')
ax = sns.countplot(x='educ',order=['HS','LTHS','Some college','College','Advanced'],data=other,ax = axes[1,1],palette='BuGn_r')
ax.set_xlabel('')
ax.set_title('Other')
#%%
#types of region
cleaned_df.centcity.sum()
cleaned_df.suburb.sum()
cleaned_df.rural.sum()
data = [['Central City', 2297], ['Suburban', 4385], ['Rural', 1893]]
regions = pd.DataFrame(data, columns = ['Type of Region', 'Count']) 
regions_ch=sns.barplot(x='Type of Region',y='Count',data=regions,palette='Greens')


# %%
# Build the model
# Mutiple regression

# age - age (Numeric)
# female - sex (0 = male, 1 = female)
# wbho - Race (white, Hispanic, Black, Other)
# forborn - Foreign born (0 = foriegn born, 1 = US born)
# citizen - US citizen (0 = No-US citizen, 1 = US citizen)
# vet - Veteran (0 = No-vateren, 1 = veteran)
# married - Married (0 = Never married, 1 = married)
# marstat - Marital status (Married, Never Married, Divorced, Widowed, Separated)
# ownchild - Number of children (Numeric)
# empl - Employed (1 = employed, 0 = unemployed or not in LF)
# unem - Unemployed (0 = employed, 1 = unemployed)
# nilf - Not in labor force (0 = Not in labor force, 1 = in labor force)
# uncov - Union coverage (0 - non-Union coverage, 1 = Union coverage)
# state - state (50 states)
# educ - Education level (HS, Some college, College, Advanced, LTHS)
# centcity - Central city (0 = no Central city, 1 = Central city)
# suburb - suburbs (0 = no suburbs area, 1 = suburbs area)
# rural - rural (0 = no rural area, 1 = rural area)
# smsastat14 - Metro CBSA FIPS Code
# ind_m03 - Major Industry Recode (Educational and health services, Wholesale and retail trade, Professional and business services, Manufacturing, Leisure and hospitality, Construction, Financial activities, Other  services, Transportation and utilities, Public administration, Agriculture, forestry, fishing, and hunting, Information, Mining, and Armed Forces)
# agric - Agriculture (0 = Non-Argiculture job, 1 = Non-Agriculture job)
# manuf - Manufacturing (0 = Non-Manufacturing job, 1 = Non-Manufacturing job)
# hourslw - Hours last week, all jobs (Numeric)
# rw - Real hourly wage, 2019$ (Numeric)
# multjobn - Number of jobs (Numeric)

from statsmodels.formula.api import ols
modelwage1Fit = ols(formula='rw ~ age + C(female) + hourslw + C(forborn) + C(married) + C(educ) + C(wbho) + ownchild', data=cleaned_df).fit()

print( type(modelwage1Fit) )
print( modelwage1Fit.summary() )

modelpredicitons = pd.DataFrame( columns=['modelwage1_LM'], data= modelwage1Fit.predict(cleaned_df)) 
# use the original dataset gpa data to find the expected model values
print(modelpredicitons.shape)
print( modelpredicitons.head() )

# All of them were significant.

# Note: citizen and uncov were not significant, so both of them were dropped from the model

#%%
# Check the VIF value (watch out for multicollinearity issues)
from statsmodels.stats.outliers_influence import variance_inflation_factor

# need to convert educ and wbho to numeric by function because VIF calculation requires numeric variables.
# Educ 
# HS = 0, Some college = 1, college = 2, advanced = 3, LTHS = 4
def cleanDfeduc(row):
  theedu = row["educ"]
  return (0 if (theedu=="HS") else 1 if (theedu=="Some college") else 2 if (theedu=="College") else 3 if (theedu=="Advanced") else 4 if (theedu=="LTHS") else np.nan)
# end function cleanDfeduc
cleaned_df['educ'] = df.apply(cleanDfeduc, axis=1)

# wbho
# White = 0, Hispanic = 1, black = 2, other = 3
def cleanDfwbho(row):
  thewbho = row["wbho"]
  return (0 if (thewbho=="White") else 1 if (thewbho=="Hispanic") else 2 if (thewbho=="Black") else 3 if (thewbho=="Other") else np.nan)
# end function cleanDfeduc
cleaned_df['wbho'] = df.apply(cleanDfwbho, axis=1)

# Get variables for which to compute VIF and add intercept term
X = cleaned_df[['age' , 'female' , 'hourslw', 'forborn', 'married', 'ownchild', 'wbho', 'educ']]
X['Intercept'] = 1

# Compute and view VIF
vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [ variance_inflation_factor(X.values, i) for i in range(X.shape[1]) ] # list comprehension

# View results using print
print(vif)
# According to the results, VIF is very small in all independent variables, so it means there are not multicollinearity issues for this model

# %%
#logistic regression to predict someone's gender based on their responses
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#choosing variables
logit_df=cleaned_df[['uncov','age','female','wbho','citizen','vet','multjobn','rw']]
#making a 4:1 train/test split
X_train, X_test, y_train, y_test = train_test_split(logit_df.drop('female',axis=1), logit_df['female'], test_size=0.20, random_state=101)
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
#results
confusion_matrix = confusion_matrix(y_test, predictions)
print(confusion_matrix)
print(classification_report(y_test,predictions))
logmodel.score(X_test, y_test)
#%%
#trying using cv in addition to linear regression
cv_model=LogisticRegressionCV()
cv_model.fit(X_train,y_train)
cv_predictions = cv_model.predict(X_test)
#results
print(classification_report(y_test,cv_predictions))
cv_model.score(X_test, y_test)


# %%
# Running Wage model by the six classifiers
# prepare data
#logistic regression to predict wage based on their responses
# Convert wage to be object by given 0 = low wage (lower than the mean = 23.48), 1 = high wage (higher than the mean = 23.48)
def cleanDfwage(row):
  thewage = row["rw"]
  return ("1" if (thewage >= 23.48) else "0" if (thewage < 23.48) else np.nan)
# end function cleanDfwage
cleaned_df['rw_dummy'] = cleaned_df.apply(cleanDfwage, axis=1)

xtarget = cleaned_df[['age', 'female', 'citizen', 'married', 'educ', 'wbho', 'hourslw']]
ytarget = cleaned_df['rw_dummy'] #wage 

print(type(xtarget))
print(type(ytarget))

#make a train-test split in 4:1 ratio. 
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
x_trainwage, x_testwage, y_trainwage, y_testwage = train_test_split(xtarget, ytarget, test_size = 0.2, random_state=1)

print('x_trainTitanic type',type(x_trainwage))
print('x_trainTitanic shape',x_trainwage.shape)
print('x_testTitanic type',type(x_testwage))
print('x_testTitanic shape',x_testwage.shape)
print('y_trainTitanic type',type(y_trainwage))
print('y_trainTitanic shape',y_trainwage.shape)
print('y_testTitanic type',type(y_testwage))
print('y_testTitanic shape',y_testwage.shape)

#%%
# first, run the logistic regression for wage model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm 
from statsmodels.formula.api import glm

#Original logistic regression
Original_WageModelLogitFit = glm(formula='rw ~ age + C(female) + C(citizen) + C(married) + C(educ) + C(wbho) + ownchild + C(vet)', data=cleaned_df, family=sm.families.Binomial()).fit()
print( Original_WageModelLogitFit.summary() )

# logistic regression model for wage with the train set, and score it with the test set.
sklearn_wageModellogit = LogisticRegression()  # instantiate
sklearn_wageModellogit .fit(x_trainwage, y_train)
print('Logit model accuracy (with the test set):', sklearn_wageModellogit.score(x_testwage, y_test))

sklearn_wageModellogit_predictions = sklearn_wageModellogit.predict(x_testwage)
#print(sklearn_wageModellogit_predictions)
print(sklearn_wageModellogit.predict_proba(x_trainwage[:8]))
print(sklearn_wageModellogit.predict_proba(x_testwage[:8]))

#results (Logit)
print('Logit model accuracy (with the test set):', sklearn_wageModellogit.score(x_testwage, y_testwage))
confusion_matrix = confusion_matrix(y_testwage, sklearn_wageModellogit_predictions)
print(confusion_matrix)
print(classification_report(y_testwage,sklearn_wageModellogit_predictions))

#%%
#trying using cv for wage model in addition to linear regression
cv_wagemodel=LogisticRegressionCV()
cv_wagemodel.fit(x_trainwage,y_trainwage)
cv_wagepredictions = cv_model.predict(x_testwage)
#print(cv_wagepredictions)
print(cv_wagemodel.predict_proba(x_trainwage[:8]))
print(cv_wagemodel.predict_proba(x_testwage[:8]))

#results
print(classification_report(y_testwage,cv_wagepredictions))
cv_wagepredictions.score(x_testwage, y_testwage)

#%%
# Second, move to KNN
# let's start with KNN spilt for wage model
# # KNeighborsClassifier()
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# choose k between 1 to 31
k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn_split = KNeighborsClassifier(n_neighbors=k)
    knn_split.fit(x_trainwage,y_trainwage)
    scores = knn_split.score(x_testwage,y_testwage)
    k_scores.append(scores.mean())
# plot to see clearly
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN_split')
plt.ylabel('Accuracy score')
plt.show()

# the comfortable KNN choice (n = 5) 
# from sklearn.neighbors import KNeighborsClassifier
knn_split_5 = KNeighborsClassifier(n_neighbors=5) 
# instantiate with n value given
knn_split_5.fit(x_trainwage,y_trainwage)
# knn_split.score(x_testwage,y_testwage)
knn_wagepredictions = knn_split_5.predict(x_testwage)
print(knn_wagepredictions)
print(knn_split_5.predict_proba(x_trainwage[:8]))
print(knn_split_5.predict_proba(x_testwage[:8]))

# Evaluate test-set accuracy
print("KNN (k value = 5)")
print()
print(f'KNN train score:  {knn_split_5.score(x_trainwage,y_train)}')
print(f'KNN test score:  {knn_split_5.score(x_testwage,y_testwage)}')
print(accuracy_score(y_test, knn_wagepredictions))
print(confusion_matrix(y_test, knn_wagepredictions))
print(classification_report(y_test, knn_wagepredictions))
print() 

#%%
#trying using cv_scale for wage model in addition to KNN
# write the function to plot K value increases, comparing woth CV scores
from sklearn.neighbors import KNeighborsClassifier
import sklearn 
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt 

# choose k between 1 to 31
k_range = range(1, 31)
k_scores = []
xtarget_scale = pd.DataFrame( scale(xtarget) )
ytarget_scale = ytarget.copy()
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, xtarget_scale, ytarget_scale, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())
# plot to see clearly
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

#%%
# the comfortable KNN choice (n = 17) 
# from sklearn.neighbors import KNeighborsClassifier
knn_cvscale_17 = KNeighborsClassifier(n_neighbors=5) 
# instantiate with n value given
knn_cvscale_17.fit(x_trainwage,y_trainwage)
# knn_split.score(x_testwage,y_testwage)
knn_cvscale_17_wagepredictions = knn_cvscale_17.predict(x_testwage)
print(knn_cvscale_17_wagepredictions)
print(knn_cvscale_17.predict_proba(x_trainwage[:8]))
print(knn_cvscale_17.predict_proba(x_testwage[:8]))

# Evaluate test-set accuracy
print("KNN (k value = 17)")
print()
print(f'KNN train score:  {knn_cvscale_17.score(x_trainwage,y_train)}')
print(f'KNN test score:  {knn_cvscale_17.score(x_testwage,y_testwage)}')
print(accuracy_score(y_testwage, knn_cvscale_17_wagepredictions))
print(confusion_matrix(y_testwage, knn_cvscale_17_wagepredictions))
print(classification_report(y_testwage, knn_cvscale_17_wagepredictions))
print() 

# %%
# Third, move to DecisionTreeClassifier() for wage model
# Let try entropy (max_depth = 5)
from sklearn.tree import DecisionTreeClassifier

dtreewage_entropy_5 = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=1)
# Fit dt to the training set
dtreewage_entropy_5.fit(x_trainwage,y_trainwage)
# Predict test set labels
dtreewage_entropy_5_pred = dtreewage_entropy_5.predict(x_testwage)
print(dtreewage_entropy_5_pred)
print(dtreewage_entropy_5.predict_proba(x_trainwage[:8]))
print(dtreewage_entropy_5.predict_proba(x_testwage[:8]))

# Evaluate test-set accuracy
print("DecisionTreeClassifier: entropy(max_depth = 5)")
print()
print(f'dtree train score:  {dtreewage_entropy_5.score(x_trainwage,y_trainwage)}')
print(f'dtree test score:  {dtreewage_entropy_5.score(x_testwage,y_testwage)}')
print(accuracy_score(y_testwage, dtreewage_entropy_5_pred))
print(confusion_matrix(y_testwage, dtreewage_entropy_5_pred))
print(classification_report(y_testwage, dtreewage_entropy_5_pred))
print()

#%%
# Fourth, let try SVC() for wage model
from sklearn.svm import SVC, LinearSVC
# SVC - gamma = auto
svcwage_auto = SVC(gamma='auto')
svcwage_auto.fit(x_trainwage,y_trainwage)

#Predictions
svcwage_auto_pred = svcwage_auto.predict(x_testwage)
print(svcwage_auto_pred)
print(svcwage_auto.predict_proba(x_trainwage[:8]))
print(svcwage_auto.predict_proba(x_testwage[:8]))

# Evaluate test-set accuracy
print("SVC (adjust gamma: auto)")
print()
print(f'svc train score:  {svcwage_auto.score(x_trainwage,y_trainwage)}')
print(f'svc test score:  {svcwage_auto.score(x_testwage,y_testwage)}')
print(accuracy_score(y_testwage, svcwage_auto_pred))
print(confusion_matrix(y_testwage, svcwage_auto_pred))
print(classification_report(y_testwage, svcwage_auto_pred))
print()

#%%
# Fifth, let try SVC(kernel="linear") for wage model
from sklearn.svm import SVC, LinearSVC
from sklearn import svm

svcwage_kernel_linear = svm.SVC(kernel='linear')
svcwage_kernel_linear.fit(x_trainwage,y_trainwage)

#Predictions
y_pred = svcwage_kernel_linear.predict(x_testwage)
print(y_pred)
print(svcwage_kernel_linear.predict_proba(x_trainwage[:8]))
print(svcwage_kernel_linear.predict_proba(x_testwage[:8]))

# Evaluate test-set accuracy
print("SVC (kernel='linear')")
print()
print(f'svc train score:  {svcwage_kernel_linear.score(x_trainwage,y_trainwage)}')
print(f'svc test score:  {svcwage_kernel_linear.score(x_testwage,y_testwage)}')
print(accuracy_score(y_testwage, y_pred))
print(confusion_matrix(y_testwage, y_pred))
print(classification_report(y_testwage, y_pred))
print()

#%%
# Finally, let try LinearSVC() for wage model
svcwage_linearsvc = svm.LinearSVC()
svcwage_linearsvc.fit(x_trainwage,y_trainwage)
y_pred = svcwage_linearsvc.predict(x_testwage)

#Predictions
y_pred = svcwage_kernel_linear.predict(x_testwage)
print(y_pred)
print(svcwage_linearsvc.predict_proba(x_trainwage[:8]))
print(svcwage_linearsvc.predict_proba(x_testwage[:8]))

# Evaluate test-set accuracy
print("SVC LinearSVC()")
print()
print(f'svc train score:  {svcwage_linearsvc.score(x_trainwage,y_trainwage)}')
print(f'svc test score:  {svcwage_linearsvc.score(x_testwage,y_testwage)}')
print(accuracy_score(y_testwage, y_pred))
print(confusion_matrix(y_testwage, y_pred))
print(classification_report(y_testwage, y_pred))
print()

#%%
#Calculate employment rate
emcounts = pd.crosstab(index=df['empl'], columns="count")
emcounts/emcounts.sum()

#%%
#educ level and empl
sns.barplot(x='educ', y='empl', data=df)

#%%
#number of children and empl
axes = sns.factorplot('ownchild','empl', 
                      data=df, aspect = 2.5, )

#%%
#employed, educ level, race and gender
FacetGrid = sns.FacetGrid(df, row='wbho', size=4.5, aspect=1.6)
FacetGrid.map(sns.pointplot, 'educ', 'empl', 'female', hue_order=None, color ="r",order=['HS','LTHS','Some college','College','Advanced'],palette=['blue','pink'])
FacetGrid.add_legend()
FacetGrid.set_titles("Female")

# %%
import statsmodels.api as sm  # Importing statsmodels
from statsmodels.formula.api import glm

modelemplLogitFit = glm(formula= 'empl ~ age + C(female) + C(citizen) + C(married) + C(educ) + C(wbho) + ownchild + C(vet)', data=df, family=sm.families.Binomial()).fit()
print(modelemplLogitFit.summary())


# %%
# HS = 0, Some college = 1, college = 2, advanced = 3, LTHS = 4
def cleanDfeduc(row):
  theedu = row["educ"]
  return (0 if (theedu=="HS") else 1 if (theedu=="Some college") else 2 if (theedu=="College") else 3 if (theedu=="Advanced") else 4 if (theedu=="LTHS") else np.nan)
# end function cleanDfeduc
cleaned_df['educ'] = df.apply(cleanDfeduc, axis=1)

# wbho
# White = 0, Hispanic = 1, black = 2, other = 3
def cleanDfwbho(row):
  thewbho = row["wbho"]
  return (0 if (thewbho=="White") else 1 if (thewbho=="Hispanic") else 2 if (thewbho=="Black") else 3 if (thewbho=="Other") else np.nan)
# end function cleanDfeduc
cleaned_df['wbho'] = df.apply(cleanDfwbho, axis=1)
# Prepare our X data (features, predictors, regressors) and y data (target, dependent variable)
xempl = cleaned_df[['age','female','citizen','married','educ','wbho','ownchild','vet']]
yempl = cleaned_df['empl']
print(yempl.head())

# %%
# Decision Tree, y-target is categorical, similar to KNN, (multinomial) logistic Regression, 
# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# Import train_test_split
from sklearn.model_selection import train_test_split
# Import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report

# Split dataset into 80% train, 20% test
X_trainempl, X_testempl, y_trainempl, y_testempl= train_test_split(xempl, yempl, test_size=0.2,random_state=1)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
#%%
# Instantiate dtree
dtree_empl = DecisionTreeClassifier(max_depth=5, random_state=1)
# Fit dt to the training set
dtree_empl.fit(X_trainempl,y_trainempl)
# Predict test set labels
y_predempl = dtree_empl.predict(X_testempl)
# Evaluate test-set accuracy
print(accuracy_score(y_testempl, y_predempl))
print(confusion_matrix(y_testempl, y_predempl))
print(classification_report(y_testempl, y_predempl))

# %%
# SVC
svc = SVC()
svc.fit(X_trainempl,y_trainempl)
print(f'svc train score:  {svc.score(X_trainempl,y_trainempl)}')
print(f'svc test score:  {svc.score(X_testempl,y_testempl)}')
print(confusion_matrix(y_testempl, svc.predict(X_testempl)))
print(classification_report(y_testempl, svc.predict(X_testempl)))

#%%
# SVC(kernel="linear")
svc_linearkernal = SVC(kernel="linear")
svc_linearkernal.fit(X_train,y_train)
print(f'svc_linearkernal train score:  {svc_linearkernal.score(X_train,y_train)}')
print(f'svc_linearkernal test score:  {svc_linearkernal.score(X_test,y_test)}')
print(confusion_matrix(y_test, svc_linearkernal.predict(X_test)))
print(classification_report(y_test, svc_linearkernal.predict(X_test)))

#%%
# linearSVC 
linearSVC = LinearSVC()
linearSVC.fit(X_train,y_train)
print(f'linearSVC train score:  {linearSVC.score(X_train,y_train)}')
print(f'linearSVC test score:  {linearSVC.score(X_test,y_test)}')
print(confusion_matrix(y_test, linearSVC.predict(X_test)))
print(classification_report(y_test, linearSVC.predict(X_test)))

#%% 
# logistic regression 
lr = LogisticRegression()
lr.fit(X_train,y_train)
print(f'lr train score:  {lr.score(X_train,y_train)}')
print(f'lr test score:  {lr.score(X_test,y_test)}')
print(confusion_matrix(y_test, lr.predict(X_test)))
print(classification_report(y_test, lr.predict(X_test)))

#%%
# KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
print(f'knn train score:  {knn.score(X_train,y_train)}')
print(f'knn test score:  {knn.score(X_test,y_test)}')
print(confusion_matrix(y_test, lr.predict(X_test)))
print(classification_report(y_test, knn.predict(X_test)))

#%%
# Decision tree classifier
# Instantiate dtree
dtree = DecisionTreeClassifier(max_depth=5, random_state=1)
# Fit dt to the training set
dtree.fit(X_train,y_train)
# Predict test set labels
y_pred = dtree.predict(X_test)
# Evaluate test-set accuracy
print(f'Decision tree train score:  {dtree.score(X_train,y_train)}')
print(f'Decision tree score:  {dtree.score(X_test,y_test)}')
print(confusion_matrix(y_test, dtree.predict(X_test)))
print(classification_report(y_test, dtree.predict(X_test)))
