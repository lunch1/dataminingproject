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
cols_list=['age','female','wbho','forborn','citizen','vet','married', 'marstat','ownchild','empl','unem','nilf','uncov','state','educ','centcity','suburb','rural', 'hourslw','rw', 'multjobn']
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

#%%
# histrogram 
# age
import seaborn as sns
sns.distplot(df['age'], hist = False, kde = True, 
                kde_kws = {'shade': True, "linewidth": 3},
                 color='blue').set_title('Distribution of age')  
plt.show()   

# rw  
sns.distplot(df['rw'], hist = False, kde = True, 
                kde_kws = {'shade': True, "linewidth": 3},
                 color='green').set_title('Distribution of real wage')  
plt.show()   

# hourlw
sns.distplot(df['hourslw'], hist = False, kde = True, 
                kde_kws = {'shade': True, "linewidth": 3},
                 color='red').set_title('Distribution of Hours last week')  
plt.show()     

# According to the plot, there are a little bit outlier in age. In addition, rw and hourslw have so many outlier, so we need to remove them out. 

#%%
# count missing value 
df.isnull()
df.isnull().sum()

#plot missing value by columm
import numpy as np
import matplotlib.pyplot as plt
 
# Make a missing value dataset 1:
height = [0, 0, 0, 0, 0, 1106, 0]
bars = ('age', 'female', 'wbho', 'forborn', 'citizen', 'vet', 'married')
y_pos = np.arange(len(bars))
# Create bars
plt.bar(y_pos, height)
plt.title("Count of missing value 1 (7 variables)")
plt.xticks(y_pos, bars)
plt.show()

# Make a missing value dataset 2:
height = [0, 22178, 214, 214, 214, 29404, 22741]
bars = ('marstat', 'ownchild', 'empl', 'unem', 'nilf', 'uncov', 'multjobn')
y_pos = np.arange(len(bars))
# Create bars
plt.bar(y_pos, height)
plt.title("Count of missing value 2 (7 variables)")
plt.xticks(y_pos, bars)
plt.show()

# Make a missing value dataset 3:
height = [0, 0, 0, 0, 0, 23780, 26349]
bars = ('state', 'centcity', 'suburb', 'rural', 'educ', 'hourslw', 'rw')
y_pos = np.arange(len(bars))
 
# Create bars
plt.bar(y_pos, height)
plt.title("Count of missing value 3 (7 variables)")
plt.xticks(y_pos, bars)
plt.show()

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
cleaned_df = df.merge(df_new, left_index=True, right_index=True).dropna() #drop missing value

# reset index for new dataset
cleaned_df = cleaned_df.reset_index()

#rename age_x, rw_x, hourlw_x to age, rw, and hourlw, and drop age_y, rw_y, and hourlw_y
cleaned_df = cleaned_df.rename(columns={"age_x": "age", "rw_x": "rw", "hourslw_x": "hourslw"})
cleaned_df = cleaned_df.drop(columns=['age_y', 'rw_y', 'hourslw_y','index'])

#Cleaned_data
dfChkBasics(cleaned_df, True)
print(cleaned_df.dtypes) #ready to do next part

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
#logistic regression to predict someone's gender based on their responses
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

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
# hourslw - Hours last week, all jobs (Numeric)
# rw - Real hourly wage, 2019$ (Numeric)
# multjobn - Number of jobs (Numeric)

def cleanDfwbho(row):
  thewbho = row["wbho"]
  return (0 if (thewbho=="White") else 1 if (thewbho=="Hispanic") else 2 if (thewbho=="Black") else 3 if (thewbho=="Other") else np.nan)
# end function cleanDfwbho
cleaned_df['wbho'] = df.apply(cleanDfwbho, axis=1)

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

#%%
#data exploration and graphing
# wage and other variables

# wage by gender
import matplotlib.pyplot as plt
import statistics
%matplotlib inline
plt.style.use('ggplot')

x = ['female', 'male']
energy = [statistics.mean(cleaned_df['rw'][cleaned_df.female == 0]), statistics.mean(cleaned_df['rw'][cleaned_df.female == 1])]

x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, energy, color='green')
plt.xlabel("gender")
plt.ylabel("Average hourly wage ($)")
plt.title("Wage by gender")

plt.xticks(x_pos, x)

plt.show()

#%%
# wage by citizen 
import matplotlib.pyplot as plt
import statistics
%matplotlib inline
plt.style.use('ggplot')

x = ['No-US citizen', 'US citizen']
energy = [statistics.mean(cleaned_df['rw'][cleaned_df.citizen == 0]), statistics.mean(cleaned_df['rw'][cleaned_df.citizen == 1])]

x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, energy, color='pink')
plt.xlabel("Citizen")
plt.ylabel("Average hourly wage ($)")
plt.title("Wage by citizen")

plt.xticks(x_pos, x)

plt.show()

#%%
# wage by married  
import matplotlib.pyplot as plt
import statistics
%matplotlib inline
plt.style.use('ggplot')

x = ['Never married', 'Married']
energy = [statistics.mean(cleaned_df['rw'][cleaned_df.married == 0]), statistics.mean(cleaned_df['rw'][cleaned_df.married == 1])]
# married - Married (0 = Never married, 1 = married)
x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, energy, color='blue')
plt.xlabel("Married")
plt.ylabel("Average hourly wage ($)")
plt.title("Wage by married")

plt.xticks(x_pos, x)

plt.show()

#%%
# wage by education
import matplotlib.pyplot as plt
import statistics
%matplotlib inline
plt.style.use('ggplot')

x = ['HS', 'Some college', 'College', 'Advanced', 'LTHS']
energy = [statistics.mean(cleaned_df['rw'][cleaned_df.educ == "HS"]), statistics.mean(cleaned_df['rw'][cleaned_df.educ == "Some college"]), statistics.mean(cleaned_df['rw'][cleaned_df.educ == "College"]), statistics.mean(cleaned_df['rw'][cleaned_df.educ == "Advanced"]), statistics.mean(cleaned_df['rw'][cleaned_df.educ == "LTHS"])]
# educ - Education level (HS, Some college, College, Advanced, LTHS)
x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, energy, color='gold')
plt.xlabel("Education level")
plt.ylabel("Average hourly wage ($)")
plt.title("Wage by education level")

plt.xticks(x_pos, x)

plt.show()


#%%
# wage by race
import matplotlib.pyplot as plt
import statistics
%matplotlib inline
plt.style.use('ggplot')

x = ['White', 'Hispanic', 'Black', 'Other']
energy = [statistics.mean(cleaned_df['rw'][cleaned_df.wbho == 0]), statistics.mean(cleaned_df['rw'][cleaned_df.wbho == 1]), statistics.mean(cleaned_df['rw'][cleaned_df.wbho == 2]), statistics.mean(cleaned_df['rw'][cleaned_df.wbho == 3])]
# wbho - Race (white, Hispanic, Black, Other)
x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, energy, color='purple')
plt.xlabel("Race")
plt.ylabel("Average hourly wage ($)")
plt.title("Wage by race")

plt.xticks(x_pos, x)

plt.show()

#%%
# wage by rural
import matplotlib.pyplot as plt
import statistics
%matplotlib inline
plt.style.use('ggplot')

x = ['No rural', 'Rural']
energy = [statistics.mean(cleaned_df['rw'][cleaned_df.rural == 0]), statistics.mean(cleaned_df['rw'][cleaned_df.rural == 1])]
# rural - rural (0 = no rural area, 1 = rural area)
x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, energy, color='grey')
plt.xlabel("Rural")
plt.ylabel("Average hourly wage ($)")
plt.title("Wage by rural")

plt.xticks(x_pos, x)

plt.show()


#%% 
# Wage (rw) VS Age VS Hours last week (Hourslw) by gender
import matplotlib.pyplot as plt
import seaborn as sns

cols = ['age', 'rw', 'hourslw']
sns.pairplot(cleaned_df, vars=cols, hue='female')
plt.show()



#%%
# To run wage prediction model 
# First, create new variable (rw_dummy) 
# Running Wage model by the six classifiers
# prepare data
#logistic regression to predict wage based on their responses
# Convert wage to be object by given 0 = low wage (lower than the mean = 23.48), 1 = high wage (higher than the mean = 23.48)
def cleanDfwage(row):
  thewage = row["rw"]
  return ("1" if (thewage >= 23.48) else "0" if (thewage < 23.48) else np.nan)
# end function cleanDfwage
cleaned_df['rw_dummy'] = cleaned_df.apply(cleanDfwage, axis=1)

#wage type chart
data = [['High wage', 4170], ['Low wage', 6248]]
wagechart = pd.DataFrame(data, columns = ['Level', 'Count']) 
wagechart_ch=sns.barplot(x='Level',y='Count',data=wagechart,palette='Greens')


# %%
# Build the model
# Mutiple linear regression

from statsmodels.formula.api import ols
modelwage1Fit = ols(formula='rw ~ age + C(female) + hourslw + C(forborn) + C(married) + C(educ) + C(wbho) + C(rural)', data=cleaned_df).fit()

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
X = cleaned_df[['age' , 'female' ,'citizen', 'married', 'rural', 'wbho', 'educ']]
X['Intercept'] = 1

# Compute and view VIF
vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [ variance_inflation_factor(X.values, i) for i in range(X.shape[1]) ] # list comprehension

# View results using print
print(vif)
# According to the results, VIF is very small in all independent variables, so it means there are not multicollinearity issues for this model


#%%
# %%
# prepare data
def cleanDfwage(row):
  thewage = row["rw"]
  return ("1" if (thewage >= 23.48) else "0" if (thewage < 23.48) else np.nan)
# end function cleanDfwage
cleaned_df['rw_dummy'] = cleaned_df.apply(cleanDfwage, axis=1)

try: cleaned_df.rw_dummy = pd.to_numeric( cleaned_df.rw_dummy )
except: print("Cannot handle to_numeric for column: rw_dummy")
finally: print(cleaned_df.rw_dummy.describe(), '\n', cleaned_df.rw_dummy.value_counts(dropna=False))
# above doesn't work, since there are many strings there.

xtarget = cleaned_df[['age', 'female', 'citizen', 'married', 'educ', 'wbho', 'rural']]
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
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import statsmodels.api as sm 
from statsmodels.formula.api import glm

#Original logistic regression
Original_WageModelLogitFit = glm(formula='rw ~ age + C(female) + C(citizen) + C(married) + C(educ) + C(wbho) + C(rural)', data=cleaned_df, family=sm.families.Binomial()).fit()
print( Original_WageModelLogitFit.summary() )

# logistic regression model for wage with the train set, and score it with the test set.
sklearn_wageModellogit = LogisticRegression()  # instantiate
sklearn_wageModellogit .fit(x_trainwage, y_trainwage)
print('Logit model accuracy (with the test set):', sklearn_wageModellogit.score(x_testwage, y_testwage))

sklearn_wageModellogit_predictions = sklearn_wageModellogit.predict(x_testwage)
#print(sklearn_wageModellogit_predictions)

#results (Logit)
confusion_matrix = confusion_matrix(y_testwage, sklearn_wageModellogit_predictions)
print(confusion_matrix)
print(classification_report(y_testwage,sklearn_wageModellogit_predictions))
print(f"The logit accuracy score is {accuracy_score(y_testwage, sklearn_wageModellogit_predictions)}")
print(f"The logit precision score is {precision_score(y_testwage, sklearn_wageModellogit_predictions, average='weighted')}")
print(f"The logit recall score is {recall_score(y_testwage, sklearn_wageModellogit_predictions, average='weighted')}")
print(f"The logit f1 score is {f1_score(y_testwage, sklearn_wageModellogit_predictions, average='weighted')}")

#%%
#timing logistic regression
from sklearn.model_selection import cross_val_score
%timeit -r 1 print(f'\n logit accuracy score: { cross_val_score(sklearn_wageModellogit, x_trainwage, y_trainwage, cv = 10 , scoring = "accuracy" ) } \n ' )

#%%
# logistic regression with CV model for wage with the train set, and score it with the test set.
CV_wageModellogit = LogisticRegressionCV()  # instantiate
CV_wageModellogit .fit(x_trainwage, y_trainwage)
print('Logit model accuracy (with the test set):', CV_wageModellogit.score(x_testwage, y_testwage))

CV_wageModellogit_predictions = CV_wageModellogit.predict(x_testwage)
#print(sklearn_wageModellogit_predictions)

#results (Logit)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_testwage, CV_wageModellogit_predictions)
print(confusion_matrix)
print(classification_report(y_testwage,CV_wageModellogit_predictions))
print(f"The logit cv accuracy score is {accuracy_score(y_testwage, CV_wageModellogit_predictions)}")
print(f"The logit cv precision score is {precision_score(y_testwage, CV_wageModellogit_predictions, average='weighted')}")
print(f"The logit cv recall score is {recall_score(y_testwage, CV_wageModellogit_predictions, average='weighted')}")
print(f"The logit cv f1 score is {f1_score(y_testwage, CV_wageModellogit_predictions, average='weighted')}")

#%%
#timing cv logistic regression
%timeit -r 1 print(f'\n logit CV accuracy score: { cross_val_score(CV_wageModellogit, x_trainwage, y_trainwage, cv = 10 , scoring = "accuracy" ) } \n ' )


#%%
# Second, move to KNN
# let's start with KNN spilt for wage model
# # KNeighborsClassifier()
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# choose k between 1 to 31
k_range = range(1, 11)
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

#%%
# the comfortable KNN choice (n = 8) 
# from sklearn.neighbors import KNeighborsClassifier
knn_split = KNeighborsClassifier(n_neighbors=8) 
# instantiate with n value given
knn_split.fit(x_trainwage,y_trainwage)
# knn_split.score(x_testwage,y_testwage)
knn_wagepredictions = knn_split.predict(x_testwage)
print(knn_wagepredictions)


# Evaluate test-set accuracy
print("KNN (k value = 8)")
print()
print(f'KNN train score:  {knn_split.score(x_trainwage,y_trainwage)}')
print(f'KNN test score:  {knn_split.score(x_testwage,y_testwage)}')
print(accuracy_score(y_testwage, knn_wagepredictions))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_testwage, knn_wagepredictions)
print(confusion_matrix)
print(classification_report(y_testwage, knn_wagepredictions))
print(f"The knn accuracy score is {accuracy_score(y_testwage, knn_wagepredictions)}")
print(f"The knn precision score is {precision_score(y_testwage, knn_wagepredictions, average='weighted')}")
print(f"The knn recall score is {recall_score(y_testwage, knn_wagepredictions, average='weighted')}")
print(f"The knn f1 score is {f1_score(y_testwage, knn_wagepredictions, average='weighted')}")

#%%
#timing knn
%timeit -r 1 print(f'\n knn accuracy score: { cross_val_score(knn_split, x_trainwage, y_trainwage, cv = 10 , scoring = "accuracy" ) } \n ' )


# %%
# Third, move to DecisionTreeClassifier() for wage model
# Let try max (max_depth = 5)
from sklearn.tree import DecisionTreeClassifier

dtreewage_max_5 = DecisionTreeClassifier(max_depth=5, random_state=1)
# Fit dt to the training set
dtreewage_max_5.fit(x_trainwage,y_trainwage)
# Predict test set labels
dtreewage_max_5_pred = dtreewage_max_5.predict(x_testwage)


# Evaluate test-set accuracy
print("DecisionTreeClassifier: max(max_depth = 5)")
print()
print(f'dtree train score:  {dtreewage_max_5.score(x_trainwage,y_trainwage)}')
print(f'dtree test score:  {dtreewage_max_5.score(x_testwage,y_testwage)}')
print(accuracy_score(y_testwage, dtreewage_max_5_pred))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_testwage, dtreewage_max_5_pred))
print(classification_report(y_testwage, dtreewage_max_5_pred))
print()
print(f"The dtree_max_5 accuracy score is {accuracy_score(y_testwage, dtreewage_max_5_pred)}")
print(f"The dtree_max_5  precision score is {precision_score(y_testwage, dtreewage_max_5_pred, average='weighted')}")
print(f"The dtree_max_5  recall score is {recall_score(y_testwage, dtreewage_max_5_pred, average='weighted')}")
print(f"The dtree_max_5  f1 score is {f1_score(y_testwage, dtreewage_max_5_pred, average='weighted')}")

#%%
# plot tree
from sklearn.tree import export_graphviz
xnamelist = ['age' , 'female' ,'citizen', 'married', 'rural', 'wbho', 'educ']
filename = 'tree'
export_graphviz(dtreewage_max_5, out_file = filename + '.dot' , feature_names = xnamelist, rounded=True,
filled=True)

#%%
#timing dtree
%timeit -r 1 print(f'\n dtreewage_max_5 accuracy score: { cross_val_score(dtreewage_max_5, x_trainwage, y_trainwage, cv = 10 , scoring = "accuracy" ) } \n ' )

#%%
# The plot
# Receiver Operator Characteristics (ROC)
# Area Under the Curve (AUC)
from sklearn.metrics import roc_auc_score, roc_curve
# generate a no skill prediction 
nonskill_probs = [0 for _ in range(len(y_testwage))]
# predict probabilities
tree_probs = dtreewage_max_5.predict_proba(x_testwage)
# keep probabilities for the positive outcome 
tree_probs = tree_probs[:, 1]
# calculate scores
nonskill_auc = roc_auc_score(y_testwage, nonskill_probs)
tree_auc = roc_auc_score(y_testwage, tree_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (nonskill_auc))
print('Decision Tree: ROC AUC=%.3f' % (tree_auc))
# calculate roc curves
nonskill_fpr, nonskill_tpr, _ = roc_curve(y_testwage, nonskill_probs)
tree_fpr, tree_tpr, _ = roc_curve(y_testwage, tree_probs)
# plot the roc curve for the model
plt.plot(nonskill_fpr, nonskill_tpr, linestyle='--', label='No Skill')
plt.plot(tree_fpr, tree_tpr, marker='.', label='Decision Tree')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()


#%%
# try 'gini'
dtreewage_gini = DecisionTreeClassifier(criterion='gini', random_state=1)
# Fit dt to the training set
dtreewage_gini.fit(x_trainwage,y_trainwage)
# Predict test set labels
dtreewage_gini_pred = dtreewage_gini.predict(x_testwage)
# Evaluate test-set accuracy
print("DecisionTreeClassifier: gini")
print()
print(f'dtree train score:  {dtreewage_gini.score(x_trainwage,y_trainwage)}')
print(f'dtree test score:  {dtreewage_gini.score(x_testwage,y_testwage)}')
print(accuracy_score(y_testwage, dtreewage_gini_pred))
print(confusion_matrix(y_testwage, dtreewage_gini_pred))
print(classification_report(y_testwage, dtreewage_gini_pred))
print()
print(f"The dtree_gini accuracy score is {accuracy_score(y_testwage, dtreewage_gini_pred)}")
print(f"The dtree_gini precision score is {precision_score(y_testwage, dtreewage_gini_pred, average='weighted')}")
print(f"The dtree_gini recall score is {recall_score(y_testwage, dtreewage_gini_pred, average='weighted')}")
print(f"The dtree_gini f1 score is {f1_score(y_testwage, dtreewage_gini_pred, average='weighted')}")


#%%
#timing dtree gini
%timeit -r 1 print(f'\n dtree gini accuracy score: { cross_val_score(dtreewage_gini, x_trainwage, y_trainwage, cv = 10 , scoring = "accuracy" ) } \n ' )



#%%
# Finally, let try SVC() for wage model
from sklearn.svm import SVC, LinearSVC
from sklearn import svm
svcwage =SVC()
svcwage.fit(x_trainwage,y_trainwage)

#Predictions
y_pred_svc = svcwage.predict(x_testwage)


# Evaluate test-set accuracy
print("SVC()")
print()
print(f'svc train score:  {svcwage.score(x_trainwage,y_trainwage)}')
print(f'svc test score:  {svcwage.score(x_testwage,y_testwage)}')
print(accuracy_score(y_testwage, y_pred_svc))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_testwage, y_pred_svc))
print(classification_report(y_testwage, y_pred_svc))
print()
print(f"The SVC accuracy score is {accuracy_score(y_testwage, y_pred_svc)}")
print(f"The SVC precision score is {precision_score(y_testwage, y_pred_svc, average='weighted')}")
print(f"The SVC recall score is {recall_score(y_testwage, y_pred_svc, average='weighted')}")
print(f"The SVC f1 score is {f1_score(y_testwage, y_pred_svc, average='weighted')}")

#%%
#timing SVC
%timeit -r 1 print(f'\n SVC accuracy score: { cross_val_score(svcwage, x_trainwage, y_trainwage, cv = 10 , scoring = "accuracy" ) } \n ' )


#%%
# let try LinearSVC() for wage model
from sklearn.svm import SVC, LinearSVC
svcwage_linearsvc = svm.LinearSVC()
svcwage_linearsvc.fit(x_trainwage,y_trainwage)

#Predictions
wage_pred_linearSVC = svcwage_linearsvc.predict(x_testwage)

# Evaluate test-set accuracy
print("SVC LinearSVC()")
print()
print(f'svc train score:  {svcwage_linearsvc.score(x_trainwage,y_trainwage)}')
print(f'svc test score:  {svcwage_linearsvc.score(x_testwage,y_testwage)}')
print(accuracy_score(y_testwage, wage_pred_linearSVC))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_testwage, wage_pred_linearSVC))
print(classification_report(y_testwage, wage_pred_linearSVC))
print()
print(f"The LinearSVC accuracy score is {accuracy_score(y_testwage, wage_pred_linearSVC)}")
print(f"The LinearSVC precision score is {precision_score(y_testwage, wage_pred_linearSVC, average='weighted')}")
print(f"The LinearSVC recall score is {recall_score(y_testwage, wage_pred_linearSVC, average='weighted')}")
print(f"The LinearSVC f1 score is {f1_score(y_testwage, wage_pred_linearSVC, average='weighted')}")

#%%
#timing SVC
%timeit -r 1 print(f'\n LinearSVC accuracy score: { cross_val_score(svcwage_linearsvc, x_trainwage, y_trainwage, cv = 10 , scoring = "accuracy" ) } \n ' )

#%%
#Calculate employment rate
emcounts = pd.crosstab(index=df['empl'], columns="count")
emcounts/emcounts.sum()

#%%
#educ level and empl
import seaborn as sns
sns.barplot(x='educ', y='empl', order=['LTHS','HS','Some college','College','Advanced'], data=df)

#%%
#number of children and empl
axes = sns.factorplot('ownchild','empl', 
                      data=df, aspect = 2.5, )

#%%
#employed, educ level, race and gender
FacetGrid = sns.FacetGrid(df, row='wbho', size=4.5, aspect=1.6)
FacetGrid.map(sns.pointplot, 'educ', 'empl', 'female', hue_order=None, color ="r",order=['LTHS','HS','Some college','College','Advanced'],palette=['blue','pink'])
FacetGrid.add_legend()
FacetGrid.set_titles("Female")

# %%
import statsmodels.api as sm  # Importing statsmodels
from statsmodels.formula.api import glm

modelemplLogitFit = glm(formula= 'empl ~ age + C(female) + C(citizen) + C(married) + C(educ) + C(wbho) + ownchild + C(vet)', data=df, family=sm.families.Binomial()).fit()
print(modelemplLogitFit.summary())


# %%
def cleanDfeduc(row):
  theedu = row["educ"]
  return (0 if (theedu=="HS") else 1 if (theedu=="Some college") else 2 if (theedu=="College") else 3 if (theedu=="Advanced") else 4 if (theedu=="LTHS") else np.nan)
# end function cleanDfeduc
df['educ'] = df.apply(cleanDfeduc, axis=1)

# wbho
# White = 0, Hispanic = 1, black = 2, other = 3
def cleanDfwbho(row):
  thewbho = row["wbho"]
  return (0 if (thewbho=="White") else 1 if (thewbho=="Hispanic") else 2 if (thewbho=="Black") else 3 if (thewbho=="Other") else np.nan)
# end function cleanDfeduc
df['wbho'] = df.apply(cleanDfwbho, axis=1)
df = df.dropna(subset=['age','female','citizen','married','educ','wbho','ownchild','vet','empl'])
# Prepare our X data (features, predictors, regressors) and y data (target, dependent variable)
xempl = df[['age','female','citizen','married','educ','wbho','ownchild','vet']]
yempl = df['empl']
print(type(xempl))
print(type(yempl))
# %%
# Decision Tree, y-target is categorical, similar to KNN, (multinomial) logistic Regression, 
# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# Import train_test_split
from sklearn.model_selection import train_test_split
# Split dataset into 80% train, 20% test
X_trainempl, X_testempl, y_trainempl, y_testempl= train_test_split(xempl,yempl, stratify=yempl,test_size=0.20, random_state=1)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import statsmodels.api as sm 
from statsmodels.formula.api import glm


#%% 
# logistic regression 
lr_empl = LogisticRegression()
lr_empl.fit(X_trainempl,y_trainempl)
predictionsempl = lr_empl.predict(X_testempl)

print(classification_report(y_testempl,predictionsempl))
lr_empl.score(X_testempl, y_testempl)
print(f"The logit accuracy score is {accuracy_score(y_testempl, predictionsempl)}")
print(f"The logit precision score is {precision_score(y_testempl, predictionsempl)}")
print(f"The logit recall score is {recall_score(y_testempl, predictionsempl)}")
print(f"The logit f1 score is {f1_score(y_testempl, predictionsempl)}")

#timing logistic regression
from sklearn.model_selection import cross_val_score
%timeit -r 1 print(f'\n logit accuracy score: { cross_val_score(lr_empl, X_trainempl, y_trainempl, cv = 10 , scoring = "accuracy" ) } \n ' )
#1.44 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)

#%%
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
lr_employ = lr_empl.predict_proba(X_testempl)
lr_employ = lr_employ[:, 1]
ns_employ = [0 for _ in range(len(y_testempl))]
ns_auc = roc_auc_score(y_testempl, ns_employ)
lr_auc = roc_auc_score(y_testempl, lr_employ)
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
ns_fpr, ns_tpr, _ = roc_curve(y_testempl, ns_employ)
lr_fpr, lr_tpr, _ = roc_curve(y_testempl, lr_employ)
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic',color='red')
pyplot.xlabel('False Positive Rate',fontsize=12)
pyplot.ylabel('True Positive Rate',fontsize=12)
pyplot.title('Logistic Regression',fontsize=15)
pyplot.legend()
pyplot.show()

#%%
# logistic regression with CV model for empl with the train set, and score it with the test set.
CV_emplModellogit = LogisticRegressionCV()  # instantiate
CV_emplModellogit .fit(X_trainempl, y_trainempl)
print('Logit model accuracy (with the test set):', CV_emplModellogit.score(X_testempl, y_testempl))

CV_emplModellogit_predictions = CV_emplModellogit.predict(X_testempl)
#print(sklearn_wageModellogit_predictions)

print(classification_report(y_testempl,CV_emplModellogit_predictions))
print(f"The logit cv accuracy score is {accuracy_score(y_testempl, CV_emplModellogit_predictions)}")
print(f"The logit cv precision score is {precision_score(y_testempl, CV_emplModellogit_predictions, average='weighted')}")
print(f"The logit cv recall score is {recall_score(y_testempl, CV_emplModellogit_predictions, average='weighted')}")
print(f"The logit cv f1 score is {f1_score(y_testempl, CV_emplModellogit_predictions, average='weighted')}")

%timeit -r 1 print(f'\n logit CV accuracy score: { cross_val_score(CV_emplModellogit, X_trainempl, y_trainempl, cv = 10 , scoring = "accuracy" ) } \n ' )
#15.1 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)

#%%
# KNN
#knn model for empl
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
knn_empl = KNeighborsClassifier( )
#this will run a range of k values and return the best parameters
k_range = list(range(1,11))
weights_options = ['uniform','distance']
k_grid = dict(n_neighbors=k_range, weights = weights_options)
grid = GridSearchCV(knn_empl, k_grid, cv=10, scoring = 'precision')
grid.fit(X_trainempl, y_trainempl)
grid.cv_results_
print ("Best Score: ",str(grid.best_score_))
print ("Best Parameters: ",str(grid.best_params_))
print ("Best Estimators: ",str(grid.best_estimator_))
#so the best value of k is 2
y_pred_empl = grid.predict(X_testempl)

print(f"The knn accuracy score is {accuracy_score(y_testempl, y_pred_empl)}")
print(f"The knn precision score is {precision_score(y_testempl, y_pred_empl)}")
print(f"The knn recall score is {recall_score(y_testempl, y_pred_empl)}")
print(f"The knn f1 score is {f1_score(y_testempl, y_pred_empl)}")
#timing knn
%timeit -r 1 print(f'\n knn accuracy score: { cross_val_score(knn_empl, X_trainempl, y_trainempl, cv = 10 , scoring = "accuracy" ) } \n ' )
#1.28 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)

# %%
# SVC
svc_empl = SVC()
svc_empl.fit(X_trainempl,y_trainempl)
svc_predictempl = svc_empl.predict(X_testempl)

print(f"The svc accuracy score is {accuracy_score(y_testempl, svc_predictempl)}")
print(f"The svc precision score is {precision_score(y_testempl, svc_predictempl)}")
print(f"The svc recall score is {recall_score(y_testempl, svc_predictempl)}")
print(f"The svc f1 score is {f1_score(y_testempl, svc_predictempl)}")

%timeit -r 1 print(f'\n svc accuracy score: { cross_val_score(svc_empl, X_trainempl, y_trainempl, cv = 10 , scoring = "accuracy" ) } \n ' )
#2min 11s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)

#%%
# Decision tree classifier
# Instantiate dtree
dtree = DecisionTreeClassifier(max_depth=5, random_state=1)
# Fit dt to the training set
dtree.fit(X_trainempl,y_trainempl)
# Predict test set labels
dtree_predempl = dtree.predict(X_testempl)
# Evaluate test-set accuracy
print(f"The dtree_empl accuracy score is {accuracy_score(y_testempl, dtree_predempl)}")
print(f"The dtree_empl  precision score is {precision_score(y_testempl, dtree_predempl, average='weighted')}")
print(f"The dtree_empl  recall score is {recall_score(y_testempl, dtree_predempl, average='weighted')}")
print(f"The dtree_empl  f1 score is {f1_score(y_testempl, dtree_predempl, average='weighted')}")

%timeit -r 1 print (f'\n tree CV accuracy score: { cross_val_score(dtree, X_trainempl, y_trainempl, cv = 10 , scoring = "accuracy" ) } \n ' ) 
# 191 ms ± 0 ns per loop (mean ± std. dev. of 1 run, 10 loops each



# %%

# Data Mining Project - Ignatios Draklellis

# age - age (Numeric)
# female - sex (0 = male, 1 = female)
# wbho - White = 0, Hispanic = 1, black = 2, other = 3
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
#
# 


#%%
# Prepare Geography Data
import matplotlib.pyplot as plt 
import numpy as np
dfGeo = df[['age', 'rw', 'rural', 'forborn', 'nilf', 'hourslw', 'multjobn', 'educ', 'wbho']]

#Recode
dfGeo.wbho[dfGeo.wbho == 'White'] = 0
dfGeo.wbho[dfGeo.wbho == 'Hispanic'] = 1
dfGeo.wbho[dfGeo.wbho == 'Black'] = 2
dfGeo.wbho[dfGeo.wbho == 'Other'] = 3

dfGeo.educ[dfGeo.educ == 'LTHS'] = 0
dfGeo.educ[dfGeo.educ == 'HS'] = 1
dfGeo.educ[dfGeo.educ == 'Some college'] = 2
dfGeo.educ[dfGeo.educ == 'College'] = 3
dfGeo.educ[dfGeo.educ == 'Advanced'] = 4


dfGeo = dfGeo.dropna()

dfGeo.tail(5)

#%%
#Data Visuals and Graphing for Geography

#Descriptive Statistics
geoVars=dfGeo[['age','hourslw','rw']]
tab1=geoVars.describe()
print(tab1)

#relabeling
dfGeo.rural[dfGeo.rural == 0] = 'Urban'
dfGeo.rural[dfGeo.rural == 1] = 'Rural'

dfGeo.forborn[dfGeo.forborn == 0] = 'Immigrant'
dfGeo.forborn[dfGeo.forborn == 1] = 'Native'

dfGeo = dfGeo.rename(columns={"rural": "Geography", "forborn": "Foreign Born"})

#Violin Plots
import seaborn as sns
sns.catplot(x="Geography", y="rw", hue="Foreign Born",
            kind="violin", split=False, data=dfGeo)
plt.ylabel("Real Wages (2019 US Dollars)")
plt.title("Foreign and Native Born Wages Based on Geography")
plt.show()

# Grouped Bar Plots
sns.catplot(x="educ", y="rw", hue="Geography", data=dfGeo, height=6, kind="bar", palette="muted")
plt.xlabel("Education Level")
plt.ylabel("Real Wages (2019 US Dollars)")
plt.title("Income Disparities by Geography")
plt.show()

sns.catplot(x="educ", y="rw", hue="Foreign Born", data=dfGeo, height=6, kind="bar", palette="muted")
plt.xlabel("Education Level")
plt.ylabel("Real Wages (2019 US Dollars)")
plt.title("Income Disparities by Immigration")
plt.show()

sns.catplot(x="Geography", y="rw", hue="Foreign Born", data=dfGeo, height=6, kind="bar", palette="muted")
plt.xlabel("Education Level")
plt.ylabel("Real Wages (2019 US Dollars)")
plt.title("Income Disparities by Geography and Immigration")
plt.show()


#%%
#Reload original data and encode strings to ints
dfGeo = df[['age', 'rw', 'rural', 'forborn', 'nilf', 'hourslw', 'multjobn', 'educ', 'wbho']]

#Recode
dfGeo.wbho[dfGeo.wbho == 'White'] = 0
dfGeo.wbho[dfGeo.wbho == 'Hispanic'] = 1
dfGeo.wbho[dfGeo.wbho == 'Black'] = 2
dfGeo.wbho[dfGeo.wbho == 'Other'] = 3

dfGeo.educ[dfGeo.educ == 'LTHS'] = 0
dfGeo.educ[dfGeo.educ == 'HS'] = 1
dfGeo.educ[dfGeo.educ == 'Some college'] = 2
dfGeo.educ[dfGeo.educ == 'College'] = 3
dfGeo.educ[dfGeo.educ == 'Advanced'] = 4


dfGeo = dfGeo.dropna()

dfGeo.head(25)


#%%
#Make 4:1 train/test split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(dfGeo.drop('rural',axis=1), dfGeo['rural'], test_size=0.20)


#Logistic Regression
geoLogit = LogisticRegression() #instantiate Logit
geoLogit.fit(X_train,y_train)
predictions = geoLogit.predict(X_test)
#Results
confusion_matrix = confusion_matrix(y_test, predictions)
print(confusion_matrix)
print(classification_report(y_test,predictions))
geoLogit.score(X_test, y_test)


#CV on Logit
cv_model=LogisticRegressionCV()
cv_model.fit(X_train,y_train)
cv_predictions = cv_model.predict(X_test)
#Results
print(classification_report(y_test,cv_predictions))
cv_model.score(X_test, y_test)


#%%
# Mutiple Linear Regression
from statsmodels.formula.api import ols

linearGeoModel = ols(formula='rural ~ rw + C(nilf) + hourslw + C(forborn) + C(multjobn) + C(educ) + C(wbho)', data=dfGeo).fit()

print( type(linearGeoModel) )
print( linearGeoModel.summary() )

#OLS Predictions 
modelpredicitons = pd.DataFrame( columns=['geoModel_LM'], data= linearGeoModel.predict(dfGeo)) 
print(modelpredicitons.shape)
print( modelpredicitons.head() )



# %%
#Train Test 4:1 Split
xtarget = dfGeo[['age', 'rw', 'forborn', 'nilf', 'hourslw', 'multjobn', 'educ', 'wbho']]
ytarget = dfGeo['rural'] 

print(type(xtarget))
print(type(ytarget))

#make a train-test split in 4:1 ratio. 
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import glm
import statsmodels.api as sm


x_trainGeo, x_testGeo, y_trainGeo, y_testGeo = train_test_split(xtarget, ytarget, test_size = 0.2, random_state=1)

print('x_trainGeo type',type(x_trainGeo))
print('x_trainGeo shape',x_trainGeo.shape)
print('x_testGeo type',type(x_testGeo))
print('x_testGeo shape',x_testGeo.shape)
print('y_trainGeo type',type(y_trainGeo))
print('y_trainGeo shape',y_trainGeo.shape)
print('y_testGeo type',type(y_testGeo))
print('y_testGeo shape',y_testGeo.shape)


#logistic regression
geoLogitModelFit = glm(formula='rural ~ rw + C(nilf) + hourslw + C(forborn) + C(multjobn) + C(educ) + C(wbho)', data=dfGeo, family=sm.families.Binomial()).fit()
print( geoLogitModelFit.summary() )

# logistic regression model for geography with the train set, and score it with the test set.
sklearn_geoModellogit = LogisticRegression()  # instantiate
sklearn_geoModellogit.fit(x_trainGeo, y_trainGeo)
print('Logit model accuracy (with the test set):', sklearn_geoModellogit.score(x_testGeo, y_testGeo))

sklearn_geoModellogit_predictions = sklearn_geoModellogit.predict(x_testGeo)
#print(sklearn_geoModellogit_predictions)
print(sklearn_geoModellogit.predict_proba(x_trainGeo[:8]))
print(sklearn_geoModellogit.predict_proba(x_testGeo[:8]))

#results (Logit)
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print('Logit model accuracy (with the test set):', sklearn_geoModellogit.score(x_testGeo, sklearn_geoModellogit_predictions))
confusion_matrix = confusion_matrix(y_testGeo, sklearn_geoModellogit_predictions)
print(confusion_matrix)
print(classification_report(y_testGeo,sklearn_geoModellogit_predictions))


#%%
# K Nearest Neighbors (KNN)
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# choose k between 1 to 31
k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn_split = KNeighborsClassifier(n_neighbors=k)
    knn_split.fit(x_trainGeo,y_trainGeo)
    scores = knn_split.score(x_testGeo,y_testGeo)
    k_scores.append(scores.mean())
# plot to see clearly
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN_split')
plt.ylabel('Accuracy score')
plt.show()

# the comfortable KNN choice (n = 5) 
knn_split_5 = KNeighborsClassifier(n_neighbors=5) 
# instantiate with n value given
knn_split_5.fit(x_trainGeo,y_trainGeo)
# knn_split.score(x_testGeo,y_testGeo)
knn_Geopredictions = knn_split_5.predict(x_testGeo)
print(knn_Geopredictions)
print(knn_split_5.predict_proba(x_trainGeo[:8]))
print(knn_split_5.predict_proba(x_testGeo[:8]))

# Evaluate test-set accuracy
print("KNN (k value = 5)")
print()
print(f'KNN train score:  {knn_split_5.score(x_trainGeo,y_trainGeo)}')
print(f'KNN test score:  {knn_split_5.score(x_testGeo,y_testGeo)}')
print(accuracy_score(y_testGeo, knn_Geopredictions))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_testGeo, knn_Geopredictions)
print(confusion_matrix)
print(classification_report(y_testGeo, knn_Geopredictions))
print() 

#%%
#Decision Trees
from sklearn.tree import DecisionTreeClassifier

dtreeGeo = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=1)
# Fit dt to the training set
dtreeGeo.fit(x_trainGeo,y_trainGeo)
# Predict test set labels
dtreeGeoPred = dtreeGeo.predict(x_testGeo)
print(dtreeGeoPred)
print(dtreeGeo.predict_proba(x_trainGeo[:8]))
print(dtreeGeo.predict_proba(x_testGeo[:8]))

# Evaluate test-set accuracy
print("DecisionTreeClassifier: entropy(max_depth = 5)")
print()
print(f'dtree train score:  {dtreeGeo.score(x_trainGeo,y_trainGeo)}')
print(f'dtree test score:  {dtreeGeo.score(x_testGeo,y_testGeo)}')
print(accuracy_score(y_testGeo, dtreeGeoPred))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_testGeo, dtreeGeoPred))
print(classification_report(y_testGeo, dtreeGeoPred))
print()


#%%
#SVC() for Geography model
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix

# SVC 
svcGeoAuto = SVC(gamma='auto', probability=True)
svcGeoAuto.fit(x_trainGeo,y_trainGeo)

#Predictions
svcGeoAuto_pred = svcGeoAuto.predict(x_testGeo)
print(svcGeoAuto_pred)
print(svcGeoAuto.predict_proba(x_trainGeo[:8]))
print(svcGeoAuto.predict_proba(x_testGeo[:8]))

# Evaluate test-set accuracy
print("SVC (adjust gamma: auto)")
print()
print(f'svc train score:  {svcGeoAuto.score(x_trainGeo,y_trainGeo)}')
print(f'svc test score:  {svcGeoAuto.score(x_testGeo,y_testGeo)}')
print(accuracy_score(y_testGeo, svcGeoAuto_pred))
print(confusion_matrix(y_testGeo, svcGeoAuto_pred))
print(classification_report(y_testGeo, svcGeoAuto_pred))
print()


#%%
#Linear Kernel SVC
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import SVC, LinearSVC
from sklearn import svm


svcGeo_linearsvc = svm.LinearSVC()
svcGeo_linearsvc.fit(x_trainGeo,y_trainGeo)
y_pred = svcGeo_linearsvc.predict(x_testGeo)


#Predictions
y_pred_linearsvc = svcGeo_linearsvc.predict(x_testGeo)
print(y_pred)
print(svcGeo_linearsvc._predict_proba_lr(x_trainGeo[:8]))
print(svcGeo_linearsvc._predict_proba_lr(x_testGeo[:8]))

# Evaluate test-set accuracy
print("SVC LinearSVC()")
print()
print(f'svc train score:  {svcGeo_linearsvc.score(x_trainGeo,y_trainGeo)}')
print(f'svc test score:  {svcGeo_linearsvc.score(x_testGeo,y_testGeo)}')
print(accuracy_score(y_testGeo, y_pred_linearsvc))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_testGeo, y_pred_linearsvc))
print(classification_report(y_testGeo, y_pred_linearsvc))
print()



#%%
#Run Time Calculations for Geo Models
from sklearn.model_selection import cross_val_score
import timeit

%timeit -r 1 print(f'\n cv_model CV accuracy score: { cross_val_score(cv_model, x_trainGeo, y_trainGeo, cv = 10 , scoring = "accuracy" ) } \n ' ) 
# 13.1 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)


%timeit -r 1 print(f'\n linear SVC CV accuracy score: { cross_val_score(svcGeo_linearsvc, x_trainGeo, y_trainGeo, cv = 10 , scoring = "accuracy" ) } \n ' ) 
# 22.4 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)

%timeit -r 1 print(f'\n logistic CV accuracy score: { cross_val_score(geoLogit, x_trainGeo, y_trainGeo, cv = 10 , scoring = "accuracy" ) } \n ' ) 
# 1.12 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)

%timeit -r 1 print(f'\n KNN CV accuracy score: { cross_val_score(knn_split_5, x_trainGeo, y_trainGeo, cv = 10 , scoring = "accuracy" ) } \n ' ) 
# 1.66 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)

%timeit -r 1 print(f'\n dtree CV accuracy score: { cross_val_score(dtreeGeo, x_trainGeo, y_trainGeo, cv = 10 , scoring = "accuracy" ) } \n ' ) 
# 514 ms ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)




# %%
# Receiver Operator Characteristics (ROC)
# Area Under the Curve (AUC)
from sklearn.metrics import roc_auc_score, roc_curve

ns_probs = [0 for _ in range(len(y_testGeo))]
lr_probs = geoLogit.predict_proba(x_testGeo)
lr_probs = lr_probs[:, 1]
ns_auc = roc_auc_score(y_testGeo, ns_probs)
lr_auc = roc_auc_score(y_testGeo, lr_probs)
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
ns_fpr, ns_tpr, _ = roc_curve(y_testGeo, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_testGeo, lr_probs)
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

#Dtree
ns_probs = [0 for _ in range(len(y_testGeo))]
lr_probs = dtreeGeo.predict_proba(x_testGeo)
lr_probs = lr_probs[:, 1]
ns_auc = roc_auc_score(y_testGeo, ns_probs)
lr_auc = roc_auc_score(y_testGeo, lr_probs)
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
ns_fpr, ns_tpr, _ = roc_curve(y_testGeo, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_testGeo, lr_probs)
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='dtreeGeo')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()



#%%
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
# Plotting decision regions -- Still trying to get this to work
plot_decision_regions(x_testGeo.values, y_testGeo.values, clf=geoLogit, legend=3, filler_feature_values={2:1} , filler_feature_ranges={2: 3} )
# filler_feature_values is used when you have more than 2 predictors, then 
# you need to specify the ones not shown in the 2-D plot. For us, 
# the rank is at poition 2, and the value can be 1, 2, 3, or 4.
# also need to specify the filler_feature_ranges for +/-, otherwise only data points with that feature value will be shown.


# Adding axes annotations
plt.xlabel(x_testGeo.columns[0])
plt.ylabel(x_testGeo.columns[1])
plt.title(geoLogit.__class__.__name__)
plt.show()


# And the decision tree result
plot_decision_regions(x_testGeo.values, y_testGeo.values, clf=dtreeGeo, legend=3, filler_feature_values={2:1} , filler_feature_ranges={2: 3} )
plt.xlabel(x_testGeo.columns[0])
plt.ylabel(x_testGeo.columns[1])
plt.title(dtreeGeo.__class__.__name__)
plt.show()



#%%

