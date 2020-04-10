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
cols_list=['age','female','wbho','forborn','citizen','vet','married','ownchild','empl','unem','nilf','uncov','state','educ','centcity','suburb','rural','smsastat14','ind_m03','agric','manuf','servs','hourslw','rw', 'faminc', 'multjobn', 'wage3']
df= pd.read_csv(filepath, usecols=cols_list)  

dfChkBasics(df, True)
print(df.dtypes)

# Data dict
# age - age of US worker
# female - sex (0 = male, 1 = female)
# wbho - Race
# forborn - Foreign born
# citizen - US citizen
# vet - Veteran
# married - Married
# ownchild - Number of children
# empl - Employed
# unem - Unemployed
# nilf - Not in labor force
# uncov - Union coverage
# state - state
# educ - Education level
# centcity - Central city
# suburb - suburbs
# rural - rural
# smsastat14 - Metro CBSA FIPS Code
# ind_m03 - Major Industry Recode
# agric - Agriculture
# manuf - Manufacturing
# servs - Services
# hourslw - Hours last week, all jobs
# rw - Real wage, 2019$
# multjobn - Number of jobs
# wage3 - Hourly wage (both hourly and non-hourly workers)
# faminc - Family income band


# %%
