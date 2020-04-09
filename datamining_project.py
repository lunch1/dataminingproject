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
os.chdir('../data')  
# sometime when I opened the workspace from another folder, the 
# working directory getcwd() will be in the wrong place. 
# You can change it with chdir()
dirpath = os.getcwd() # print("current directory is : " + dirpath)
filepath = os.path.join( dirpath ,'cepr_org_2019.csv')
df= pd.read_csv(filepath)  

dfChkBasics(df, True)
print(df.dtypes)


# %%
# clean price from object to be numuric
print(df.price.describe(), '\n', df.price.value_counts(dropna=False) )

def clean_currency(x):
    """ If the value is a string, then remove currency symbol and delimiters
    otherwise, the value is numeric and can be converted
    """
    if isinstance(x, object):
        return(x.replace('$', '').replace(',', '')) 
    return(x) 


#%%
df['price'] = df['price'].apply(clean_currency).astype('float')
df['weekly_price'] = df['weekly_price'].dropna().apply(clean_currency).astype('float')
df['monthly_price'] = df['monthly_price'].dropna().apply(clean_currency).astype('float')
df['security_deposit'] = df['security_deposit'].dropna().apply(clean_currency).astype('float')
df['cleaning_fee'] = df['cleaning_fee'].dropna().apply(clean_currency).astype('float')
df['extra_people'] = df['extra_people'].dropna().apply(clean_currency).astype('float')
print(df.dtypes)  

#%%
#range the daily price 
def cleanDfdailyprice(row):
  theprice = row["price"]
  return ("$0 - $50" if (theprice <= 50) else "$51 - $100" if (51 <= theprice <= 100) else "$101 - $150" if (101 <= theprice <= 150) else "$151 - $200" if (151 <= theprice <= 200) else "$201 - $250" if (201 <= theprice <= 250) else "$251 - $300" if (251 <= theprice <= 300) else "$301 - $500" if (301 <= theprice <= 500) else "$501 - $1,000" if (501 <= theprice <= 1000) else "$1,001 up" if (theprice >= 1001) else np.nan)

df['daily_price'] = df.apply(cleanDfdailyprice, axis=1)
print(df.dtypes)  

#%%
#range the weekly price 
def cleanDfweeklyprice(row):
  weeklyprice = row["weekly_price"]
  return ("$0 - $300" if (weeklyprice <= 300) else "$501 - $700" if (501 <= weeklyprice  <= 700) else "$801 - $1,000" if (801 <= weeklyprice <= 1000) else "$1,001 - $1,500" if (1001 <= weeklyprice <= 1500) else "$1,501 - $3,000" if (1501 <= weeklyprice <= 3000) else "$3,000 up" if (weeklyprice >= 3000) else "No offer")

df['weekly_price2'] = df.apply(cleanDfweeklyprice, axis=1)
print(df.dtypes)  

#%%
#range the monthly price 
def cleanDfmonthlyprice(row):
  monthlyprice = row["monthly_price"]
  return ("$0 - $1,500" if (monthlyprice <= 1500) else "$1,501 - $3,000" if (1501 <= monthlyprice  <= 3000) else "$3,001 - $5,000" if (3001 <= monthlyprice <= 5000) else "$5,001 - $10,000" if (5001 <= monthlyprice <= 10000) else "$10,000 up"  if (monthlyprice >= 10000) else "No offer")

df['monthly_price2'] = df.apply(cleanDfmonthlyprice, axis=1)
print(df.dtypes) 

#%%
#review rate of 50%
review_rate = 0.5
#average stay night in NYC
ava_nights = 6.4
#maximun occupancy 
max_occupancy = 0.7

#estimated booking per month
df['bookings_per_month'] = round(df['reviews_per_month']/review_rate, 3)

print (df['bookings_per_month'].head())


#%%
df['ava_nights'] = 6.4
df['max_occupancy'] = 0.7

dfmax = df.loc[:,['minimum_nights', 'ava_nights']]
df['max'] = dfmax.max(axis=1)
dfmin = df.loc[:,['bookings_per_month', 'max']]
df['min'] = (dfmin['bookings_per_month']*dfmin['max'])/30
dfmin2 = df.loc[:,['min', 'max_occupancy']]
df['min2'] = dfmin2.min(axis=1)
df['est_occupancy'] = (round(df['min2'], 3)*100)

df = df.drop(columns=['max', 'min', 'min2'])

print (df['est_occupancy'].head())



#%%

#estimated nights per year
df['est_nights_per_year'] = round(df['est_occupancy']*365, 0)/100

print (df['est_nights_per_year'])

#%%

#estimated income per month
df['est_income_per_month'] = round(df['est_occupancy']*30*df['price'], 2)/100

print (df['est_income_per_month'])

print(df.est_income_per_month.describe(), '\n', df.est_income_per_month.value_counts(dropna=False) )


#%%

# convert availability_365 to be high or low
def cleanDfavaliable(row):
  theavailability = row["availability_365"]
  return ("high" if (theavailability >= 60) else "low" if (theavailability < 60) else np.nan)
# end function cleanDfHappy

df['availability_level'] = df.apply(cleanDfavaliable, axis=1)
print(df.dtypes)  
 

#%%
print(df.daily_price.describe(), '\n', df.daily_price.value_counts(dropna=False) )

# %%
# convert review_scores_rating to be high, moderate, or low
def cleanDfreviewscores(row):
  thescore = row["review_scores_rating"]
  return ("high" if (thescore >= 90) else "moderate" if (thescore >= 60) else "low" if (thescore <= 59) else np.nan)

df['review_scores_rating_level'] = df.apply(cleanDfreviewscores, axis=1)
print(df.dtypes)  


# %%
df.to_csv(r'C:\Users\admin\Desktop\Education\Fouth Semester\Data visualization\Project 2\data\cleaned_data.csv', index=False) 

# %%
