#simple linear regression model to predict the relative humidity values based on the temperature values recorded in the dataset.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('C:\Python Projects\AirQualityUCI(1).csv' ,sep=';')

df = df.drop(columns=['Unnamed: 15', 'Unnamed: 16'], axis=1) 

df = df.dropna()

dt_series = pd.Series(data = [item.split("/")[2] + "-" + item.split("/")[1] + "-" + item.split("/")[0] for item in df['Date']], index=df.index) + ' ' + pd.Series(data=[str(item).replace(".", ":") for item in df['Time']], index=df.index)
dt_series = pd.to_datetime(dt_series)

df = df.drop(columns=['Date', 'Time'], axis=1)
df.insert(loc=0, column='DateTime', value=dt_series)

year_series = dt_series.dt.year

month_series = dt_series.dt.month

day_series = dt_series.dt.day

day_name_series = dt_series.dt.day_name()

df['Year'] = year_series
df['Month'] = month_series
df['Day'] = day_series
df['Day Name'] = day_name_series

df = df.sort_values(by='DateTime')

def comma_to_period(series):
    new_series = pd.Series(data=[float(str(item).replace(',', '.')) for item in series], index=df.index)
    return new_series

cols_to_correct = ['CO(GT)', 'C6H6(GT)', 'T', 'RH', 'AH']
for col in cols_to_correct: 
    df[col] = comma_to_period(df[col]) 

df = df.drop(columns=['NMHC(GT)', 'CO(GT)', 'NOx(GT)', 'NO2(GT)'], axis=1)

aq_2004_df = df[df['Year'] == 2004]
aq_2005_df = df[df['Year'] == 2005]

for col in aq_2004_df.columns[1:-4]:
  median = aq_2004_df.loc[aq_2004_df[col] != -200, col].median()
  aq_2004_df[col] = aq_2004_df[col].replace(to_replace=-200, value=median)

for col in aq_2005_df.columns[1:-4]:
  median = aq_2005_df.loc[aq_2005_df[col] != -200, col].median()
  aq_2005_df[col] = aq_2005_df[col].replace(to_replace=-200, value=median)

group_2004_month = aq_2004_df.groupby(by='Month')
group_2005_month = aq_2005_df.groupby(by='Month')

df = pd.concat([aq_2004_df, aq_2005_df])

df.info()



corr_df = df.iloc[:, 1:-4].corr()
plt.figure(figsize = (10, 6), dpi = 96)
sns.heatmap(data = corr_df, annot = True) 
plt.show()


plt.figure(figsize = (12, 4), dpi = 96)
sns.regplot(x = 'T', y = 'RH', data = df, color = 'teal')
plt.xlabel("Temperature")
plt.ylabel("Relative Humidity")
plt.show()


from ipywidgets import interactive

def simulate_straight_lines(slope, intercept):
  plt.figure(figsize = (12, 5), dpi = 96)
  x_coordinates = np.arange(df['T'].min() - 10, df['T'].max() + 10) 
  plt.plot(x_coordinates, slope * x_coordinates + intercept)
  plt.scatter(df['T'], df['RH'])
  plt.xlabel('Temperature')
  plt.ylabel('Relative Humidity')
  plt.show()

interactive_plot = interactive(simulate_straight_lines, 
                               slope = (-2, 0, 0.1), 
                               intercept = (-100, 100, 2)) 
interactive_plot



from sklearn.model_selection import train_test_split
X=df['T']
y=df['RH']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


feature=np.arange(1,11)
target=np.arange(11,21)
feature_train,feature_test,target_train,target_test=train_test_split(feature,target,test_size=0.4)
print(feature_train,feature_test)


feature=np.arange(1,11)
target=np.arange(11,21)
feature_train,feature_test,target_train,target_test=train_test_split(feature,target,test_size=0.4,random_state=42)
print(feature_train,feature_test)


def errors_product():
  prod=(X_train-X_train.mean())*(y_train-y_train.mean())
  return prod
def squared_errors():
  sq_errors=(X_train-X_train.mean())**2
  return sq_errors


slope=errors_product().sum()/squared_errors().sum()
intercept=(y_train.mean()-slope*X_train.mean())
print(slope,intercept)


plt.figure(figsize=(10,6))
plt.scatter(df['T'],df['RH'])
plt.plot(df['T'],slope*df['T']+intercept,color='Black')
plt.show()

y_pred=slope*X_test+intercept
print(y_pred)

def rh_predict(temp):
  print('Relative Humidity is ', -1.1119759072310362*temp + 69.71462369190169)
rh_predict(58)