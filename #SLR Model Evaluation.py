#SLR Model Evaluation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

csv_file = 'https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/air-quality/AirQualityUCI.csv'
df = pd.read_csv(csv_file, sep=';')

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


from sklearn.model_selection import train_test_split

X = df['T'] 
y = df['RH'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42) # Test set will have 33% of the values.




def errors_product():
  prod = (X_train - X_train.mean()) * (y_train - y_train.mean())
  return prod

def squared_errors():
  sq_errors = (X_train - X_train.mean()) ** 2
  return sq_errors

slope = errors_product().sum()/ squared_errors().sum() 
intercept = y_train.mean() - slope * X_train.mean()

print(f"Slope: {slope} \nIntercept: {intercept}")




plt.style.use('dark_background')
plt.figure(figsize = (12, 5), dpi = 96)
plt.title("Regression Line", fontsize = 16)
plt.scatter(df['T'], df['RH'])
plt.plot(df['T'], slope * df['T'] + intercept, color = 'r', linewidth = 2, label = '$y = −1.1120x + 69.6911$')
plt.xlabel("Temperature")
plt.ylabel("Relative Humidity")
plt.legend()
plt.show()



def r_squared(X,y):
  y_pred=slope*X+intercept
  sq_errors=(y-y_pred)**2
  sq_total=(y-y.mean())**2
  sse=sq_errors.sum()
  sst=sq_total.sum()
  r2=1-(sse/sst)
  return r2
r_squared(X_train,y_train)


np.corrcoef(X_train,y_train)**2


r_squared(X_test,y_test)


def mean_squared_error(X,y_actual):
  y_pred=slope*X+intercept
  sq_error=(y_actual-y_pred)**2
  mse=sq_error.sum()/len(y_actual)
  return mse

def root_mean_sq_error(X,y_actual):
  y_pred=slope*X+intercept
  sq_error=(y_actual-y_pred)**2
  
  mse=sq_error.sum()/len(y_actual)
  rmse=np.sqrt(mse)
  return rmse

def mean_abs_error(X,y_actual):
  y_pred=slope*X+intercept
  abs_error=np.abs(y_actual-y_pred)
  mae=np.sum(abs_error)/len(y_actual)

  
  return mae




print(mean_squared_error(X_train,y_train))
print(root_mean_sq_error(X_train,y_train))
print(mean_abs_error(X_train,y_train))

print(mean_squared_error(X_test,y_test))
print(root_mean_sq_error(X_test,y_test))
mean_abs_error(X_test,y_test)


print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

type(X_train.values)
X_train_reshaped=X_train.values.reshape(-1,1)
X_test_reshaped=X_test.values.reshape(-1,1)
y_train_reshaped=y_train.values.reshape(-1,1)
y_test_reshaped=y_test.values.reshape(-1,1)


from sklearn.linear_model import LinearRegression


lr=LinearRegression()
lr.fit(X_train_reshaped,y_train_reshaped)
print(lr.coef_,lr.intercept_)




from sklearn.metrics import r2_score , mean_squared_error ,  mean_absolute_error
y_train_pred=lr.predict(X_train_reshaped)
y_test_pred=lr.predict(X_test_reshaped)

print(r2_score(y_train_reshaped,y_train_pred))
print(mean_squared_error(y_train_reshaped,y_train_pred))
print(np.sqrt(mean_squared_error(y_train_reshaped,y_train_pred)))
print(mean_absolute_error(y_train_reshaped,y_train_pred))

print(r2_score(y_test_reshaped,y_test_pred))
print(mean_squared_error(y_test_reshaped,y_test_pred))
print(np.sqrt(mean_squared_error(y_test_reshaped,y_test_pred)))
print(mean_absolute_error(y_test_reshaped,y_test_pred))



