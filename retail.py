import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose  
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler  
import pickle

from sklearn.metrics import r2_score, mean_squared_error 
import statsmodels.api as sm
import statsmodels.formula.api as smf 

from sklearn.linear_model import LinearRegression  
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge 
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA 

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

stores = pd.read_csv("stores data-set.csv")
sales = pd.read_csv("sales data-set.csv")
features = pd.read_csv("Features data set.csv")

print("#INFORMATION ABOUT DATASET#")
print(stores.head())
print(sales.head())
print(stores.head()) 

# MERGE DATASET

features = features.merge(stores, on = 'Store')
df = features.merge(sales, on = ['Store','Date','IsHoliday'])
df=df.fillna(0)

print(df.shape)

print(df.describe())  

### FEATURE ENGINEERING ####

# NUMERICAL AND CATEGORICAL DATA

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
      Veri Setindeki kategorik,numerik ve kategorik fakat kordinal değişkenlerin isimlerini verir.
      Not : Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir
      Parameters
      ----------
             dataframe : dataframe
                   Değişken isimleri alınmak istenen dataframe
             cat_th : int,optional
                   Numerik fakat kategorik olan değişkenler için sınıf eşit değeri
             car_th : int , optional
                   Kategorik gibi gözüküp fakat kordinal değişkenler için sınıf eşik değeri
      Returns
      -------
           cat_cols: list
                   Kategorik değişken listesi
           num_cols: list
                   Numerik değişken listesi
      # bazı fonksiyonlar bool gibi ticket gibi kategorik ama sayısal olan değişkenleri de ifade edilmek için bu yöntemi uygulayacağız
      """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "0"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                dataframe[col].dtypes != "0"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                dataframe[col].dtypes == "0"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations:{dataframe.shape[0]}")
    print(f"Variables:{dataframe.shape[1]}")
    print(f'cat_cols:"{len(cat_cols)}')
    print(f'num_cols:"{len(num_cols)}')
    print(f'cat_but_car:"{len(cat_but_car)}')
    print(f'num_but_cat:"{len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df) 

# NUMERIC

def numSummary(dataframe, numericalCol, plot=False):
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1]
    print(dataframe[numericalCol].describe(quantiles).T)

    if plot:
        dataframe[numericalCol].hist()
        plt.xlabel(numericalCol)
        plt.title(numericalCol)
        plt.show(block=True)


for col in num_cols:
    print(f"{col}:")
    numSummary(df, col, True)  
    
# CATEGORICAL

# is holiday "Yes" or "No" -> holiday distribution

sns.countplot(x="IsHoliday", data=df)
plt.title("Holiday Distribution")
plt.show() 

sns.countplot(x="Type", data=df) 
plt.title("What type of") 
plt.show()

print(cat_cols) 


# CORRELATION

corr = df[num_cols].corr()
print(corr)

# ısı haritası

sns.set(rc={'figure.figsize':(12,12)})
sns.heatmap(corr, cmap="RdBu", annot=True)
plt.show(block = True)  

######################## ANALYZE AND VISUALISATION #################################

# Assuming df is your DataFrame with a 'Date' column in the format 'MM/DD/YYYY'

df['Year'] = df.Date.apply(lambda x: int(str(x)[-4:]))
df['Month'] = df.Date.apply(lambda x: int(str(x)[:2]))
df['Year-Month'] = df.Date.apply(lambda x: str(x)[:7])
df['Day'] = df.Date.apply(lambda x: int(str(x)[3:5])) 


# Assuming df is your DataFrame with columns "Year", "Month", and "Fuel_Price"

# Create the "Year-Month" column by combining "Year" and "Month"

df['Year-Month'] = df['Year'].astype(str) + '-' + df['Month'].astype(str)

# Line Plot showing the change in fuel price over the span of 3 years 

plot_no = 1
plt.subplots(figsize=(20, 10))
plt.xticks(rotation=60)
sns.lineplot(data=df, x='Year-Month', y='Fuel_Price')
plt.title('Line Plot showing the change in fuel price over the span of 3 years', fontsize=20)
plt.savefig(str(plot_no) + '_plot.png')
plot_no += 1 

# BarPlot showing the change in fuel price with respect the type of the store with holidays grouped 

_ = plt.subplots(figsize = (20,10))
_ = plt.ylim(3.1,3.45)
plots = sns.barplot(data = df, x = 'IsHoliday', y = 'Fuel_Price', hue = 'Type')
_ = plt.title('BarPlot showing the change in fuel price with respect the type of the store with holidays grouped')
for bar in plots.patches:
    _ = plots.annotate(format(bar.get_height(), '.2f'), 
                   (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height() - (bar.get_height()-3.1)/2), ha='center', va='center',
                   size=15, xytext=(0, 0),bbox=dict(boxstyle="round4,pad=0.6", fc="w", ec="black", lw=2),
                   textcoords='offset points')
plt.savefig(str(plot_no)+'_plot.png')
plot_no +=1 


# LinePlot showing the change in fuel price with respect the type of the store 

_ = plt.subplots(figsize = (20,10))
_ = sns.lineplot(data = df, x = 'Type', y = 'Fuel_Price', hue = 'IsHoliday',style = 'IsHoliday', markers = True, ci = 68)
_ = plt.title('LinePlot showing the change in fuel price with respect the type of the store')
plt.savefig(str(plot_no)+'_plot.png')
plot_no +=1 

# Lineplot showing the change in fuel price with respect to the change in temperature 
# boş döndürüyor resmi 

_ = plt.subplots(figsize=(20, 10))
_ = sns.lineplot(data=df, x='Correct_Column_Name', y='Fuel_Price', hue='IsHoliday', style='IsHoliday', markers=True, ci=68)
_ = plt.xlabel('Temperature range')
_ = plt.title('Lineplot showing the change in fuel price with respect to the change in temperature')
plt.savefig(str(plot_no)+'_plot.png')
plot_no +=1 

# Lineplot showing the change in fuel price in each month over the span of 3 years

_ = plt.subplots(figsize = (20,10))
_ = sns.lineplot(data = df, x = 'Date', y = 'Fuel_Price')
_ = plt.title('Lineplot showing the change in fuel price in each month over the span of 3 years')
plt.savefig(str(plot_no)+'_plot.png')
plot_no +=1 

# Barplot showing the observation counts for each recorded month

_ = plt.subplots(figsize = (20,10))
_ = sns.countplot(data = df,x='Year',hue='Month')
_ = plt.title('Barplot showing the observation counts for each recorded month')
plt.savefig(str(plot_no)+'_plot.png')
plot_no +=1

# Barplot showing the change in Fuel price with respect to the type of the store

_ = plt.subplots(figsize = (20,10))
plots = sns.barplot(data = df, x = 'Type', y = 'Fuel_Price')
for bar in plots.patches:
    _ = plots.annotate(format(bar.get_height(), '.2f'), 
                   (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 23),
                   textcoords='offset points');
_ = plt.ylim(2.5,3.5)
_ = plt.title('Barplot showing the change in Fuel price with respect to the type of the store')
plt.savefig(str(plot_no)+'_plot.png')
plot_no +=1 


df_rolled_mean = df.set_index('Date').rolling(window = 2948).mean().reset_index()
df_rolled_std = df.set_index('Date').rolling(window = 2948).std().reset_index()  

#Lineplot showing the change in Weekly_Sales in each month over the span of 3 years

fig,ax = plt.subplots(figsize = (20,10))
_ = sns.lineplot(data = df, x = 'Year-Month', y = 'Weekly_Sales', ax = ax, ci = 1)
_ = plt.xticks(rotation = 60)
_ = plt.title('Lineplot showing the change in Weekly_Sales in each month over the span of 3 years')
plt.savefig(str(plot_no)+'_plot.png')
plot_no +=1 

# there was peak during end of the years 2010 and 2011 but not during 2012

_ = df[['Date', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']].plot(x = 'Date', subplots = True, figsize = (20,15))
plt.savefig(str(plot_no)+'_plot.png')
plot_no +=1 

# Data spread of total weekly sales volume of the retail chain 

df_average_sales_week = df.groupby(by=['Date'], as_index=False)['Weekly_Sales'].sum()
df_average_sales = df_average_sales_week.sort_values('Weekly_Sales', ascending=False)

_ = plt.figure(figsize=(20,8))
_ = plt.plot(df_average_sales_week.Date, df_average_sales_week.Weekly_Sales)
_ = plt.title('Data spread of total weekly sales volume of the retail chain')
_ = plt.xlabel('Date')
_ = plt.ylabel('Weekly Sales')
plt.savefig(str(plot_no)+'_plot.png')
plot_no +=1 

# top stories 

ts = df_average_sales_week.set_index('Date')
# Top performing type of stores in term of sales
df_top_stores = df.groupby(by=['Type'], as_index=False)['Weekly_Sales'].sum()
df_top_stores.sort_values('Weekly_Sales', ascending=False) 

# top performing stores in term of sales 

# Top performing stores in term of sales
df_top_stores = df.groupby(by=['Store'], as_index=False)['Weekly_Sales'].sum()
df_top_stores.sort_values('Weekly_Sales', ascending=False)[:3] 

######################## OUTLIERS ########################### 


def outliers_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quantile1 = dataframe[col_name].quantile(q1)
    quantile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quantile3 - quantile1
    up_limit = quantile3 + 1.5 * interquantile_range
    low_limit = quantile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outliers_thresholds(dataframe, col_name)
    # yukarıda yapılan any yani bool ile herhangi bir aykırı değer var mı yok mu sorusuna denk gelir
    if dataframe[(dataframe[col_name]>up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col , "=>", check_outlier(df,col))


def grab_outlier(dataframe, col_name, index = False):
    low,up = outliers_thresholds(dataframe,col_name)
    if dataframe[(dataframe[col_name]>low) | (dataframe[col_name] < up)].shape[0] > 10:
        print(dataframe[(dataframe[col_name]<low)| (dataframe[col_name] > up)].head())

    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name]> up))])
        if index:
            outlier_index = dataframe[((dataframe[col_name]< low) | (dataframe[col_name]> up))].index
            return outlier_index

        for col in num_cols:
            print(col, grab_outlier(df, col, True))  
            

# Remove outliers 

def remove_outliers(dataframe, col_name):
    low_limit, up_limit = outliers_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit ) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

for col in num_cols:
        df = remove_outliers(df,col)

for col in num_cols:
    print(col,check_outlier(df,col))  
    

# Check missing values 

def missing_values_table(dataframe,na_name = False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss,np.round(ratio,2)], axis=1, keys=['n_miss','ratio'])
    print(missing_df,end="\n")

    if na_name:
        return na_columns

missing_values_table(df) 

########################### ENCODING ##################### 



df = pd.get_dummies(df,columns=["IsHoliday","Type"], drop_first=True)

print(df.head())

print(df.info()) 

# List of column names to be deleted
columns_to_delete = ['Date', 'Year-Month']

# Drop the specified columns from the DataFrame
df = df.drop(columns_to_delete, axis=1) 

print(df.info()) 


# List of boolean columns to be converted to integers
boolean_columns = ['IsHoliday_True', 'Type_B', 'Type_C']

# Convert boolean columns to integers
df[boolean_columns] = df[boolean_columns].astype(int) 

print(df.info()) 


######################## ML MODELLING ################### 


# Assuming your DataFrame is named 'df'

# Select the independent variables (X) and dependent variable (y)
X = df[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']]  # Independent variables
y = df['Weekly_Sales']  # Dependent variable

# Apply a logarithmic transformation to selected independent variables
X_transformed = X.copy()
X_transformed[['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']] = np.log1p(X_transformed[['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']])

# No transformation for CPI
X_transformed['CPI'] = X['CPI']

# Apply a logarithmic transformation to the dependent variable
y_transformed = np.log1p(y)



#X = df.drop('Weekly_Sales', axis=1)
#y = df['Weekly_Sales'] 

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# Scale the features using StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) 


# Create an instance of the LinearRegression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation scores on the test set
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Print evaluation scores
print('R-squared:', r2)
print('Mean Squared Error:', mse)

# Plot the actual test data points and the predicted values
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('Actual Weekly Sales')
plt.ylabel('Predicted Weekly Sales')
plt.title('Multiple Linear Regression')
plt.show() 

# R-squared: -9.805689862529832e-06 
# Mean Squared Error: 123803168.81633493 

# OLS Regression 

reg_ols = sm.OLS(y, X)
est = reg_ols.fit()
est.summary()  

y_pred = est.predict(X)

r_2 = r2_score(y, y_pred)
rmse = mean_squared_error(y, y_pred, squared=False)
mae = mean_absolute_error(y, y_pred)

print("R squared:", r_2)
print("RMSE:", rmse)
print("MAE:", mae) 

# R squared: -0.8812265468674796 
# RMSE: 15218.217083933321 
# MAE: 10416.089033491802 

y_test_pred = est.predict(X_test)

r_2 = r2_score(y_test, y_test_pred)
rmse = mean_squared_error(y_test, y_test_pred, squared=False)
mae = mean_absolute_error(y_test, y_test_pred)

print("R squared:", r_2)
print("RMSE:", rmse)
print("MAE:", mae) 

# R squared: -0.8812265468674796
# RMSE: 15218.217083933321 
# MAE: 10416.089033491802 

# Lasso, Ridge Regression   

# Lasso

reg_lasso = Lasso().fit(X,y)

y_pred = reg_lasso.predict(X)

r_2 = reg_lasso.score(X, y)
rmse = mean_squared_error(y, y_pred, squared=False)
mae = mean_absolute_error(y, y_pred)

print("R squared:", r_2)
print("RMSE:", rmse)
print("MAE:", mae) 

# R squared: 0.0
# RMSE: 11095.406693403622
# MAE: 8776.059471864835 

# Ridge Regression 

reg_ridge = Ridge().fit(X,y)

y_pred = reg_ridge.predict(X)

r_2 = reg_ridge.score(X, y)
rmse = mean_squared_error(y, y_pred, squared=False)
mae = mean_absolute_error(y, y_pred)

print("R squared:", r_2)
print("RMSE:", rmse)
print("MAE:", mae)  

# R squared: 0.0 
# RMSE: 11095.406693403622 
# MAE: 8776.059471864835  

y_test_pred = reg_ridge.predict(X_test)

r_2 = reg_ridge.score(X_test, y_test)
rmse = mean_squared_error(y_test, y_test_pred, squared=False)
mae = mean_absolute_error(y_test, y_test_pred)

print("R squared:", r_2)
print("RMSE:", rmse)
print("MAE:", mae) 

# R squared: -6.27558569421538e-06 
# RMSE: 11126.667595580362 
# MAE: 8797.247159626953 

# GBDT, XGBoost and LightBGM  
 
 # GBDT
 
reg_gbdt = GradientBoostingRegressor().fit(X,y)

y_pred = reg_gbdt.predict(X)

r_2 = reg_gbdt.score(X, y)
rmse = mean_squared_error(y, y_pred, squared=False)
mae = mean_absolute_error(y, y_pred)

print("R squared:", r_2)
print("RMSE:", rmse)
print("MAE:", mae) 

# R squared: 4.440892098500626e-16
# RMSE: 11095.40669340362
# MAE: 8776.059471864795 

y_test_pred = reg_gbdt.predict(X_test)

r_2 = reg_gbdt.score(X_test, y_test)
rmse = mean_squared_error(y_test, y_test_pred, squared=False)
mae = mean_absolute_error(y_test, y_test_pred)

print("R squared:", r_2)
print("RMSE:", rmse)
print("MAE:", mae) 

# R squared: -6.27558569421538e-06
# RMSE: 11126.667595580362 
# MAE: 8797.247159626913 

# XGBoost

reg_xgb = XGBRegressor(objective='reg:squarederror', n_estimators=2000)
reg_xgb = reg_xgb.fit(X,y)

y_pred = reg_xgb.predict(X)

r2 = reg_xgb.score(X, y)
rmse = mean_squared_error(y, y_pred, squared=False)
mae = mean_absolute_error(y, y_pred)

print("R squared:", r_2)
print("RMSE:", rmse)
print("MAE:", mae)  


# R squared: -6.27558569421538e-06
# RMSE: 11095.406693403678 
# MAE: 8776.059165911107 

y_test_pred = reg_xgb.predict(X_test)

r2_test = reg_xgb.score(X_test, y_test)
rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
mae_test = mean_absolute_error(y_test, y_test_pred)

print("R squared:", r2_test)
print("RMSE:", rmse_test)
print("MAE:", mae_test)  

# R squared: -6.2750742939599036e-06 
# RMSE: 11126.667592735288 
# MAE: 8797.246849436051  

# LightGBM 

reg_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=60, max_depth = 9, learning_rate=0.5, n_estimators=2000, reg_alpha=0.6, subsample=0.6, colsample_bytree = 0.8, scale_pos_weight = 5)
reg_lgb.fit(X, y, verbose=False) 

y_pred = reg_lgb.predict(X)

r_2 = reg_lgb.score(X, y)
rmse = mean_squared_error(y, y_pred, squared=False)
mae = mean_absolute_error(y, y_pred)

print("R squared:", r_2)
print("RMSE:", rmse)
print("MAE:", mae) 


# R squared: 6.661338147750939e-16
# RMSE: 11095.406693403618
# MAE: 8776.059471672612 


y_test_pred = reg_lgb.predict(X_test)

r2_test = reg_lgb.score(X_test, y_test)
rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
mae_test = mean_absolute_error(y_test, y_test_pred)

print("R squared:", r2_test)
print("RMSE:", rmse_test)
print("MAE:", mae_test)  


# R squared: -6.275585373138881e-06
# RMSE: 11126.667595578574
# MAE: 8797.247159432069 

# Time Series 



