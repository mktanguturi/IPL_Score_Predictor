import matplotlib
import numpy as np
import pandas as pd
import datetime as datetime

from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import pickle as pkl

print("Import successful\n")


# Read the data into pandas data frame
data = pd.read_csv("ipl.csv")


# Display the data to understand how the data looks and columns available. 
# This helps in getting better idea about the data
print("Sample Data:\n", data.head(10))

# Know the size of data we have at hand
print("Data Shape", data.shape)

# data type of each column need to be understood. 
print("Data types of each column:\n", data.info())

# Check how many Null values exist in the data. We should handle these Null values else it will impact the model outcome
print("Total Null values in the data:\n", data.isnull().sum())

# Obtain summary statistics to understand how good is data. 

## 1. Measurement of central tendency. (Mean, Median, Mode & Frequency)
## 2. Measurement of dispersion of the data (Min/Max, Range, Variance, Std Dev, Percentile)
### Variance measures the variation in the data while Percentile captures spread of data for a specific value i.e
### above and below a particular point. 
## Check how much the range differs between columns. If the difference in ranges is not high, we have good 
## data at hand.
print("Summary Statistics of our data:\n", data.describe())


# Data cleaning

## Step1: Unnecessary column removal. Here we need to identify the columns we think are of no use and drop them

col_to_remove = ['mid' , 'venue' , 'batsman', 'bowler', 'striker', 'non-striker']

data.drop(col_to_remove, axis=1, inplace=True)

print("Dataset post removal of unnecessary columns:\n", data.head())
# Shape should be checked to ensure we did not mess up the data in any way in our removal step.
print("Dataset shape post removal of unnecessary columns:", data.shape)



# Filterout the inconsistent Teams i.e. pick only those teams which are playing IPL consistently. 
# In a way we are removing outliers from Teams point of view. 

# Find all unique teams.

print("Unique Teams in the data: \n", data['bat_team'].unique())

print("-----------------------\n")

# We know that there are teams which played only in one or two IPLs. Remove those and select only the ones which 
# have played consistently. 

consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals', 'Mumbai Indians', 
'Kings XI Punjab', 'Royal Challengers Bangalore', 'Delhi Daredevils', 'Sunrisers Hyderabad']

data = data[(data['bat_team'].isin(consistent_teams)) & (data['bowl_team'].isin(consistent_teams))]

print("Teams in Dataset post removal of inconsistent Teams:\n")
print("Batting Teams:\n", data['bat_team'].unique())
print("Bowling Teams:\n", data['bowl_team'].unique())


# Data can be used as a way to split the data into train and test. 
# Eg: All matches played before 2016 can be training data and 2017 on wards can be test data. 

# Since date is now shown as string data in our pandas frames, we should convert it into int so that
# we can use the date to split the data into train and test. 

# convert the date into int. 

data['date'] = data['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))

print("Data after converting data into int64:\n")
print(data.head(3))
print(data.info())

# Consider data only for overs > 5. This will give us better possibility of predition. 
print("Data before pruning < 5 overs:",data.size)
data = data[data['overs'] > 5.0]
print("Data after pruning < 5 overs:",data.size)

# Convert all the string columns into numeric as models only understand numeric data.
# Use ONE HOT Encoding. 

# There are 2 columns bat_team & bowl_team that need to be converted. 

categorical_data = pd.get_dummies(data=data, columns=['bat_team', 'bowl_team'])

print("Converted categorical value into numeric:\n", categorical_data.head(3))
print("Datasize post one-hot encoding: ", categorical_data.size)


# Split the data into test and train based on year. 
# All data upto 2016 will be train data.
# All data upto 2017 will be test data. 

train_x = categorical_data.drop(labels='total', axis=1)[categorical_data['date'].dt.year <= 2016]
test_x = categorical_data.drop(labels='total', axis=1)[categorical_data['date'].dt.year >= 2017]

print("Train_X Shape:", train_x.shape)
print("Test_X Shape:", test_x.shape)

train_y = categorical_data[categorical_data['date'].dt.year <= 2016]['total'].values
test_y = categorical_data[categorical_data['date'].dt.year >= 2017]['total'].values

print("Train_Y Shape:", train_y.shape)
print("Test_Y Shape:", test_y.shape)


# We no longer need date field now as it adds no value post split of data. 
train_x.drop(labels='date', axis=1, inplace=True)
test_x.drop(labels='date', axis=1, inplace=True)
print("Post removal of date column: \n")
print("Train_X Shape:", train_x.shape)
print("Test_X Shape:", test_x.shape)


# Use Linear Regression as this is a regression problem.
lr_ipl = LinearRegression()

# Fit the training data
lr_ipl.fit(train_x, train_y)

print("Linear Regression Coefs: ", lr_ipl.coef_)

# Predict using the model and params from above steps.
pred_ipl = lr_ipl.predict(test_x)

# Visualize the results. 
print("Display the predicted Vs actual values:\n")
print("The line shows predicted data and bars show the actual data\n")
sns.distplot(test_y-pred_ipl)
#plt.show()


# Check the metrics to understand how good or bad is our model. 
print("\nMean Absolute Error: ", mean_absolute_error(test_y, pred_ipl))
print("\nMean Squared Error: ", mean_squared_error(test_y, pred_ipl))
print("\nR^2 Error: ", r2_score(test_y, pred_ipl))
print("\nRMSE: ", np.sqrt(mean_squared_error(test_y, pred_ipl)))


# Save the Model using pickle Lib. 
print("Saving the Model using Pickle:\n")
filename = "IPL_Score_Predictor.pkl"
pkl.dump(lr_ipl, open(filename, 'wb'))

# Load the pickle file and check if the coeffients are proper. 
load_lr = pkl.load(open(filename, 'rb'))
print("Coefficients after loading the Model from pkl file: ", load_lr.coef_)
#print(test_x.iloc[1])
Predict_score = load_lr.predict(np.array(test_x.iloc[10]).reshape(1, -1))
print("Predict a value: ", Predict_score)
print("Actual Score: ", test_y[10])



# Deploy the Model. 
''' 
Parameters:
1. Application
2. Data
3. Runtime
4. Middle-ware
5. OS
6. Server
7. Services
8. Storage
9. Networking

Three types of deployment for a Model:
1. Local: Here user needs to worry about all 1 to 6 parameters. 7 to 9 are not applicable. 
2. IAAS: Infrastructure as Service: Here User need to worry about 1, 2, 3 & 5. 4 is not applicable. 7,8 & 9 are taken care by the provider. 
3. PAAS: Platform as Service. Here User need to worry about 1 & 2. 4 is not applicable. 3, 5, 7,8 & 9 are taken care by the provider. 

Hence PAAS is best way to deploy the model. 

There are free platforms that offer PAAS. Heroku is one such case. 

'''

