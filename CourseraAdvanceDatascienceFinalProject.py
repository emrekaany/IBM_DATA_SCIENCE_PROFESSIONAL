# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 12:10:28 2024

@author: ky4642
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from bayes_opt import BayesianOptimization
import numpy as np
from sklearn.metrics import mean_squared_error
import mpld3
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression



# Replace 'path_to_file.csv' with the path to the CSV file you want to read
file_path = 'C:\\Users\ky4642\Downloads\insurance.csv'
df = pd.read_csv(file_path)




#####
#Part: Data Quality Checks
#####

print("Corrupts in data are as following")

# Checking for missing values
print(f"Number of null values found: {df.isnull().sum()}")

# Checking for duplicates
print(f"Number of duplicated values found:   {df.duplicated().sum()}")

# Check for age being 18 or older
invalid_age = df[df['age'] < 0]

# Check for valid BMI range
invalid_bmi = df[(df['bmi'] < 10) | (df['bmi'] > 50)]

# Check for non-negative and plausible children count
invalid_children = df[df['children'] < 0]

# Output results
print(f"Number of invalid ages found: {invalid_age.shape[0]}")
print(f"Number of invalid BMI values found: {invalid_bmi.shape[0]}")
print(f"Invalid children counts found: {invalid_children.shape[0]}")


#We remove corrupt rows
df=df.dropna()
df=df[df['age']>0]
df=df[(df['bmi'] > 10) & (df['bmi'] < 50)]
df = df[df['children'] > 0]



########
#Part: we encode dataframe and create feature of the number of times gave birth
########

for column in [ 'sex', 'smoker', 'region']:
    df[column+'_encoded']  = df[column].astype('category').cat.codes

# Display the first few rows of the DataFrame to confirm it's loaded correctly
print(df.head())

df['number_of_times_gave_birth']= (1- df['sex_encoded'])* df['children']

df_featured= df.drop(columns=['sex', 'smoker', 'region' ])
print(df_featured)


# First, I determine unuseful columns for feature selection 
correlation_matrix = df_featured.corr()
correlation_with_target = correlation_matrix['charges'].sort_values()

# Print correlations with the target variable
print(correlation_with_target)

# Optionally, drop columns with low correlation
low_correlation_cols = correlation_with_target[correlation_with_target.abs() < 0.1].index  # threshold can vary
df_featured = df_featured.drop(columns=low_correlation_cols)


#########
# Part: Plots
#########

# Create a list to hold HTML strings for each plot
plots_html = []

plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='charges', hue='smoker_encoded', data=df, palette=['red', 'green'], style='smoker_encoded', markers=['o', 'X'])
plt.title('Age vs. Charges by Smoking Status')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.legend(title='Smoker')
plt.close()



# Plot 4: Box Plot for Charges by Region
plt.figure(figsize=(12, 6))
sns.boxplot(x='region_encoded', y='charges', data=df)
plt.title('Medical Charges by Region')
plt.xlabel('Region')
plt.ylabel('Charges')
plots_html.append(mpld3.fig_to_html(plt.gcf()))
plt.close()

# Plot 5: Heat Map of Correlation Matrix
plt.figure(figsize=(10, 8))
columns = ['age', 'bmi', 'children', 'charges', 'sex_encoded', 'smoker_encoded', 'region_encoded', 'number_of_times_gave_birth']
corr_matrix = df[['age', 'bmi', 'children', 'charges', 'sex_encoded', 'smoker_encoded', 'region_encoded', 'number_of_times_gave_birth']].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', xticklabels=columns, yticklabels=columns)
plt.title('Correlation Matrix')
ticks = np.arange(len(columns)) + 0.5
plt.xticks(ticks=ticks,labels= columns,rotation=45)
plt.yticks(ticks=ticks,labels= columns,rotation=0)
plots_html.append(mpld3.fig_to_html(plt.gcf()))
plt.close()

# Plot 6: Scatter Plot for BMI vs. Charges
plt.figure(figsize=(10, 6))
sns.scatterplot(x='bmi', y='charges', data=df, hue='smoker_encoded')
plt.title('BMI vs. Charges by Smoking Status')
plt.xlabel('BMI')
plt.ylabel('Charges')
plots_html.append(mpld3.fig_to_html(plt.gcf()))
plt.close()

# Plot 7: Line Plot for Average Charges by Age (Grouped by Smoker Status)
# Calculating average charges by age for each smoker status
average_charges = df.groupby(['age', 'smoker_encoded'])['charges'].mean().unstack()
plt.figure(figsize=(12, 6))
average_charges.plot(kind='line')
plt.title('Average Charges by Age and Smoker Status')
plt.xlabel('Age')
plt.ylabel('Average Charges')
plt.legend(title='Smoker')
plots_html.append(mpld3.fig_to_html(plt.gcf()))
plt.close()


    
X = df_featured.drop(columns=['charges'])  # Features
y = df_featured['charges']  # Target variable
y = np.array(y).reshape(-1, 1)
Xscaler = MinMaxScaler()
Yscaler=MinMaxScaler()
X=pd.DataFrame(Xscaler.fit_transform(X), columns=X.columns)
y=pd.DataFrame(Yscaler.fit_transform(y))


# Split data into train and validation/test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# Split temp into validation and test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


df_performances=pd.DataFrame( {
    'Model': [],
    'Training_MAPE': [],
    'Validation_MAPE': [],
    'Testing_MAPE': []
})

def keras_model(n_units, dropout_rate, optimizer_index):
    # Map optimizer index to optimizer name
    optimizers = ['adam', 'sgd']
    optimizer = optimizers[int(optimizer_index)]
    
    # Build model
    model = Sequential([
        Dense(int(n_units), activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(dropout_rate),
        Dense(int(n_units), activation='relu'),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    # Fit model
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_data=(X_val, y_val))
    
    # Evaluate model
    mse = model.evaluate(X_val, y_val, verbose=0)
    return -mse  # Negative MSE because BayesianOptimization maximizes the function

def objective(n_units, dropout_rate, optimizer_index):
    """ Wrapper of the model that ensures the parameters are in the correct format """
    return keras_model(
        n_units=n_units,
        dropout_rate=dropout_rate,
        optimizer_index=optimizer_index
    )


pbounds = {
    'n_units': (32, 128),  # Example: number of units in a layer
    'dropout_rate': (0.1, 0.5),  # Dropout rate
    'optimizer_index': (0, 1)  # Index to select optimizer
}



optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=2,  # Number of random exploratory steps before fitting the Gaussian process
    n_iter=10,  # Number of optimization steps
)



print("Best parameters: ", optimizer.max['params'])


best_parameters=optimizer.max['params']

model = Sequential([
    Dense(int(best_parameters ['n_units']), activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(best_parameters ['dropout_rate']),
    Dense(int(best_parameters ['n_units']), activation='relu'),
    Dropout(best_parameters ['dropout_rate']),
    Dense(1)
])
model.compile(loss='mean_squared_error')

# Fit model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_data=(X_val, y_val))

#Error_for_test= model.evaluate(X_test, y_test, verbose=0 )

y_pred= model.predict(X_test)

y_pred=Yscaler.inverse_transform(y_pred)
y_test= Yscaler.inverse_transform(y_test)
# MAPE hesaplama fonksiyonu
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# MAPE hesaplama
mape = mean_absolute_percentage_error(y_test, y_pred)

print("MAPE ratio for Fully Connected Neural Network model on feature selected data is :" + str(mape))

# New row DataFrame with scalar values wrapped in lists
new_row = pd.DataFrame({
    'Model': ['Fully Connected Neural Network model on feature selected data'],
    'Training_MAPE': [mean_absolute_percentage_error(Yscaler.inverse_transform(y_train), Yscaler.inverse_transform(model.predict(X_train)))],
    'Validation_MAPE': [mean_absolute_percentage_error(Yscaler.inverse_transform(y_val), Yscaler.inverse_transform(model.predict(X_val)))],
    'Testing_MAPE': [mape]
})

# Concatenate the new row
df_performances = pd.concat([df_performances, new_row], ignore_index=True)







# Split data into train and validation/test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# Split temp into validation and test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr= lr.predict(X_test)
y_pred_lr=Yscaler.inverse_transform(y_pred_lr)
y_test= Yscaler.inverse_transform(y_test)
mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr)
print("MAPE ratio for Linear Regression model on feature selected data is :" + str(mape_lr))

# New row DataFrame with scalar values wrapped in lists
new_row = pd.DataFrame({
    'Model': ['Linear Regression model on feature selected data'],
    'Training_MAPE': [mean_absolute_percentage_error(Yscaler.inverse_transform(y_train), Yscaler.inverse_transform(lr.predict(X_train)))],
    'Validation_MAPE': [mean_absolute_percentage_error(Yscaler.inverse_transform(y_val), Yscaler.inverse_transform(lr.predict(X_val)))],
    'Testing_MAPE': [mape_lr]
})

# Concatenate the new row
df_performances = pd.concat([df_performances, new_row], ignore_index=True)






# Split data into train and validation/test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# Split temp into validation and test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_RF=[ rf.predict(X_test)]
y_pred_RF=Yscaler.inverse_transform(y_pred_RF)
y_test= Yscaler.inverse_transform(y_test)
mape_RF = mean_absolute_percentage_error(y_test, y_pred_RF)
print("MAPE ratio for Random Forest Regressor model on feature selected data is :" + str(mape_RF))



# New row DataFrame with scalar values wrapped in lists
new_row = pd.DataFrame({
    'Model': ['Random Forest Regressor model on feature selected data'],
    'Training_MAPE': [mean_absolute_percentage_error(Yscaler.inverse_transform(y_train.to_numpy().reshape(-1, 1)), Yscaler.inverse_transform(rf.predict(X_train).reshape(-1, 1)))],
    'Validation_MAPE': [mean_absolute_percentage_error(Yscaler.inverse_transform(y_val.to_numpy().reshape(-1, 1)), Yscaler.inverse_transform(rf.predict(X_val).reshape(-1, 1)))],
    'Testing_MAPE': [mape_RF]
})

# Concatenate the new row
df_performances = pd.concat([df_performances, new_row], ignore_index=True)






df_featured2= df.drop(columns=['sex', 'smoker', 'region' ])
print(df_featured2)


   
X2 = df_featured2.drop(columns=['charges'])  # Features
y2 = df_featured2['charges']  # Target variable
y2 = np.array(y).reshape(-1, 1)
Xscaler2 = MinMaxScaler()
Yscaler2=MinMaxScaler()
X2=pd.DataFrame(Xscaler2.fit_transform(X2), columns=X2.columns)
y2=pd.DataFrame(Yscaler2.fit_transform(y2))






# Split data into train and validation/test
X_train, X_temp, y_train, y_temp = train_test_split(X2, y2, test_size=0.3, random_state=42)

# Split temp into validation and test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

def keras_model(n_units, dropout_rate, optimizer_index):
    # Map optimizer index to optimizer name
    optimizers = ['adam', 'sgd']
    optimizer = optimizers[int(optimizer_index)]
    
    # Build model
    model = Sequential([
        Dense(int(n_units), activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(dropout_rate),
        Dense(int(n_units), activation='relu'),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    # Fit model
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_data=(X_val, y_val))
    
    # Evaluate model
    mse = model.evaluate(X_val, y_val, verbose=0)
    return -mse  # Negative MSE because BayesianOptimization maximizes the function

def objective(n_units, dropout_rate, optimizer_index):
    """ Wrapper of the model that ensures the parameters are in the correct format """
    return keras_model(
        n_units=n_units,
        dropout_rate=dropout_rate,
        optimizer_index=optimizer_index
    )



pbounds = {
    'n_units': (32, 128),  # Example: number of units in a layer
    'dropout_rate': (0.1, 0.5),  # Dropout rate
    'optimizer_index': (0, 1)  # Index to select optimizer
}



optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=2,  # Number of random exploratory steps before fitting the Gaussian process
    n_iter=10,  # Number of optimization steps
)



print("Best parameters: ", optimizer.max['params'])


best_parameters=optimizer.max['params']

model = Sequential([
    Dense(int(best_parameters ['n_units']), activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(best_parameters ['dropout_rate']),
    Dense(int(best_parameters ['n_units']), activation='relu'),
    Dropout(best_parameters ['dropout_rate']),
    Dense(1)
])
model.compile(loss='mean_squared_error')

# Fit model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_data=(X_val, y_val))

#Error_for_test= model.evaluate(X_test, y_test, verbose=0 )

y_pred= model.predict(X_test)

y_pred=Yscaler.inverse_transform(y_pred)
y_test= Yscaler.inverse_transform(y_test)
# MAPE hesaplama
mape = mean_absolute_percentage_error(y_test, y_pred)

print("MAPE ratio for Fully Connected Neural Network model on feature gerenated data is :" + str(mape))

# New row DataFrame with scalar values wrapped in lists
new_row = pd.DataFrame({
    'Model': ['Fully Connected Neural Network model on feature gerenated data'],
    'Training_MAPE': [mean_absolute_percentage_error(Yscaler2.inverse_transform(y_train), Yscaler2.inverse_transform(model.predict(X_train)))],
    'Validation_MAPE': [mean_absolute_percentage_error(Yscaler2.inverse_transform(y_val), Yscaler2.inverse_transform(model.predict(X_val)))],
    'Testing_MAPE': [mape]
})

# Concatenate the new row
df_performances = pd.concat([df_performances, new_row], ignore_index=True)






# Split data into train and validation/test
X_train, X_temp, y_train, y_temp = train_test_split(X2, y2, test_size=0.3, random_state=42)
# Split temp into validation and test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr= lr.predict(X_test)
y_pred_lr=Yscaler.inverse_transform(y_pred_lr)
y_test= Yscaler.inverse_transform(y_test)
mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr)
print("MAPE ratio for Linear Regression model on feature gerenated data is :" + str(mape_lr))



# New row DataFrame with scalar values wrapped in lists
new_row = pd.DataFrame({
    'Model': ['Linear Regression model on feature generated data'],
    'Training_MAPE': [mean_absolute_percentage_error(Yscaler2.inverse_transform(y_train), Yscaler2.inverse_transform(lr.predict(X_train)))],
    'Validation_MAPE': [mean_absolute_percentage_error(Yscaler2.inverse_transform(y_val), Yscaler2.inverse_transform(lr.predict(X_val)))],
    'Testing_MAPE': [mape_lr]
})

# Concatenate the new row
df_performances = pd.concat([df_performances, new_row], ignore_index=True)






# Split data into train and validation/test
X_train, X_temp, y_train, y_temp = train_test_split(X2, y2, test_size=0.3, random_state=42)
# Split temp into validation and test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_RF=[ rf.predict(X_test)]
y_pred_RF=Yscaler.inverse_transform(y_pred_RF)
y_test= Yscaler.inverse_transform(y_test)
mape_RF = mean_absolute_percentage_error(y_test, y_pred_RF)
print("MAPE ratio for Random Forest Regressor model on feature gerenated data is :" + str(mape_RF))



# New row DataFrame with scalar values wrapped in lists
new_row = pd.DataFrame({
    'Model': ['Random Forest Regressor model on feature generated data'],
    'Training_MAPE': [mean_absolute_percentage_error(Yscaler2.inverse_transform(y_train.to_numpy().reshape(-1, 1)), Yscaler2.inverse_transform(rf.predict(X_train).reshape(-1, 1)))],
    'Validation_MAPE': [mean_absolute_percentage_error(Yscaler2.inverse_transform(y_val.to_numpy().reshape(-1, 1)), Yscaler2.inverse_transform(rf.predict(X_val).reshape(-1, 1)))],
    'Testing_MAPE': [mape_RF]
})

# Concatenate the new row
df_performances = pd.concat([df_performances, new_row], ignore_index=True)



plots_html.append(df_performances.to_html(index=False))

# Save all plots into an HTML file
html_file = '<html><head><title>Insurance Data Visualizations</title></head><body>'
html_file += "\n".join(plots_html)
html_file += '</body></html>'

with open('C:\\Users\\ky4642\\Pictures\\visualizations.html', 'w') as f:
    f.write(html_file)
    
    