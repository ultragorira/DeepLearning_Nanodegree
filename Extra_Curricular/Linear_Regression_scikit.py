# TODO: Add import statements
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Assign the dataframe to this variable.
# TODO: Load the data
bmi_life_data = pd.read_csv(r'C:\Users\ultra\Python\DeepLearning_Nanodegree\Extra_Curricular\bmi_and_life_expectancy.csv')
x_values, y_values = bmi_life_data[['BMI']], bmi_life_data[['Life expectancy']]

# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model
bmi_life_model = LinearRegression()
bmi_life_model.fit(x_values.values, y_values.values)


# Make a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict([[21.07931]])
print(laos_life_exp)
