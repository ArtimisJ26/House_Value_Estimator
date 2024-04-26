import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
import joblib
# pip install pandas
# pip install -U scikit-learn
# Write just "import joblib" instead of "from sklearn.externals import joblib"

# Load the data set
df = pd.read_csv("ml_house_data_set.csv")

'''
## Feature engineering: 
-> Drop unrelated features
-> Combine multiple features that quantify to the same thing eg: 2'4"-> 28"
-> Replace categorical data with one-hot encoding data  
'''

# Remove the fields from the data set that we don't waht to include in our model
del df['house_number']
del df['unit_number']
del df['street_name']
del df['zip_code']

# Replace categorical data with one-hot encoded data
features_df = pd.get_dummies(df, columns=['garage_type', 'city'])

# Remove the sale price (variable to be predicted) from the feature data
del features_df['sale_price']

# print(features_df)

'''
## Data split and model training
-> Create X and y arrays
-> Split data into train and test sets
-> Train the model
'''

# Create the X and y arrays
# Use "to_numpy()" instead of "as_matrix())"
X = features_df.to_numpy()
y = df['sale_price'].to_numpy()

'''
## The lecture did a split of 70%-30%, whereas I did a split of 80%-20% which resulted in a better training and test accuracy i.e. lower Mean absolute error
70-30 Split output: 
    Training Set Mean Absolute Error: 48727.0015
    Test Set Mean Absolute Error: 59225.2075

80-20 Split output:
    Training Set Mean Absolute Error: 48678.9051
    Test Set Mean Absolute Error: 58710.3788
'''
# Split the data set in training set (70%) and testing set (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit the regression model
model = ensemble.GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    min_samples_leaf=9,
    max_features=0.1,
    loss='huber',
    random_state=0
)
model.fit(X_train, y_train)

# Save the trained model to a file so we can use it in other programs
joblib.dump(model, 'trained_house_classifier_model.pkl')

'''
## Checking the accuracy of the trained model
-> Check Training accuracy
-> Check Test accuracy
'''

# Find the error rate on the training set
mse = mean_absolute_error(y_train, model.predict(X_train))
print("Training Set Mean Absolute Error: %.4f" % mse)

# Find the error rate on the test set
mse = mean_absolute_error(y_test, model.predict(X_test))
print("Test Set Mean Absolute Error: %.4f" % mse)