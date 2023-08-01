# Data Science using NumPy and Pandas

I start by loading the dataset from a given URL using Pandas, then perform data preprocessing by removing rows with missing values and converting the 'Date' column to datetime type. I also extract the year from the 'Date' column and create a new 'Year' column.

Next, I perform feature engineering by defining a custom function extract_features() that calculates a new feature based on existing columns and apply this function to the DataFrame and store the results in a new 'CustomFeature' column.

After feature engineering, I split the dataset into training and testing sets using train_test_split() from scikit-learn amd then create a Linear Regression model and train it on the training data.

I evaluate the model's performance on the testing data by calculating Mean Squared Error (MSE) and R-squared (R2) using scikit-learn's metrics. Additionally, I print the feature importance, which shows the coefficients (weights) of each feature in the linear regression model.

Finally, I save the trained model using joblib to a file named 'linear_regression_model.pkl'.

Note: This code assumes that you have a dataset in CSV format available at a given URL and that the dataset contains columns 'Date', 'Feature1', 'Feature2', 'Feature3', 'CustomFeature', and 'Target'. 