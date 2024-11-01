import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
url = "D:\\Marketingcampaigns.csv"
df = pd.read_csv(url)

# Data exploration
print(df.columns)
pd.set_option('display.max_columns', None)
print(df.head())
print(df.info)
print(df.describe())
# for counting the number of missing values for each column
print(df.isnull().sum())
# Removing duplicates
print(df.drop_duplicates())

# Preprocessing
# Identify target & features (i.e feature selection/ Engineering)
x = df[['Age', 'Gender', 'Location', 'Email Opened',
       'Email Clicked', 'Product page visit', 'Discount offered']]
y = df['Purchased']
# Encoding categorical variable from the independent variable/feature
x = pd.get_dummies(x, columns=['Location'], drop_first=True)
'''# Check for multi collinearity by using correlation matrix
correlation_matrix = x.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()'''
# Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)  # Fit and transform on the training set
x_test = scaler.transform(x_test)        # Only transform on the test set
model = LogisticRegression()
# parameters selection using Grid Search
param_grid = {'C': [0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],  # Optimization algorithm
    'max_iter': [3000,5000,9000]}  # Maximum number of iterations
grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                           scoring='accuracy', cv=5, n_jobs=-1)
grid_search.fit(x_train, y_train)
best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)
print("Best cross-validated score:", grid_search.best_score_)

# Fitting & predicting
best_model.fit(x_train, y_train)
y_pred =best_model.predict(x_test)
print(y_pred)

# Model Evaluation
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Visualize confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()