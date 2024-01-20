from prediction.utils import make_prediction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression

csv_file_path = 'prediction/cars.csv'
df = pd.read_csv(csv_file_path, sep=';')
print("Dataset Headers:")
print(df.head())

X = df.drop('Acceleration', axis=1, inplace=False)
y = df['Acceleration']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
y_test_numeric = pd.to_numeric(y_test, errors='coerce')
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_features = X.select_dtypes(include=['object']).columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                             ('regressor', LinearRegression())])

pipeline.fit(X_train, y_train)
predictions = make_prediction(pipeline, X_test)
print(f'Predictions on the test set: {predictions}')

#Data Visualization
# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=predictions)
plt.title('Scatter Plot of Actual vs. Predicted MPG')
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
scatter_plot_filename = 'scatter_plot.png'
plt.savefig(scatter_plot_filename)
plt.close()

residuals = y_test_numeric - predictions

# Residual plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_numeric, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Actual MPG')
plt.ylabel('Residuals')
residual_plot_filename = 'residual_plot.png'
plt.savefig(residual_plot_filename)
plt.close()

# Distribution plot for y_test
plt.figure(figsize=(10, 6))
sns.countplot(x=y_test)
plt.title('Distribution of Actual MPG')
plt.xlabel('Actual MPG')
plt.ylabel('Count')
distribution_actual_filename = 'distribution_actual.png'
plt.savefig(distribution_actual_filename)
plt.close()

# Distribution plot for predictions
plt.figure(figsize=(10, 6))
sns.kdeplot(predictions, label='Predicted MPG', shade=True)
plt.title('Distribution Plot of Predicted MPG')
plt.xlabel('MPG')
plt.legend()
distribution_predicted_filename = 'distribution_predicted.png'
plt.savefig(distribution_predicted_filename) 
plt.close()

download_links = {
    'Scatter Plot': scatter_plot_filename,
    'Residual Plot': residual_plot_filename,
    'Distribution of Actual MPG': distribution_actual_filename,
    'Distribution Plot of Predicted MPG': distribution_predicted_filename
}

for title, filename in download_links.items():
    print(f'Download {title}: <a href="{filename}" download>{filename}</a>')
