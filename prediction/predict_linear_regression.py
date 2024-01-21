from prediction.utils import make_prediction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from django.conf import settings


def perform_prediction_and_visualization(csv_file_path, target_variable):
    df = pd.read_csv(csv_file_path, sep=';')
    X = df.drop(target_variable, axis=1, inplace=False)
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    y_test_numeric = pd.to_numeric(y_test, errors='coerce')

    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])

    categorical_features = X.select_dtypes(include=['object']).columns
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                               ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                                   ('cat', categorical_transformer, categorical_features)])

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])

    pipeline.fit(X_train, y_train)

    predictions = make_prediction(pipeline, X_test)
    print(f'Predictions on the test set: {predictions}')

    # Data Visualization
    residuals = y_test_numeric - predictions
    residuals_list = residuals.tolist()
    folder_name = os.path.join(settings.STATIC, 'images')

    scatter_plot_path = os.path.join(folder_name, 'scatter_plot.png')
    residual_plot_path = os.path.join(folder_name, 'residual_plot.png')
    distribution_actual_path = os.path.join(folder_name, 'distribution_actual.png')
    distribution_predicted_path = os.path.join(folder_name, 'distribution_predicted.png')

    # Scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test_numeric, y=predictions)
    plt.title(f'Scatter Plot of Actual vs. Predicted {target_variable}')
    plt.xlabel(f'Actual {target_variable}')
    plt.ylabel(f'Predicted {target_variable}')
    plt.savefig(scatter_plot_path)
    plt.close()

    # Residual plot
    residuals = y_test_numeric - predictions
    residuals_list = residuals.tolist()  # Convert residuals to a Python list
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test_numeric, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel(f'Actual {target_variable}')
    plt.ylabel('Residuals')
    plt.savefig(residual_plot_path)
    plt.close()

    # Distribution plot for y_test
    plt.figure(figsize=(10, 6))
    sns.countplot(x=y_test)
    plt.title(f'Distribution of Actual {target_variable}')
    plt.xlabel(f'Actual {target_variable}')
    plt.ylabel('Count')
    plt.savefig(distribution_actual_path)
    plt.close()

    # Distribution plot for predictions
    plt.figure(figsize=(10, 6))
    sns.kdeplot(predictions, label=f'Predicted {target_variable}', fill=True)
    plt.title(f'Distribution Plot of Predicted {target_variable}')
    plt.xlabel(f'{target_variable}')
    plt.legend()
    plt.savefig(distribution_predicted_path)
    plt.close()

    # Update the image URLs to reflect the new folder structure
    scatter_plot_url = os.path.join(settings.STATIC_URL, 'images', 'scatter_plot.png')
    residual_plot_url = os.path.join(settings.STATIC_URL, 'images', 'residual_plot.png')
    distribution_actual_url = os.path.join(settings.STATIC_URL, 'images', 'distribution_actual.png')
    distribution_predicted_url = os.path.join(settings.STATIC_URL, 'images', 'distribution_predicted.png')

    download_links = {
        'Scatter Plot': scatter_plot_url,
        'Residual Plot': residual_plot_url,
        'Distribution of Actual MPG': distribution_actual_url,
        'Distribution Plot of Predicted MPG': distribution_predicted_url
    }

    return {
        'predictions': predictions.tolist(),
        'residuals': residuals_list,
        'download_links': download_links,
    }