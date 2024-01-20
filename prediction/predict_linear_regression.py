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
from io import BytesIO
import base64

def perform_prediction_and_visualization(csv_file_path, target_variable):
    df = pd.read_csv(csv_file_path, sep=';')
    X = df.drop(target_variable, axis=1, inplace=False)
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    y_test_numeric = pd.to_numeric(y_test, errors='coerce')

    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])

    categorical_features = X.select_dtypes(include=['object']).columns
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),('cat', categorical_transformer, categorical_features)])

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),('regressor', LinearRegression())])

    pipeline.fit(X_train, y_train)

    predictions = make_prediction(pipeline, X_test)
    print(f'Predictions on the test set: {predictions}')
    

    # Data Visualization
    # Scatter plot
    scatter_plot_base64 = ''
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=predictions)
    plt.title(f'Scatter Plot of Actual vs. Predicted {target_variable}')
    plt.xlabel(f'Actual {target_variable}')
    plt.ylabel(f'Predicted {target_variable}')
    scatter_plot_buffer = BytesIO()
    plt.savefig(scatter_plot_buffer, format='png')
    scatter_plot_base64 = base64.b64encode(scatter_plot_buffer.getvalue()).decode('utf-8')
    plt.close()

    residuals = y_test_numeric - predictions

    # Residual plot
    residual_plot_base64 = ''
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test_numeric, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel(f'Actual {target_variable}')
    plt.ylabel('Residuals')
    residual_plot_buffer = BytesIO()
    plt.savefig(residual_plot_buffer, format='png')
    residual_plot_base64 = base64.b64encode(residual_plot_buffer.getvalue()).decode('utf-8')
    plt.close()

    # Distribution plot for y_test
    distribution_actual_base64 = ''
    plt.figure(figsize=(10, 6))
    sns.countplot(x=y_test)
    plt.title(f'Distribution of Actual {target_variable}')
    plt.xlabel(f'Actual {target_variable}')
    plt.ylabel('Count')
    distribution_actual_buffer = BytesIO()
    plt.savefig(distribution_actual_buffer, format='png')
    distribution_actual_base64 = base64.b64encode(distribution_actual_buffer.getvalue()).decode('utf-8')
    plt.close()

    # Distribution plot for predictions
    distribution_predicted_base64 = ''
    plt.figure(figsize=(10, 6))
    sns.kdeplot(predictions, label=f'Predicted {target_variable}', shade=True)
    plt.title(f'Distribution Plot of Predicted {target_variable}')
    plt.xlabel(f'{target_variable}')
    plt.legend()
    distribution_predicted_buffer = BytesIO()
    plt.savefig(distribution_predicted_buffer, format='png')
    distribution_predicted_base64 = base64.b64encode(distribution_predicted_buffer.getvalue()).decode('utf-8')
    plt.close()

    download_links = {
        'Scatter Plot': scatter_plot_base64,
        'Residual Plot': residual_plot_base64,
        'Distribution of Actual MPG': distribution_actual_base64,
        'Distribution Plot of Predicted MPG': distribution_predicted_base64
    }

    return {
        'predictions': predictions,
        'residuals': residuals,
        'download_links': download_links,
        'scatter_plot_base64': scatter_plot_base64,
        'residual_plot_base64': residual_plot_base64,
        'distribution_actual_base64': distribution_actual_base64,
        'distribution_predicted_base64': distribution_predicted_base64,
    }
