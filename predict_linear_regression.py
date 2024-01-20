from prediction.utils import train_linear_regression, make_prediction
import pandas as pd
import numpy as np

data = {'feature1': [1, 2, 3, 4, 5], 'feature2': [2, 3, 4, 5, 6], 'target': [3, 4, 5, 6, 7]}
df = pd.DataFrame(data)
features = df[['feature1', 'feature2']]
target = df['target']
trained_model = train_linear_regression(features, target)
new_data = np.array([[6, 7]])
feature_names = features.columns.tolist()
prediction = make_prediction(trained_model, new_data, feature_names)
print(f'Prediction: {prediction}')
