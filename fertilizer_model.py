import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder # preprocessing
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv("fertilizer.csv")

X = df[["Temperature", "Humidity", "Moisture", "Soil Type", "Crop Type", "Nitrogen", "Potassium", "Phosphorous"]]
y = df["Fertilizer"]

# *X

# *y.head(30)

le = LabelEncoder() # so that y, which are strings, are machine understandable
y_encoded = le.fit_transform(y)
#* y_encoded

pipe = Pipeline([
  ("preprocessor", ColumnTransformer(
    transformers=[
      ('cat', OneHotEncoder(), ['Soil Type', 'Crop Type']),
      ('num', StandardScaler(), ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous'])
    ],
    remainder='passthrough' # if col not mentioned in transformers, ignore
  )),
  ("model", RandomForestClassifier())
])

param_grid = [
  {
    'model': [DecisionTreeClassifier()],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5, 10]
  },
  {
    'model': [XGBClassifier(n_jobs=-1)],
    'model__n_estimators': [50, 100, 150],
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__max_depth': [3, 5, 7]
  },
  {
    'model': [RandomForestClassifier()],
    'model__n_estimators': [50, 100, 150],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10]
  }
]


mod = GridSearchCV(
  estimator = pipe,
  cv=5,
  param_grid=param_grid,
  n_jobs=-1
)

mod.fit(X, y_encoded)

#* pd.DataFrame(mod.cv_results_)

#* print(mod.best_estimator_)

#* mod.best_params_

#* mod.best_score_

# new_data = pd.DataFrame({
#   "Temperature": [31],
#   "Humidity": [62],
#   "Moisture": [44],
#   "Soil Type": ["Sandy"],
#   "Crop Type": ["Barley"],
#   "Nitrogen": [21],
#   "Potassium": [20],
#   "Phosphorous": [2.5]
# })

#* new_data

# predictions = mod.predict(new_data)
#print(predictions[0]) # predictions return array, we access the first ele

# decoded_predictions = le.inverse_transform(predictions)

#decoded_predictions

pickle.dump(mod,open('mod.pkl','wb'))
pickle.dump(le,open('le.pkl','wb'))