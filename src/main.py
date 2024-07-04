import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
import sys 
import warnings

warnings.simplefilter(action='ignore', category=Warning)

# Učitavanje podataka
dataset_path = "../data/AmesHousing.csv"
data = pd.read_csv(dataset_path)
show_plot = False

# Čišćenje podataka
# Uklanjanje kolona sa previše nedostajućih vrednosti
data.drop(columns=['Alley', 'Pool QC', 'Fence', 'Misc Feature'], inplace=True)

# Popunjavanje nedostajućih vrednosti
data['Lot Frontage'].fillna(data['Lot Frontage'].median(), inplace=True)
data['Garage Yr Blt'].fillna(data['Garage Yr Blt'].median(), inplace=True)
data.fillna('None', inplace=True)

# Kodiranje kategorijskih promenljivih
data = pd.get_dummies(data)

# Definisanje ulaznih (X) i izlaznih (y) podataka
X = data.drop(columns='SalePrice')
y = data['SalePrice']

# Podela podataka na trening i test setove
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Skaliranje podataka
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Kreiranje modela
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5)
}

# Treniranje i evaluacija modela
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    list_y_pred = list(y_pred)
    list_y_test = list(y_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}:")
    print(f"  RMSE: {rmse}")
    print(f"  MAE: {mae}")
    print(f"  R^2: {r2}")

# Opciona vizualizacija rezultata
import matplotlib.pyplot as plt

if show_plot:
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, models['Random Forest'].predict(X_test), alpha=0.3)
    plt.xlabel('Stvarne cene')
    plt.ylabel('Predviđene cene')
    plt.title('Stvarne naspram predviđenih cena (Random Forest)')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
    plt.show()
