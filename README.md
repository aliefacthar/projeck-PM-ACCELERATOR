# projeck-PM-ACCELERATOR
Data scince and Forecasting Project
Introduction
This project focuses on cleaning, exploring, and analyzing data to extract valuable insights. The goal is to build robust forecasting models to predict future trends and create visually appealing dashboards to support data-driven decision-making. The project is designed to help individuals acquire data analysis and forecasting skills applicable to any domain.

Objectives
To clean and preprocess raw data for accurate analysis.
To perform Exploratory Data Analysis (EDA) to uncover trends and patterns in the data.
To develop forecasting models to predict key metrics (e.g., sales, growth rates).
To provide actionable insights through advanced analysis and visualizations.
To present findings in a clear, structured report or dashboard.
Dataset Information
The dataset used in this project contains the following columns:

Date: The date of observation.
Country/Region: Geographical region of the data.
Confirmed: Number of confirmed cases.
Deaths: Number of deaths reported.
Recovered: Number of recovered cases.
Active: Number of active cases at a given time.
The dataset was cleaned to remove missing values, handle outliers, and ensure consistency for analysis.

Key Insights
Confirmed cases and active cases exhibit seasonality, making them predictable with time-series models.
Recovery rates vary significantly across regions, highlighting the importance of localized interventions.
Advanced models such as ARIMA and Prophet accurately predict trends with low error margins.
Visualizations effectively showcase trends, anomalies, and key drivers of changes in the data.
Technologies Used
Python: Pandas, NumPy, Matplotlib, Seaborn, Statsmodels, and Prophet for data analysis and forecasting.
Power BI: For interactive dashboard creation.
GitHub: For version control and collaboration.
Contact Information

Link presentation = https://www.canva.com/design/DAGdXIsxiiE/jC--8UUSBMRfTlFsb5Rgmw/edit?utm_content=DAGdXIsxiiE&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton
Documentation = https://drive.google.com/drive/folders/1ninoDssWCfP7US1YTgCWybRr1Wgb1gY6?usp=drive_link

For questions or collaboration, please contact:
Name: Alief
Email: aliefabdur8@gmail.com
LinkedIn:(https://www.linkedin.com/in/abdurrahman-alief-acthar-66aa08314/)


code 
import pandas as pd 
data = pd.read_csv(r"C:\Users\TOSHIBA\Documents\projeck data analis\word\GlobalWeatherRepository.csv")
data.isnull().sum()

data.duplicated().sum()
data.dtypes

data.head(3)
data.columns
data['last_updated'] = pd.to_datetime(data['last_updated'])
data['sunrise'] = pd.to_datetime(data['sunrise'])
data['sunset '] = pd.to_datetime(data['sunset'])
data['moonrise'] = pd.to_datetime(data['moonrise'], errors = 'coerce')
data['moonset'] = pd.to_datetime(data['moonset'], errors='coerce')

import seaborn as sns

# Plot temperature and precipitation trends
plt.figure(figsize=(14, 7))

# Temperature trend
plt.subplot(2, 1, 1)
sns.lineplot(x='last_updated', y='temperature_celsius', data=data, marker='o', color='b')
plt.title('Temperature Trend (Celsius)')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')

# Precipitation trend
plt.subplot(2, 1, 2)
sns.lineplot(x='last_updated', y='precip_mm', data=data, marker='o', color='g')
plt.title('Precipitation Trend (mm)')
plt.xlabel('Time')
plt.ylabel('Precipitation (mm)')

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))
scatter = sns.scatterplot(x='temperature_celsius', y='precip_mm', hue='condition_text', data=data, palette='coolwarm', s=100)

# Highlight specific points
highlight = data[data['precip_mm'] > 50]  # Example: precipitation greater than 50 mm
for i in range(len(highlight)):
    plt.text(highlight['temperature_celsius'].iloc[i] + 0.2, 
             highlight['precip_mm'].iloc[i], 
             highlight['condition_text'].iloc[i], fontsize=10)

plt.title('Correlation Between Temperature and Precipitation')
plt.xlabel('Temperature (°C)')
plt.ylabel('Precipitation (mm)')
plt.legend(title='Weather Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# Hitung ringkasan pola
summary = pd.DataFrame({
    'Suhu Maks': pivot['temperature_celsius'].max(axis=1),
    'Suhu Min': pivot['temperature_celsius'].min(axis=1),
    'Rentang Suhu': pivot['temperature_celsius'].max(axis=1) - pivot['temperature_celsius'].min(axis=1),
    'Curah Hujan Maks': pivot['precip_mm'].max(axis=1),
    'Curah Hujan Min': pivot['precip_mm'].min(axis=1),
    'Rentang Curah Hujan': pivot['precip_mm'].max(axis=1) - pivot['precip_mm'].min(axis=1),
})

# Tambahkan pola unik berdasarkan kombinasi rentang suhu dan curah hujan
summary['Unik'] = summary['Rentang Suhu'] + summary['Rentang Curah Hujan']
summary_sorted = summary.sort_values('Unik', ascending=False)

# Ambil 5 lokasi paling unik
top_5_unique = summary_sorted.head(5)
print(top_5_unique)


# Filter unique locations in the pivot table
unique_locations = top_5_unique.index
filtered_pivot_unique = pivot.loc[unique_locations, :]

plt.figure(figsize=(20, 10))

# Heatmap for temperature in unique locations
plt.subplot(1, 2, 1)
sns.heatmap(filtered_pivot_unique['temperature_celsius'], cmap='coolwarm', annot=True, fmt='.1f', cbar=True, linewidths=0.5)
plt.title('Temperature Pattern for Unique Locations (Monthly)', fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Location', fontsize=12)

# Heatmap for precipitation in unique locations
plt.subplot(1, 2, 2)
sns.heatmap(filtered_pivot_unique['precip_mm'], cmap='Blues', annot=True, fmt='.1f', cbar=True, linewidths=0.5)
plt.title('Precipitation Pattern for Unique Locations (Monthly)', fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Location', fontsize=12)

plt.tight_layout()
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# For suppressing warnings
import warnings
warnings.filterwarnings('ignore')

numeric_data = data.select_dtypes(include=['number'])
data_daily = numeric_data.resample('D').mean()  # Resampling daily


test_series = test_series.fillna(0)  # Ganti NaN dengan 0
forecast = pd.Series(forecast).fillna(0)  # Ganti NaN dengan 0


import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Contoh DataFrame (sesuaikan dengan dataset Anda)
# Pastikan kolom `lastupdated` berformat datetime, dan target adalah nilai yang ingin diprediksi.
data = pd.DataFrame({
    'lastupdated': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'value': np.random.randint(50, 150, size=100)
})

# 1. **Preprocessing**
data['lastupdated'] = pd.to_datetime(data['lastupdated'])  # Pastikan kolom datetime
data.set_index('lastupdated', inplace=True)  # Atur lastupdated sebagai index
data = data.sort_index()  # Pastikan data terurut berdasarkan waktu

# 2. **Train-Test Split**
# Gunakan 80% data untuk training dan sisanya untuk testing
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]

# 3. **Build Forecasting Model**
# Menggunakan Holt-Winters (Exponential Smoothing) untuk forecasting
model = ExponentialSmoothing(
    train['value'],
    seasonal=None,  # Pilihan: None, 'add', atau 'mul'
    trend='add',  # Pilihan: None, 'add', atau 'mul'
    seasonal_periods=None  # Tidak ada seasonality di data ini
).fit()

# Forecasting untuk periode test
forecast = model.forecast(len(test))

# 4. **Evaluate the Model**
# Hitung metrik evaluasi
mae = mean_absolute_error(test['value'], forecast)
rmse = np.sqrt(mean_squared_error(test['value'], forecast))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# 5. **Visualize the Results**
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(train.index, train['value'], label='Training Data')
plt.plot(test.index, test['value'], label='Actual Test Data')
plt.plot(test.index, forecast, label='Forecast', linestyle='--')
plt.legend()
plt.title('Forecasting using Exponential Smoothing')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()


import pandas as pd
from scipy.stats import zscore

# Assuming 'df' is the DataFrame containing your data
df = pd.read_csv(r'C:\Users\TOSHIBA\Documents\projeck data analis\word\GlobalWeatherRepository.csv')  # Replace with your actual data loading method

# Numerical columns to check for outliers
numeric_columns = ['temperature_celsius', 'humidity', 'wind_mph', 'pressure_mb', 'precip_mm']

# Calculate Z-scores for these columns
z_scores = df[numeric_columns].apply(zscore)

# Define a threshold for outliers (e.g., Z-score greater than 3 or less than -3)
outliers = (z_scores > 3) | (z_scores < -3)

# Display the rows that have outliers
outlier_data = df[outliers.any(axis=1)]
print(outlier_data)


from sklearn.ensemble import IsolationForest

# Fit an Isolation Forest model
model = IsolationForest(contamination=0.05)  # Adjust contamination as needed
df['anomaly'] = model.fit_predict(df[numeric_columns])

# -1 indicates an outlier, 1 indicates normal data
outlier_data_ml = df[df['anomaly'] == -1]
print(outlier_data_ml)


import seaborn as sns
import matplotlib.pyplot as plt

# Box plot for a variable (e.g., temperature)
sns.boxplot(data=df, x='temperature_celsius')
plt.show()

# Scatter plot for two variables (e.g., wind speed vs temperature)
sns.scatterplot(x='wind_mph', y='temperature_celsius', data=df, hue='anomaly')
plt.show()


data.columns

# Convert 'last_updated' to datetime if it's not already in datetime format
data['last_updated'] = pd.to_datetime(data['last_updated'], errors='coerce')

# Set 'last_updated' as the index
data.set_index('last_updated', inplace=True)

# Focus on the temperature column (e.g., 'temperature_celsius')
data = data[['temperature_celsius']]

# Drop rows with missing temperature values (if any)
data = data.dropna()

# Resample the data to daily frequency (use 'D' for daily)
data = data.resample('D').mean()

# Split the data into training and test sets (80% train, 20% test)
train = data.iloc[:-30]  # Use all data except the last 30 days for training
test = data.iloc[-30:]   # Use the last 30 days for testing

# Check the result
print(data.head())

# Split the data into training and test sets (80% train, 20% test)
train = data_daily.iloc[:-30]  # Use all data except the last 30 days for training
test = data_daily.iloc[-30:]   # Use the last 30 days for testing

# Verify the split
print(f"Training set shape: {train.shape}")
print(f"Test set shape: {test.shape}")


from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Use 'temperature_celsius' as the target variable for forecasting
train_temp = train['temperature_celsius']
test_temp = test['temperature_celsius']

# Fit ARIMA model
arima_model = ARIMA(train_temp, order=(5, 1, 0))  # Example order (p, d, q)
arima_model_fit = arima_model.fit()

# Forecast with ARIMA model
arima_forecast = arima_model_fit.forecast(steps=len(test_temp))

# Calculate RMSE for ARIMA
arima_rmse = mean_squared_error(test_temp, arima_forecast, squared=False)
print(f"ARIMA RMSE: {arima_rmse}")


from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Use 'temperature_celsius' as the target variable for forecasting
train_temp = train['temperature_celsius']
test_temp = test['temperature_celsius']

# Fit ARIMA model
arima_model = ARIMA(train_temp, order=(5, 1, 0))  # Example order (p, d, q)
arima_model_fit = arima_model.fit()

# Forecast with ARIMA model
arima_forecast = arima_model_fit.forecast(steps=len(test_temp))

# Calculate RMSE for ARIMA
arima_rmse = mean_squared_error(test_temp, arima_forecast, squared=False)
print(f"ARIMA RMSE: {arima_rmse}")


# Data RMSE
models = ['ARIMA', 'Linear Regression', 'Random Forest', 'Ensemble']
rmse_values = [11.953182339060994, 11.189016471190559, 13.764775872918417, 11.604163107906919]

# Membuat visualisasi bar chart
plt.figure(figsize=(10, 6))
plt.bar(models, rmse_values, color=['blue', 'green', 'red', 'orange'])

# Menambahkan judul dan label
plt.title('Comparison of RMSE for Different Models', fontsize=14)
plt.xlabel('Model', fontsize=12)
plt.ylabel('RMSE', fontsize=12)

# Menambahkan angka pada setiap bar
for i, value in enumerate(rmse_values):
    plt.text(i, value + 0.2, f'{value:.2f}', ha='center', fontsize=12)

# Menampilkan plot
plt.tight_layout()
plt.show()


import seaborn as sns

# Select relevant columns for correlation analysis
correlation_data = data[['air_quality_PM2.5', 'air_quality_Carbon_Monoxide', 'air_quality_Ozone',
                          'temperature_celsius', 'wind_mph', 'humidity']]

# Compute correlation matrix
correlation_matrix = correlation_data.corr()

# Visualize the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title("Correlation between Air Quality and Weather Parameters")
plt.show()


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define features (X) and target (y)
X = data[['humidity', 'wind_mph', 'pressure_mb', 'precip_mm', 'air_quality_PM2.5', 'air_quality_Carbon_Monoxide']]
y = data['temperature_celsius']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Get feature importance
feature_importances = model.feature_importances_
features = X.columns

# Plot feature importance
plt.barh(features, feature_importances)
plt.title("Feature Importance for Temperature Prediction")
plt.xlabel("Importance")
plt.show()


import plotly.express as px
import pandas as pd

# Assuming 'data' is your DataFrame containing 'longitude' and 'latitude' columns

# Create a scatter plot on a geographical map
fig = px.scatter_geo(data,
                     lat='latitude',  # Replace with the correct column name for latitude
                     lon='longitude',  # Replace with the correct column name for longitude
                     color='temperature_celsius',  # Optional: color by temperature or any other feature
                     size_max=10,  # Adjust point size
                     title="Geographical Distribution of Temperature")

# Customize the figure size for better visibility
fig.update_layout(
    autosize=False,
    width=1000,  # Set the desired width
    height=800,  # Set the desired height
)

# Show the plot
fig.show()


# Aggregate data by country and calculate average temperature and air quality
country_data = data.groupby('country').agg({'temperature_celsius': 'mean', 'air_quality_PM2.5': 'mean'}).reset_index()

# Sort by temperature or air quality (you can choose whichever metric is more relevant for you)
top_10_countries = country_data.sort_values(by='temperature_celsius', ascending=False).head(10)

# Plot the data for top 10 countries with a smaller figure size
fig, ax = plt.subplots(figsize=(8, 6))  # Adjusted size for smaller plot
top_10_countries.set_index('country')[['temperature_celsius', 'air_quality_PM2.5']].plot(kind='bar', ax=ax)
plt.title("Top 10 Countries: Average Temperature and Air Quality")
plt.xlabel("Country")
plt.ylabel("Average Values")
plt.xticks(rotation=45)  # Slightly reduce rotation for better readability
plt.show()




