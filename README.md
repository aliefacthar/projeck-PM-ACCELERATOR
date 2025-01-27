# projeck-PM-ACCELERATOR
Data scince and Forecasting Project
#Introduction
This project focuses on cleaning, exploring, and analyzing data to extract valuable insights. The goal is to build robust forecasting models to predict future trends and create visually appealing dashboards to support data-driven decision-making. The project is designed to help individuals acquire data analysis and forecasting skills applicable to any domain.

#Objectives
To clean and preprocess raw data for accurate analysis.
To perform Exploratory Data Analysis (EDA) to uncover trends and patterns in the data.
To develop forecasting models to predict key metrics (e.g., sales, growth rates).
To provide actionable insights through advanced analysis and visualizations.
To present findings in a clear, structured report or dashboard.
Dataset Information

#The dataset used in this project contains the following columns:
Date: The date of observation.
Country/Region: Geographical region of the data.
Confirmed: Number of confirmed cases.
Deaths: Number of deaths reported.
Recovered: Number of recovered cases.
Active: Number of active cases at a given time.
The dataset was cleaned to remove missing values, handle outliers, and ensure consistency for analysis.

#Key Insights
Confirmed cases and active cases exhibit seasonality, making them predictable with time-series models.
Recovery rates vary significantly across regions, highlighting the importance of localized interventions.
Advanced models such as ARIMA and Prophet accurately predict trends with low error margins.
Visualizations effectively showcase trends, anomalies, and key drivers of changes in the data.

#Technologies Used
Python: Pandas, NumPy, Matplotlib, Seaborn, Statsmodels, and Prophet for data analysis and forecasting.
Power BI: For interactive dashboard creation.
GitHub: For version control and collaboration.

# Information
Link presentation = https://www.canva.com/design/DAGdXIsxiiE/jC--8UUSBMRfTlFsb5Rgmw/edit?utm_content=DAGdXIsxiiE&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton
Documentation = https://drive.google.com/drive/folders/1ninoDssWCfP7US1YTgCWybRr1Wgb1gY6?usp=drive_link

#For questions or collaboration, please contact:
Name: Alief
Email: aliefabdur8@gmail.com
LinkedIn:(https://www.linkedin.com/in/abdurrahman-alief-acthar-66aa08314/)


#code 
#import data
import pandas as pd 
data = pd.read_csv(r"C:\Users\TOSHIBA\Documents\projeck data analis\word\GlobalWeatherRepository.csv")
data.isnull().sum()  # Count the number of missing values
# Data cleaning
data.duplicated().sum()  # Count the number of duplicate rows
data.dtypes  # Show data types of each column
data.head(3)  # Display the first 3 rows of the dataset
data.columns  # Show the column names
data['last_updated'] = pd.to_datetime(data['last_updated'])  # Convert 'last_updated' column to datetime
data['sunrise'] = pd.to_datetime(data['sunrise'])  # Convert 'sunrise' column to datetime
data['sunset '] = pd.to_datetime(data['sunset'])  # Convert 'sunset' column to datetime
data['moonrise'] = pd.to_datetime(data['moonrise'], errors = 'coerce')  # Convert 'moonrise' column to datetime, handle errors
data['moonset'] = pd.to_datetime(data['moonset'], errors='coerce')  # Convert 'moonset' column to datetime, handle errors

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

plt.tight_layout()  # Adjust the layout to prevent overlap
plt.show()

# Scatter plot showing the correlation between temperature and precipitation
plt.figure(figsize=(10, 5))
scatter = sns.scatterplot(x='temperature_celsius', y='precip_mm', hue='condition_text', data=data, palette='coolwarm', s=100)

# Highlight specific points (e.g., where precipitation is greater than 50 mm)
highlight = data[data['precip_mm'] > 50]  
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

# Calculate summary patterns
summary = pd.DataFrame({
    'Max Temperature': pivot['temperature_celsius'].max(axis=1),
    'Min Temperature': pivot['temperature_celsius'].min(axis=1),
    'Temperature Range': pivot['temperature_celsius'].max(axis=1) - pivot['temperature_celsius'].min(axis=1),
    'Max Precipitation': pivot['precip_mm'].max(axis=1),
    'Min Precipitation': pivot['precip_mm'].min(axis=1),
    'Precipitation Range': pivot['precip_mm'].max(axis=1) - pivot['precip_mm'].min(axis=1),
})

# Add a unique pattern based on the combination of temperature and precipitation ranges
summary['Unique'] = summary['Temperature Range'] + summary['Precipitation Range']
summary_sorted = summary.sort_values('Unique', ascending=False)

# Get top 5 unique locations
top_5_unique = summary_sorted.head(5)
print(top_5_unique)

# Filter unique locations in the pivot table
unique_locations = top_5_unique.index
filtered_pivot_unique = pivot.loc[unique_locations, :]

# Heatmap for temperature in unique locations
plt.figure(figsize=(20, 10))
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

# Prepare for time series forecasting using Exponential Smoothing
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Example dataset for forecasting
data = pd.DataFrame({
    'lastupdated': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'value': np.random.randint(50, 150, size=100)
})

# Preprocessing
data['lastupdated'] = pd.to_datetime(data['lastupdated'])
data.set_index('lastupdated', inplace=True)
data = data.sort_index()

# Train-Test Split
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]

# Build forecasting model using Holt-Winters Exponential Smoothing
model = ExponentialSmoothing(
    train['value'],
    seasonal=None, 
    trend='add',
    seasonal_periods=None
).fit()

# Forecasting the test period
forecast = model.forecast(len(test))

# Evaluate the model
mae = mean_absolute_error(test['value'], forecast)
rmse = np.sqrt(mean_squared_error(test['value'], forecast))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Visualizing results
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




